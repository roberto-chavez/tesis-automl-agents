"""Entrenamiento secuencial por bloques y adaptación al coordinador de rondas."""
from __future__ import annotations

from collections import OrderedDict
import json
import time
from uuid import uuid4

import numpy as np
import optuna
from sklearn.utils.class_weight import compute_class_weight

from . import trainer
from .catalog import CATALOG
from .config import RunConfig
from .dataset import DatasetProfile, DatasetError, prepare_dataset
from .evaluation import evaluate_validation
from .events import CancellationToken, RunLogger, write_json
from .resources import ResourceMonitor
from .schemas import DomainError, TrialRef, TrialResult
from .state_store import StateStore, fingerprint
from .policies import BudgetManager, PolicyError, SearchCancellation

optuna.logging.set_verbosity(optuna.logging.WARNING)


class SearchEngine:
    def __init__(self, cfg: RunConfig, profile: DatasetProfile, files: dict, logger: RunLogger):
        self.cfg, self.log = cfg, logger
        self.profile, validated_files, self.splits = prepare_dataset(
            cfg.dataset_dir, cfg.run_dir, cfg.val_fraction, cfg.test_fraction, cfg.seed)
        if profile.classes != self.profile.classes or files != validated_files:
            raise DatasetError("El dataset cambió después del perfilado.", "dataset_changed")
        self.files = validated_files
        self.env = trainer.configure_tf()
        self.models_dir = cfg.run_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir = cfg.run_dir / "attempts"
        self.store = StateStore(cfg.run_dir)
        self.budget = BudgetManager(self.store, cfg)
        self.stop_reason = None
        self.records: list[dict] = []
        self.results: list[TrialResult] = []
        self.best_per_candidate: dict[str, float] = {}
        self.best_checkpoints: dict[str, str] = {}
        self.cancellation = CancellationToken(cfg.run_dir)
        self.interrupted = False
        self._studies = {}
        self._ds = OrderedDict()
        self._ds_bytes = 0
        self.class_weight = None
        self._restore_records()
        if self.profile.imbalance_ratio > 1.5:
            y = np.array(self.splits["train"][1])
            w = compute_class_weight("balanced", classes=np.arange(self.profile.n_classes), y=y)
            self.class_weight = {i: float(x) for i, x in enumerate(w)}

    def _restore_records(self):
        self.records = self.store.records()
        self.results = [TrialResult.model_validate(r["result"]) for r in self.records]
        self.best_per_candidate, self.best_checkpoints = {}, {}
        for rec in self.records:
            saved = (rec["status"] == "completed" and
                     rec["val_acc"] > self.best_per_candidate.get(rec["candidate_id"], -1))
            rec["saved_as_best"] = saved
            if saved:
                self.best_per_candidate[rec["candidate_id"]] = rec["val_acc"]
                self.best_checkpoints[rec["candidate_id"]] = rec["checkpoint"]

    def cancel(self):
        self.cancellation.cancel()

    def base_ds(self, split: str, cand):
        if split == "test" and not self.store.get("search_closed", False):
            raise PolicyError("phase_denied", "Test sólo está disponible después del cierre de búsqueda")
        key = (split, cand.architecture, cand.img_size)
        if key in self._ds:
            self._ds.move_to_end(key)
            return self._ds[key][0]
        p, y = self.splits[split]
        size = len(p) * cand.img_size ** 2 * 3 * 4
        candidate_bytes = sum(len(paths) for paths, _ in self.splits.values()) * cand.img_size ** 2 * 3 * 4
        cache = candidate_bytes < trainer.CACHE_LIMIT_BYTES
        cost = size if cache else 0
        # Límite agregado y de entradas: también acota pipelines sin caché en RAM.
        while self._ds and (len(self._ds) >= 2 or
                            self._ds_bytes + cost > trainer.CACHE_LIMIT_BYTES):
            _, (_, old_size) = self._ds.popitem(last=False)
            self._ds_bytes -= old_size
        ds = trainer.load_base_dataset(p, y, cand.img_size, CATALOG[cand.architecture], cache=cache)
        self._ds[key] = (ds, cost)
        self._ds_bytes += cost
        return ds

    def _resolve_space(self, cand):
        # También protege frente a un cambio de opciones sin incrementar version.
        signature = fingerprint({k: cand.to_dict()[k] for k in (
            "architecture", "img_size", "lr_values", "dropout_values", "batch_sizes", "augment_values")})
        versions = self.store.get("spaces", {})
        spaces = versions.setdefault(cand.id, {})
        if str(cand.version) in spaces and spaces[str(cand.version)] != signature:
            cand.version = next((int(v) for v, sig in spaces.items() if sig == signature),
                                max(map(int, spaces)) + 1)
        return signature, versions, spaces

    def _study(self, cand):
        signature, versions, spaces = self._resolve_space(cand)
        spaces[str(cand.version)] = signature
        self.store.put("spaces", versions)
        key = (cand.id, cand.version)
        if key not in self._studies:
            self._studies[key] = optuna.create_study(
                study_name=f"{cand.id}-v{cand.version}", direction=self.cfg.metric_policy.direction,
                storage=f"sqlite:///{(self.cfg.run_dir / 'optuna.sqlite').resolve().as_posix()}",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=self.cfg.seed),
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=1, max_resource=self.cfg.max_epochs, reduction_factor=3))
            # Cubre también una caída entre ask() y la reserva del intento.
            study = self._studies[key]
            records = {r["trial"]: r for r in self.store.records() if r["study_id"] == study.study_name}
            for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
                rec = records.get(trial.number)
                if rec and rec["status"] == "completed":
                    study.tell(trial.number, rec["val_acc"])
                else:
                    state = optuna.trial.TrialState.PRUNED if rec and rec["pruned"] else optuna.trial.TrialState.FAIL
                    study.tell(trial.number, state=state)
        return self._studies[key]

    @staticmethod
    def _error(exc):
        if isinstance(exc, (MemoryError, trainer.tf.errors.ResourceExhaustedError)):
            return DomainError(code="out_of_memory", message=str(exc) or "Memoria agotada", recoverable=True)
        if isinstance(exc, FloatingPointError):
            return DomainError(code="numerical_error", message=str(exc), recoverable=True)
        recoverable = isinstance(exc, (ValueError, OSError, trainer.tf.errors.OpError))
        return DomainError(code="training_error" if recoverable else "unexpected_error",
                           message=str(exc) or type(exc).__name__, recoverable=recoverable)

    def execute_trial(self, cand, study, trial, *, round_idx=0, cancellation=None,
                      slot=None, retry=None) -> TrialResult:
        """Ejecuta y publica un intento incluso cuando entrenamiento o evaluación fallan."""
        token = SearchCancellation(cancellation or self.cancellation, self.budget)
        reference = TrialRef(candidate_id=cand.id, candidate_version=cand.version,
                             study_id=study.study_name,
                             trial_id=retry["trial_id"] if retry else uuid4().hex, attempt_id=uuid4().hex)
        seed = retry["seed"] if retry else self.cfg.seed + trial.number
        directory = self.attempts_dir / reference.attempt_id
        params = {
            "lr": trial.suggest_categorical("lr", cand.lr_values),
            "dropout": trial.suggest_categorical("dropout", cand.dropout_values),
            "batch": trial.suggest_categorical("batch", cand.batch_sizes),
            "augment": trial.suggest_categorical("augment", cand.augment_values),
        }
        data = {name: json.loads((self.cfg.run_dir / 'data' / name).read_text('utf-8'))
                for name in ('dataset_manifest.json', 'split_manifest.json')}
        signature = fingerprint({"params": params, "seed": seed, "data": data,
                                 "architecture": cand.architecture, "size": cand.img_size,
                                 "pretrained": self.cfg.pretrained, "epochs": self.cfg.max_epochs})
        for previous in self.store.attempts():
            if previous["metadata"].get("fingerprint") == signature and previous["status"] == "completed":
                return TrialResult.model_validate(previous["record"]["result"])
        meta = {**reference.to_dict(), "params": params, "seed": seed,
                "round": round_idx, "trial": trial.number, "architecture": cand.architecture,
                "fingerprint": signature}
        reservation = self.budget.reserve(training_attempts=1, retries=int(retry is not None))
        self.store.begin_attempt(slot or reference.attempt_id, meta)
        progress, metrics, validation = {}, {}, {}
        model, checkpoint, error, fatal = None, None, None, None
        ds_tr = ds_va = None
        status, reason = "failed", "training_error"
        mon = ResourceMonitor()
        started = time.monotonic()

        def save_progress(data):
            write_json(directory / "curves.json", data.get("curves", {}))
            write_json(directory / "resources.json", {"seconds": round(time.monotonic() - started, 2)})

        try:
            with mon:
                write_json(directory / "parameters.json", {
                    "reference": reference.to_dict(), "parameters": params, "seed": seed,
                    "architecture": cand.architecture, "img_size": cand.img_size,
                    "pretrained": self.cfg.pretrained, "max_epochs": self.cfg.max_epochs})
                if token.is_cancelled():
                    status, reason = "cancelled", "cancelled"
                else:
                    ds_tr = trainer.batched(self.base_ds("train", cand), params["batch"],
                                           len(self.splits["train"][0]), True, seed)
                    ds_va = trainer.batched(self.base_ds("val", cand), 64, 0, False, 0)
                    model, res = trainer.train_trial(
                        CATALOG[cand.architecture], params, self.profile.n_classes, cand.img_size,
                        ds_tr, ds_va, self.cfg.max_epochs, self.cfg.pretrained, self.class_weight,
                        trial, seed=seed, cancellation=token, progress=progress, on_epoch=save_progress)
                    progress.update({k: res[k] for k in ("epochs", "curves")})
                    status = "cancelled" if res["cancelled"] else "pruned" if res["pruned"] else "completed"
                    reason = res["stop_reason"]
                    if progress["curves"].get("val_accuracy"):
                        metrics["accuracy"] = max(progress["curves"]["val_accuracy"])
                    if status == "completed":
                        validation = evaluate_validation(model, ds_va, self.splits["val"][1], self.profile.classes)
                        metrics = {k: validation[k] for k in ("accuracy", "f1_macro")}
                        path = self.models_dir / f"{reference.trial_id}_{reference.attempt_id}.keras"
                        temporary = path.with_name(path.stem + ".tmp.keras")
                        model.save(temporary)
                        temporary.replace(path)
                        checkpoint = str(path)
        except KeyboardInterrupt:
            status, reason = "interrupted", "keyboard_interrupt"
            self.interrupted = True
        except Exception as exc:
            error = self._error(exc)
            status, reason = "failed", error.code
            if not error.recoverable:
                fatal = exc
        finally:
            # Quitar referencias antes de limpiar incluso si fit falló sin devolver modelo.
            model = None
            ds_tr = ds_va = None
            if error is not None and error.code == "out_of_memory":
                self._ds.clear()
                self._ds_bytes = 0
            trainer.release()
            self.budget.confirm(reservation)

        result = TrialResult(reference=reference, status=status, parameters=params, metrics=metrics,
                             seed=seed, epochs=progress.get("epochs", 0), checkpoint=checkpoint,
                             error=error, curves=progress.get("curves", {}),
                             resources=mon.usage.as_dict(), stop_reason=reason, validation=validation)
        save_progress(progress)
        write_json(directory / "result.json", result.to_dict())
        self.results.append(result)
        saved = (status == "completed" and
                 metrics["accuracy"] > self.best_per_candidate.get(cand.id, -1))
        if saved:
            self.best_per_candidate[cand.id] = metrics["accuracy"]
            self.best_checkpoints[cand.id] = checkpoint
        # Vista compatible con reportes/controlador de rondas existentes.
        rec = {**reference.to_dict(), "round": round_idx, "architecture": cand.architecture,
               "trial": trial.number, "params": params, "val_acc": metrics.get("accuracy", 0.0),
               "epochs": result.epochs, "pruned": status == "pruned", "status": status,
               "resources": result.resources, "saved_as_best": saved, "checkpoint": checkpoint,
               "result": result.to_dict()}
        self.store.confirm_attempt(rec)
        self.records.append(rec)
        self.log.emit("train", f"{cand.architecture} trial {trial.number}: {status}",
                      **reference.to_dict(), status=status, result=result.to_dict(),
                      candidate=cand.id, val_acc=rec["val_acc"], pruned=rec["pruned"])
        if fatal is not None:
            raise fatal
        return result

    def run_block(self, candidates, max_attempts: int, *, round_idx=0,
                  cancellation=None, action_key=None) -> list[TrialResult]:
        """Máximo total del bloque, distribuido secuencialmente entre versiones candidatas."""
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
            raise ValueError("max_attempts debe ser un entero no negativo")
        candidates = list(candidates)
        if not candidates:
            return []
        token, results = SearchCancellation(cancellation or self.cancellation, self.budget), []
        block_key = action_key or uuid4().hex
        for index in range(max_attempts):
            if (cancellation or self.cancellation).is_cancelled() or self.interrupted:
                break
            cand = candidates[index % len(candidates)]
            recovery = None
            last = results[-1] if results else None
            if last and last.error and last.error.code == "out_of_memory":
                original = next((c for c in candidates if c.id == last.reference.candidate_id
                                 and c.version == last.reference.candidate_version), None)
                smaller = [b for b in original.batch_sizes if b < last.parameters["batch"]] if original else []
                related = [a for a in self.store.attempts()
                           if a["metadata"]["trial_id"] == last.reference.trial_id]
                if (smaller and len(related) == 1 and
                        self.budget.snapshot()["retries"] < self.store.get("limits")["max_retries"]):
                    cand = original
                    recovery = {**related[0]["metadata"],
                                "params": {**last.parameters, "batch": max(smaller)}}
            # Resolver la identidad sin crear un estudio ni gastar recursos.
            # Una rejilla incompatible no puede recuperar la acción de la versión anterior.
            self._resolve_space(cand)
            slot = f"{block_key}:{index}:{cand.id}:v{cand.version}"
            existing = self.store.action(slot)
            if existing is not None:
                results.append(TrialResult.model_validate(existing))
                continue
            previous = self.store.attempts(slot)
            # Si el resultado se confirmó antes de la caída, no volver a entrenar.
            if previous and previous[-1]["status"] not in ("running", "interrupted", "cancelled"):
                result = TrialResult.model_validate(previous[-1]["record"]["result"])
                self.store.confirm_action(slot, result.to_dict())
                results.append(result)
                continue
            retry = previous[-1]["metadata"] if previous else recovery
            # No crear siquiera un trial de Optuna si los límites ya impiden trabajo.
            try:
                self.budget.check()
            except PolicyError as exc:
                self.stop_reason = exc.error.code
                break
            study = self._study(cand)
            if retry:
                study.enqueue_trial(retry["params"])
            trial = study.ask()
            try:
                result = self.execute_trial(cand, study, trial, round_idx=round_idx, cancellation=token,
                                            slot=slot, retry=retry)
            except Exception as exc:
                committed = self.store.attempts(slot)
                record = committed[-1]["record"] if committed else None
                if record and record["trial"] == trial.number and record["status"] == "completed":
                    study.tell(trial, record["val_acc"])
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                if isinstance(exc, PolicyError):
                    self.stop_reason = exc.error.code
                    break
                raise
            if result.status == "completed":
                study.tell(trial, result.metrics["accuracy"])
            else:
                state = optuna.trial.TrialState.PRUNED if result.status == "pruned" else optuna.trial.TrialState.FAIL
                study.tell(trial, state=state)
            results.append(result)
            if result.status not in ("cancelled", "interrupted"):
                self.store.confirm_action(slot, result.to_dict())
            if result.status in ("cancelled", "interrupted"):
                break
        if self.budget.expired():
            self.stop_reason = "search_time_exhausted"
        return results

    def run_round(self, round_idx: int, candidates, n_trials: int) -> list[dict]:
        start = len(self.records)
        for cand in candidates:
            self.run_block([cand], n_trials, round_idx=round_idx,
                           action_key=f"round:{round_idx}:{cand.id}")
            if self.cancellation.is_cancelled() or self.interrupted:
                break
        return self.records[start:]
