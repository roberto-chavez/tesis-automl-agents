"""Motor de búsqueda: Optuna (TPE) + Hyperband por candidato (R2.1, R2.3, R4.1, R4.2)."""
from __future__ import annotations

from pathlib import Path

import optuna
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from . import trainer
from .catalog import CATALOG
from .config import RunConfig
from .dataset import DatasetProfile, split_files
from .events import RunLogger
from .resources import ResourceMonitor

optuna.logging.set_verbosity(optuna.logging.WARNING)


class SearchEngine:
    def __init__(self, cfg: RunConfig, profile: DatasetProfile, files: dict, logger: RunLogger):
        self.cfg, self.profile, self.log = cfg, profile, logger
        self.env = trainer.configure_tf()
        self.splits = split_files(files, profile.classes, cfg.val_fraction, cfg.test_fraction, cfg.seed)
        self.models_dir = cfg.run_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self.best_per_candidate: dict[str, float] = {}
        self._ds: dict = {}
        y = np.array(self.splits["train"][1])
        self.class_weight = None
        if profile.imbalance_ratio > 1.5:
            w = compute_class_weight("balanced", classes=np.arange(profile.n_classes), y=y)
            self.class_weight = {i: float(x) for i, x in enumerate(w)}

    def base_ds(self, split: str, cand):
        key = (split, cand.architecture, cand.img_size)
        if key not in self._ds:
            p, y = self.splits[split]
            self._ds[key] = trainer.load_base_dataset(p, y, cand.img_size, CATALOG[cand.architecture])
        return self._ds[key]

    def run_round(self, round_idx: int, candidates, n_trials: int) -> list[dict]:
        cfg, new = self.cfg, []
        for cand in candidates:
            spec = CATALOG[cand.architecture]
            n_train = len(self.splits["train"][0])
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=cfg.seed + round_idx),
                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=cfg.max_epochs,
                                                      reduction_factor=3),
            )

            def objective(trial, cand=cand, spec=spec):
                params = {
                    "lr": trial.suggest_categorical("lr", cand.lr_values),
                    "dropout": trial.suggest_categorical("dropout", cand.dropout_values),
                    "batch": trial.suggest_categorical("batch", cand.batch_sizes),
                    "augment": trial.suggest_categorical("augment", cand.augment_values),
                }
                ds_tr = trainer.batched(self.base_ds("train", cand), params["batch"], n_train, True,
                                        cfg.seed + trial.number)
                ds_va = trainer.batched(self.base_ds("val", cand), 64, 0, False, 0)
                with ResourceMonitor() as mon:
                    model, res = trainer.train_trial(
                        spec, params, self.profile.n_classes, cand.img_size, ds_tr, ds_va,
                        cfg.max_epochs, cfg.pretrained, self.class_weight, trial,
                        seed=cfg.seed + trial.number)
                rec = {"round": round_idx, "candidate_id": cand.id, "architecture": cand.architecture,
                       "trial": trial.number, "params": params, **res, "resources": mon.usage.as_dict()}
                if not res["pruned"] and res["val_acc"] > self.best_per_candidate.get(cand.id, -1):
                    self.best_per_candidate[cand.id] = res["val_acc"]
                    model.save(self.models_dir / f"{cand.id}.keras")
                    rec["saved_as_best"] = True
                self.records.append(rec)
                new.append(rec)
                self.log.emit("train", (
                    f"[ronda {round_idx}] {cand.architecture} trial {trial.number}: "
                    f"val_acc={res['val_acc']:.3f}, épocas={res['epochs']}"
                    f"{' (podado)' if res['pruned'] else ''}, {mon.usage.seconds:.1f}s"),
                    candidate=cand.id, val_acc=res["val_acc"], pruned=res["pruned"])
                trainer.release(model)
                if res["pruned"]:
                    raise optuna.TrialPruned()
                return res["val_acc"]

            study.optimize(objective, n_trials=n_trials, catch=())
        return new
