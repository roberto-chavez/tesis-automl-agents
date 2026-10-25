"""Grafo LangGraph que orquesta a los agentes (perfil → RAG → candidatas → búsqueda iterativa →
evaluación → benchmark → reportes → empaquetado)."""
from __future__ import annotations

import json
import traceback
import time
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import reporting
from .benchmark import compute_benchmark
from .config import RunConfig
from .dataset import prepare_dataset
from .events import RunLogger
from .kb import KnowledgeBase
from .llm import LLM
from .packaging import build_artifact
from .risk import assess_use_case
from .events import write_json, CancellationToken, PipelineCancelled
from .schemas import Candidate
from .state_store import StateStore, run_lock, load_run, training_lock
from .policies import BudgetManager, BudgetedLLM as _CountedLLM


class State(TypedDict, total=False):
    round: int
    stop: bool
    summary: dict


class Pipeline:
    """Contexto compartido entre nodos (objetos pesados fuera del estado del grafo)."""

    def __init__(self, cfg: RunConfig, kb: KnowledgeBase | None = None, llm: LLM | None = None):
        # Revalidar también configuraciones modificadas después de construirlas.
        cfg = RunConfig.from_dict(cfg.to_dict())
        self.cfg = cfg
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        config_path = cfg.run_dir / "config.json"
        if config_path.exists() and RunConfig.from_dict(json.loads(config_path.read_text("utf-8"))).to_dict() != cfg.to_dict():
            raise ValueError("La configuración difiere de la ejecución persistida")
        write_json(config_path, cfg.to_dict())
        self.store = StateStore(cfg.run_dir)
        self.store.reconcile()
        self.budget = BudgetManager(self.store, cfg)
        self.log = RunLogger(cfg.run_dir)
        self.llm = (llm if llm is not None else LLM(
            cfg.llm_model, timeout_seconds=cfg.llm_timeout_seconds)) if cfg.llm_enabled else LLM(
                cfg.llm_model, False, timeout_seconds=cfg.llm_timeout_seconds)
        self.llm = _CountedLLM(self.llm, self.store, cfg)
        self.kb = kb or KnowledgeBase(cfg.kb_persist_dir, cfg.embedding_backend)
        self.decisions: list[dict] = []
        self.candidates: list = []
        self._restore(self.store.get("context", {}))

    def _snapshot(self):
        context = {key: getattr(self, key) for key in (
            "risk", "decisions", "eval", "winner_id", "winner", "bench") if hasattr(self, key)}
        context["candidates"] = [c.to_dict() for c in self.candidates]
        if hasattr(self, "docs"):
            context["docs"] = [asdict(d) for d in self.docs]
        if hasattr(self, "prop"):
            context["prop"] = {**self.prop, "candidates": [c.to_dict() for c in self.prop["candidates"]]}
        return context

    def _restore(self, context):
        for key in ("risk", "decisions", "eval", "winner_id", "winner", "bench"):
            if key in context:
                setattr(self, key, context[key])
        self.candidates = [Candidate.from_dict(c) for c in context.get("candidates", [])]
        if "risk" in context and not hasattr(self, "profile"):
            self.profile, self.files, _ = prepare_dataset(
                self.cfg.dataset_dir, self.cfg.run_dir, self.cfg.val_fraction,
                self.cfg.test_fraction, self.cfg.seed)
        if "docs" in context:
            from .kb import Doc
            self.docs = [Doc(**doc) for doc in context["docs"]]
        if "prop" in context:
            self.prop = {**context["prop"], "candidates": [Candidate.from_dict(c) for c in context["prop"]["candidates"]]}
            if not hasattr(self, "engine"):
                from .search import SearchEngine
                self.engine = SearchEngine(self.cfg, self.profile, self.files, self.log)
            self.engine._restore_records()

    def _durable_node(self, name, fn):
        def execute(state):
            if CancellationToken(self.cfg.run_dir).is_cancelled() and not self.store.get("search_closed", False):
                raise PipelineCancelled()
            key = f"node:{name}:{state.get('round', 0)}" if name in ("entrenar", "controlar") else f"node:{name}"
            cached = self.store.action(key)
            if cached is not None:
                self._restore(cached["context"])
                self.store.put("context", cached["context"])
                return cached["update"]
            self.store.put("node", name)
            started = time.perf_counter()
            try:
                update = fn(state)
            finally:
                timings = self.store.get("stage_timings", {})
                timings[name] = timings.get(name, 0.) + time.perf_counter() - started
                self.store.put("stage_timings", timings)
            if update.get("summary"):
                reporting.write_persistent_reports(self.store, self.cfg)
            context = self._snapshot()
            # Una cancelación puede reanudarse en el mismo nodo y la misma ronda.
            if not (name == "entrenar" and update.get("summary")):
                self.store.confirm_action(key, {"update": update, "context": context})
            self.store.put("context", context)
            return update
        return execute

    # ---- nodos ----
    def n_profile(self, s: State) -> State:
        self.profile, self.files, _ = prepare_dataset(
            self.cfg.dataset_dir, self.cfg.run_dir, self.cfg.val_fraction,
            self.cfg.test_fraction, self.cfg.seed)
        self.risk = assess_use_case(self.cfg.use_case)
        self.store.put("training_profile", self.profile.to_dict())
        self.store.put("risk", self.risk)
        self.log.emit("perfil", f"Entrenamiento: {self.profile.n_images} imágenes, {self.profile.n_classes} clases",
                      profile=self.profile.to_dict(), risk=self.risk)
        return {}

    def n_retrieve(self, s: State) -> State:
        p = self.profile
        bucket = p.size_bucket()
        q = [f"dataset {'pequeño con pocas imágenes' if bucket == 'small' else 'grande con muchas imágenes' if bucket == 'large' else 'mediano'} de clasificación de imágenes",
             "rangos de tasa de aprendizaje y dropout", "tamaño de lote y memoria de GPU", "recursos computacionales limitados",
             "arquitecturas ligeras con transferencia de aprendizaje", "descarte temprano de configuraciones", "métricas de evaluación"]
        if p.imbalance_ratio > 1.5:
            q.append("clases desbalanceadas")
        if self.cfg.use_case:
            q.append(self.cfg.use_case)
        from .kb import Doc
        from .tools import DomainTools
        tools = DomainTools(self.cfg, self.store, kb=self.kb, lock_held=True)
        for index, query in enumerate(q):
            result = tools.call("query_evidence", {"query": query, "k": 3},
                                role="orchestrator", action_id=f"initial-evidence-{index}")
            if result.status != "completed":
                break
        self.docs = [Doc(**d) for d in self.store.get("tool_evidence", {}).values()]
        self.log.emit("rag", f"{len(self.docs)} documentos recuperados", docs=[d.to_dict() for d in self.docs])
        return {}

    def n_propose(self, s: State) -> State:
        self.store.put("phase", "search")
        from .tools import DomainTools
        # El pipeline ya posee run_lock; las llamadas no deben adquirirlo otra vez.
        self.store.put("tool_evidence", {d.id: asdict(d) for d in self.docs})
        tools = DomainTools(self.cfg, self.store, kb=self.kb, profile=self.profile,
                            llm=self.llm, lock_held=True)
        result = tools.call("propose_candidates", {}, role="orchestrator", action_id="initial-proposal")
        if result.status != "completed":
            self.budget.close_search(result.error.code)
            return {"summary": self._finish("no_model", result.error.code)}
        self.prop = self.store.get("proposal")
        self.prop["candidates"] = [Candidate.from_dict(c) for c in self.prop["candidates"]]
        self.candidates = list(self.prop["candidates"])
        self.store.put("proposal_evidence", [asdict(d) for d in self.docs])
        sp = self.prop["space"]
        reporting.write_criteria_report(self.cfg.run_dir, self.profile, self.prop, self.docs)
        self.log.emit("candidatas", f"{len(self.candidates)} candidatas ({self.prop['mode']}); espacio "
                      f"{sp['total_configs']}→{sp['candidate_configs']} (-{sp['reduction_pct']} %)",
                      candidates=[c.architecture for c in self.candidates])
        from .search import SearchEngine  # import diferido: TensorFlow es pesado
        self.engine = SearchEngine(self.cfg, self.profile, self.files, self.log)
        return {"round": 1, "stop": False}

    def n_train(self, s: State) -> State:
        from .tools import DomainTools
        from .orchestrator import Orchestrator
        tools = DomainTools(self.cfg, self.store, kb=self.kb, profile=self.profile,
                            engine=self.engine, llm=self.llm, lock_held=True)
        outcome = Orchestrator(tools).run()
        self.decisions = self.store.get("orchestrator", {}).get("history", [])
        self.engine._restore_records()
        if outcome["status"] != "closed":
            reason = outcome.get("reason") or {
                "cancelled": "user_cancelled", "interrupted": "process_interrupted"
            }.get(outcome["status"])
            summary = self._finish(outcome["status"], reason)
            return {"stop": True, "summary": summary}
        return {"stop": True}

    def n_control(self, s: State) -> State:
        # Compatibilidad del grafo persistente; las decisiones pertenecen al orquestador.
        return {"stop": True}

    def n_evaluate(self, s: State) -> State:
        from .evaluation import evaluate_frozen_selection
        try:
            final = evaluate_frozen_selection(self.store, self.engine)
        except Exception as exc:
            return {"summary": self._finish("artifact_failed", "final_evaluation_failed", error=exc)}
        self.winner = final["winner"]
        self.winner_id = self.winner["candidate_id"]
        self.eval = {self.winner_id: {"architecture": self.winner["architecture"],
                    "val_acc": self.winner["val_acc"], "test": final["metrics"]}}
        return {}

    def n_benchmark(self, s: State) -> State:
        self.bench = compute_benchmark(self.store.records(), self.cfg, self.store.get("proposal")["space"],
                                       self.store.get("stage_timings", {}))
        self.store.put("benchmark", self.bench)
        return {}

    def n_report(self, s: State) -> State:
        reporting.write_persistent_reports(self.store, self.cfg)
        self.log.emit("reportes", "Reportes generados")
        return {}

    def n_package(self, s: State) -> State:
        self.store.put("phase", "packaging")
        final = self.store.get("final_evaluation")
        try:
            art = build_artifact(self.cfg.run_dir, final["winner"], final["metrics"]["classes"], final["metrics"],
                                 smoke_timeout=self.cfg.artifact_smoke_timeout_seconds,
                                 docker_timeout=self.cfg.docker_timeout_seconds)
            verification = json.loads((art / "verification.json").read_text("utf-8"))
            self.store.put("verification", verification)
        except Exception as exc:
            return {"summary": self._finish("artifact_failed", "packaging_failed", error=exc)}
        target = final["winner"]["val_acc"] >= final["winner"]["metric_policy"]["threshold"]
        status = "artifact_failed" if verification["status"] == "failed" else "completed" if target else "target_not_reached"
        summary = self._finish(status, self.store.get("stop_reason"), artifact=art)
        self.log.emit("listo", f"Entrega: {status}; verificación: {verification['status']}")
        return {"summary": summary}

    def _finish(self, status, reason, *, artifact=None, error=None):
        """Conserva resultados útiles también sin modelo o con un artefacto fallido."""
        final = self.store.get("final_evaluation")
        proposal = self.store.get("proposal", {})
        bench = self.store.get("benchmark", {})
        summary = {"run_id": self.cfg.run_id, "status": status, "reason": reason,
                   "winner": final["winner"] if final else self.store.get("selection"),
                   "verification": self.store.get("verification", {"status": "not_run", "verified": False}),
                   "artifact": str(artifact) if artifact else None,
                   "space": proposal.get("space"), "mode": proposal.get("mode"),
                   "benchmark": {k: bench[k] for k in ("trials_total", "trials_pruned", "saving_from_early_stopping_pct") if k in bench}}
        if final:
            summary["test"] = {k: final["metrics"][k] for k in ("accuracy", "f1_macro", "n_test")}
        if error:
            summary["error"] = {"code": reason, "message": str(error), "recoverable": False}
            verification = {"status": "failed", "verified": False, "error": summary["error"]}
            self.store.put("verification", verification)
            summary["verification"] = verification
        self.store.put("delivery", summary)
        if status not in ("cancelled", "interrupted"):
            self.store.put("phase", "done")
        write_json(self.cfg.run_dir / "run_summary.json", summary)
        return summary

    def build(self, checkpointer=None):
        g = StateGraph(State)
        for name, fn in [("perfil", self.n_profile), ("rag", self.n_retrieve), ("candidatas", self.n_propose),
                         ("entrenar", self.n_train), ("controlar", self.n_control), ("evaluar", self.n_evaluate),
                         ("benchmark", self.n_benchmark), ("reportar", self.n_report), ("empaquetar", self.n_package)]:
            g.add_node(name, self._durable_node(name, fn) if hasattr(self, "store") else fn)
        g.add_edge(START, "perfil")
        g.add_edge("perfil", "rag")
        g.add_edge("rag", "candidatas")
        g.add_conditional_edges("candidatas", lambda s: END if s.get("summary") else "entrenar",
                                {END: END, "entrenar": "entrenar"})
        g.add_conditional_edges("entrenar", lambda s: END if s.get("summary") else "controlar",
                                {END: END, "controlar": "controlar"})
        g.add_conditional_edges("controlar", lambda s: "evaluar" if s["stop"] else "entrenar",
                                {"entrenar": "entrenar", "evaluar": "evaluar"})
        g.add_conditional_edges("evaluar", lambda s: END if s.get("summary") else "benchmark",
                                {END: END, "benchmark": "benchmark"})
        g.add_edge("benchmark", "reportar")
        g.add_edge("reportar", "empaquetar")
        g.add_edge("empaquetar", END)
        return g.compile(checkpointer=checkpointer)


def run_pipeline(cfg: RunConfig, kb: KnowledgeBase | None = None, llm: LLM | None = None, *,
                 resume: bool = False) -> dict:
    from langgraph.checkpoint.sqlite import SqliteSaver
    # Persistir rutas absolutas permite reanudar desde otro directorio de trabajo.
    cfg = RunConfig.from_dict({**cfg.to_dict(), "dataset_dir": str(Path(cfg.dataset_dir).resolve()),
                              "output_dir": str(Path(cfg.output_dir).resolve())})
    with training_lock(cfg.output_dir), run_lock(cfg.run_dir):
        # Una consulta/repetición de una entrega confirmada no necesita TensorFlow ni RAG.
        if (cfg.run_dir / "state.sqlite").exists():
            store = StateStore(cfg.run_dir)
            try:
                finished = store.action("node:empaquetar")
                if store.get("status") in ("completed", "target_not_reached", "artifact_failed") and finished is not None:
                    saved_config = RunConfig.from_dict(json.loads((cfg.run_dir / "config.json").read_text("utf-8"))).to_dict()
                    if saved_config != cfg.to_dict():
                        raise ValueError("La configuración difiere de la ejecución persistida")
                    delivery = store.get("delivery")
                    # Una pausa anterior no reemplaza la entrega confirmada al reanudar.
                    return delivery if delivery and delivery.get("status") == store.get("status") else finished["update"]["summary"]
            finally:
                store.close()
        pl = Pipeline(cfg, kb, llm)
        try:
            # Sólo una reanudación explícita revoca la cancelación anterior. Se
            # limpia bajo ambos bloqueos para no afectar a un proceso activo.
            if resume:
                (cfg.run_dir / "cancel.requested").unlink(missing_ok=True)
            with SqliteSaver.from_conn_string(str(cfg.run_dir / "graph.sqlite")) as saver:
                graph = pl.build(saver)
                config = {"recursion_limit": 100, "configurable": {"thread_id": cfg.run_id}}
                saved = graph.get_state(config)
                if saved.values.get("summary") and saved.values["summary"].get("status") not in ("interrupted", "cancelled"):
                    pl.store.put("status", saved.values["summary"].get("status", "completed"))
                    return saved.values["summary"]
                if saved.values.get("summary"):
                    graph.update_state(config, {"summary": None, "stop": False}, as_node="candidatas")
                pl.store.put("status", "running")
                out = graph.invoke(None if saved.values else {}, config=config)
                summary = out["summary"]
                pl.store.put("status", summary.get("status", "completed"))
                return summary
        except PipelineCancelled:
            summary = pl._finish("cancelled", "user_cancelled")
            pl.store.put("status", "cancelled")
            reporting.write_persistent_reports(pl.store, cfg)
            return summary
        except BaseException as exc:
            pl.store.put("status", "failed" if isinstance(exc, Exception) else "interrupted")
            pl.log.emit("error", f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
            raise
        finally:
            if hasattr(pl, "engine"):
                pl.engine.store.close()
            pl.store.close()


def resume_pipeline(run_id, runs_dir=None):
    cfg = load_run(run_id, runs_dir)
    if not (cfg.run_dir / "state.sqlite").exists():
        raise ValueError("Esta ejecución no tiene un journal SQLite recuperable")
    return run_pipeline(cfg, resume=True)
