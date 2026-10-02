"""Grafo LangGraph que orquesta a los agentes (perfil → RAG → candidatas → búsqueda iterativa →
evaluación → benchmark → reportes → empaquetado)."""
from __future__ import annotations

import json
import traceback
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import candidates as cand_mod
from . import controller, reporting
from .benchmark import compute_benchmark
from .catalog import CATALOG
from .config import RunConfig
from .dataset import profile_dataset
from .events import RunLogger
from .kb import KnowledgeBase
from .llm import LLM
from .packaging import build_artifact
from .risk import assess_use_case


class State(TypedDict, total=False):
    round: int
    stop: bool
    summary: dict


class Pipeline:
    """Contexto compartido entre nodos (objetos pesados fuera del estado del grafo)."""

    def __init__(self, cfg: RunConfig, kb: KnowledgeBase | None = None, llm: LLM | None = None):
        self.cfg = cfg
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        (cfg.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        self.log = RunLogger(cfg.run_dir)
        self.llm = llm if llm is not None else LLM(cfg.llm_model, cfg.use_llm)
        self.kb = kb or KnowledgeBase(cfg.kb_persist_dir, cfg.embedding_backend)
        self.decisions: list[dict] = []
        self.candidates: list = []

    # ---- nodos ----
    def n_profile(self, s: State) -> State:
        self.profile, self.files = profile_dataset(self.cfg.dataset_dir, seed=self.cfg.seed)
        self.risk = assess_use_case(self.cfg.use_case)
        self.log.emit("perfil", f"Dataset: {self.profile.n_images} imágenes, {self.profile.n_classes} clases",
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
        self.docs = self.kb.retrieve_many(q, k=3)
        self.log.emit("rag", f"{len(self.docs)} documentos recuperados", docs=[d.to_dict() for d in self.docs])
        return {}

    def n_propose(self, s: State) -> State:
        self.prop = cand_mod.propose(self.profile, self.docs, self.cfg, self.llm)
        self.candidates = list(self.prop["candidates"])
        sp = self.prop["space"]
        reporting.write_criteria_report(self.cfg.run_dir, self.profile, self.prop, self.docs)
        self.log.emit("candidatas", f"{len(self.candidates)} candidatas ({self.prop['mode']}); espacio "
                      f"{sp['total_configs']}→{sp['candidate_configs']} (-{sp['reduction_pct']} %)",
                      candidates=[c.architecture for c in self.candidates])
        from .search import SearchEngine  # import diferido: TensorFlow es pesado
        self.engine = SearchEngine(self.cfg, self.profile, self.files, self.log)
        return {"round": 1, "stop": False}

    def n_train(self, s: State) -> State:
        self.engine.run_round(s["round"], self.candidates, self.cfg.trials_per_round)
        return {}

    def n_control(self, s: State) -> State:
        r = s["round"]
        self.candidates, decs, stop = controller.decide(r, self.candidates, self.engine.records, self.cfg)
        self.decisions += decs
        for d in decs:
            self.log.emit("control", f"[ronda {r}] {d['candidate_id']}: {d['action']} — {d['reason']}")
        return {"round": r + 1, "stop": stop}

    def n_evaluate(self, s: State) -> State:
        from . import evaluation, trainer
        recs, ev = self.engine.records, {}
        for c in self.prop["candidates"]:
            path = self.cfg.run_dir / "models" / f"{c.id}.keras"
            if not path.exists():
                continue
            ds = trainer.batched(self.engine.base_ds("test", c), 64, 0, False, 0)
            m = evaluation.evaluate_model(path, ds, self.engine.splits["test"][1], self.profile.classes)
            ev[c.id] = {"architecture": c.architecture, "val_acc": round(self.engine.best_per_candidate[c.id], 4),
                        "test": m}
            self.log.emit("evaluacion", f"{c.architecture}: test acc={m['accuracy']:.3f}, F1={m['f1_macro']:.3f}")
        if not ev:
            raise RuntimeError("Ninguna configuración terminó el entrenamiento; revise el dataset o aumente el presupuesto.")
        self.eval = ev
        self.winner_id = max(ev, key=lambda k: (ev[k]["val_acc"], -CATALOG[ev[k]["architecture"]].params_m))
        best = max((r for r in recs if r.get("saved_as_best") and r["candidate_id"] == self.winner_id),
                   key=lambda r: r["val_acc"])
        cand = next(c for c in self.prop["candidates"] if c.id == self.winner_id)
        self.winner = {"candidate_id": self.winner_id, "architecture": cand.architecture,
                       "img_size": cand.img_size, "params": best["params"], "val_acc": best["val_acc"]}
        return {}

    def n_benchmark(self, s: State) -> State:
        self.bench = compute_benchmark(self.engine.records, self.cfg, self.prop["space"])
        return {}

    def n_report(self, s: State) -> State:
        rd = self.cfg.run_dir
        reporting.write_iterations(rd, self.decisions, self.engine.records)
        reporting.write_resources_csv(rd, self.engine.records)
        reporting.write_evaluation(rd, self.eval, self.winner_id)
        reporting.write_benchmark(rd, self.bench)
        reporting.write_final_report(rd, self.cfg, self.profile, self.prop, self.winner,
                                     self.eval[self.winner_id], self.engine.records, self.risk, self.bench, self.llm)
        self.log.emit("reportes", "Reportes generados")
        return {}

    def n_package(self, s: State) -> State:
        art = build_artifact(self.cfg.run_dir, self.winner, self.profile.classes, self.eval[self.winner_id]["test"])
        summary = {"run_id": self.cfg.run_id, "winner": self.winner, "test": {k: self.eval[self.winner_id]["test"][k] for k in ("accuracy", "f1_macro", "n_test")},
                   "space": self.prop["space"], "benchmark": {k: self.bench[k] for k in ("trials_total", "trials_pruned", "saving_from_early_stopping_pct")},
                   "mode": self.prop["mode"], "artifact": str(art)}
        (self.cfg.run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log.emit("listo", f"Modelo {self.winner['architecture']} listo (test acc={summary['test']['accuracy']:.3f})")
        return {"summary": summary}

    def build(self):
        g = StateGraph(State)
        for name, fn in [("perfil", self.n_profile), ("rag", self.n_retrieve), ("candidatas", self.n_propose),
                         ("entrenar", self.n_train), ("controlar", self.n_control), ("evaluar", self.n_evaluate),
                         ("benchmark", self.n_benchmark), ("reportar", self.n_report), ("empaquetar", self.n_package)]:
            g.add_node(name, fn)
        g.add_edge(START, "perfil")
        g.add_edge("perfil", "rag")
        g.add_edge("rag", "candidatas")
        g.add_edge("candidatas", "entrenar")
        g.add_edge("entrenar", "controlar")
        g.add_conditional_edges("controlar", lambda s: "evaluar" if s["stop"] else "entrenar",
                                {"entrenar": "entrenar", "evaluar": "evaluar"})
        g.add_edge("evaluar", "benchmark")
        g.add_edge("benchmark", "reportar")
        g.add_edge("reportar", "empaquetar")
        g.add_edge("empaquetar", END)
        return g.compile()


def run_pipeline(cfg: RunConfig, kb: KnowledgeBase | None = None, llm: LLM | None = None) -> dict:
    pl = Pipeline(cfg, kb, llm)
    try:
        out = pl.build().invoke({}, config={"recursion_limit": 100})
        return out["summary"]
    except Exception as e:
        pl.log.emit("error", f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        raise
