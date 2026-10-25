"""Trayectorias observables con herramientas y SQLite reales, sin entrenamiento pesado."""
from dataclasses import replace
import json
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

from automl_agents.config import RunConfig
from automl_agents.events import CancellationToken
from automl_agents.kb import Doc
from automl_agents.orchestrator import Orchestrator, LLMPolicy, RulesPolicy, decision
from automl_agents.policies import BudgetManager
from automl_agents.schemas import Candidate, Plan, TrialResult
from automl_agents.state_store import StateStore
from automl_agents.tools import DomainTools


class Engine:
    def __init__(self, cfg, store, outcomes):
        self.cfg, self.store, self.outcomes = cfg, store, iter(outcomes)
        self.budget = BudgetManager(store, cfg)
        self.cancellation = CancellationToken()
        self.interrupted = False
        self.stop_reason = None
        self.calls = 0
        self.curves = {}

    def run_block(self, candidates, max_attempts, action_key):
        self.calls += 1
        c = candidates[0]
        reservation = self.budget.reserve(training_attempts=1)
        try:
            score = next(self.outcomes, .5)
            ref = {**c.reference.to_dict(), "study_id": "study", "trial_id": uuid4().hex,
                   "attempt_id": uuid4().hex}
            params = {"lr": c.lr_values[0], "dropout": c.dropout_values[0], "batch": 16, "augment": False}
            meta = {**ref, "params": params, "seed": 42, "round": 0, "trial": self.calls,
                    "architecture": c.architecture}
            self.store.begin_attempt(action_key, meta)
            checkpoint = self.cfg.run_dir / (ref["attempt_id"] + ".keras")
            checkpoint.write_text("checkpoint")
            result = TrialResult(reference=ref, status="failed" if score is None else "completed",
                                 parameters=params, seed=42, curves=self.curves,
                                 metrics={} if score is None else {"accuracy": score},
                                 checkpoint=None if score is None else str(checkpoint),
                                 error={"code": "out_of_memory", "message": "OOM", "recoverable": True} if score is None else None)
            self.store.confirm_attempt(self.store.make_record(meta, result))
            return [result]
        finally:
            self.budget.confirm(reservation)


@pytest.fixture
def make_tools(tmp_path):
    opened = []
    def make(outcomes=(.99,), **config):
        cfg = RunConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), run_id=uuid4().hex,
                        img_size_override=32, pretrained=False, trials_per_round=1,
                        max_training_attempts=12, **config)
        store = StateStore(cfg.run_dir)
        opened.append(store)
        c = Candidate("one", "MobileNetV3Small", "test", [], [.001], [0.], [16], [False], 32)
        store.put("tool_candidates", {c.id: c.to_dict()})
        kb = SimpleNamespace(query=lambda *a: [Doc("guide", "Guide", "Text", "guide", .8)])
        engine = Engine(cfg, store, outcomes)
        return DomainTools(cfg, store, kb=kb, engine=engine)
    yield make
    for store in opened:
        store.close()


def actions(tools):
    return [h["result"]["tool"] for h in tools.store.get("orchestrator")["history"]]


def test_observations_change_trajectory_and_verify_plan(make_tools):
    good = make_tools([.99], use_llm=False)
    failed = make_tools([None], use_llm=False)
    assert Orchestrator(good).run() == {"status": "closed", "reason": "target_reached"}
    assert Orchestrator(failed).run()["status"] == "no_model"
    assert actions(good) == ["train_block", "close_search"]
    assert actions(failed) == ["train_block", "diagnose_trial", "discard_candidate", "close_search"]
    plan = Plan.model_validate(failed.store.get("plan"))
    for task in plan.tasks:
        assert task.status == "completed"
        assert failed.store.get("orchestrator_result:" + task.result_id)["status"] == "completed"
    assert plan.tasks[-1].dependencies == [plan.tasks[-2].task_id]


def test_stagnation_budget_and_selection_immutable(make_tools):
    t = make_tools([.6] * 8, use_llm=False, max_stagnant_blocks=2)
    assert Orchestrator(t).run()["reason"] == "stagnation"
    assert t.engine.calls == 3
    selected = t.store.get("selection")
    t.store.put("test_opened", True)
    t.budget.close_search("different")
    assert t.store.get("selection") == selected
    assert t.store.get("stop_reason") == "stagnation"
    assert Orchestrator(t).run()["status"] == "closed"
    assert t.engine.calls == 3
    result = t.call("train_block", {"candidates": [{"candidate_id": "one", "candidate_version": 1}],
                                  "max_attempts": 1}, role="orchestrator", action_id="after-test")
    assert result.error.code == "phase_denied"
    limited = make_tools([.5], use_llm=False, max_decisions=1)
    assert Orchestrator(limited).run()["reason"] == "budget_exhausted"
    assert limited.engine.calls == 1


def test_revision_uses_grid_and_new_version(make_tools):
    t = make_tools([.6, .99], use_llm=False)
    c = Candidate.from_dict(t.store.get("tool_candidates")["one"])
    c = replace(c, lr_values=[.0001, .0003, .001, .003, .01])
    t.store.put("tool_candidates", {c.id: c.to_dict()})
    assert Orchestrator(t).run()["reason"] == "target_reached"
    assert actions(t) == ["train_block", "revise_candidate", "train_block", "close_search"]
    assert t.store.get("selection")["candidate_version"] == 2
    assert "one:v1" in t.store.get("candidate_history")


class Repeater:
    def decide(self, observation):
        return decision("retrieve", "Consultar", query="same")


def test_llm_recovers_evidence_and_reads_reference_after_training_failure(make_tools):
    t = make_tools([None, .99])
    class Client:
        available = True
        last_usage = None
        step = 0
        def complete_json(self, system, user):
            o = json.loads(user)["observation"]
            assert "checkpoint" not in user and "curves" not in o
            self.step += 1
            if self.step in (1, 4):
                return decision("train", "Obtener validación", candidates=[
                    {"candidate_id": "one", "candidate_version": 1}], max_attempts=1).to_dict()
            assert o["records"]["recent"][-1]["status"] == "failed"
            if self.step == 2:
                return decision("retrieve", "Consultar guía tras OOM", query="memoria GPU batch",
                                kinds=["guide"]).to_dict()
            aid = o["evidence_references"][0]["artifact_id"]
            return decision("read", "Leer fragmento de la guía", artifact_id=aid, length=100).to_dict()
    client = Client()
    assert Orchestrator(t, LLMPolicy(client, t.cfg, t.store)).run()["reason"] == "target_reached"
    assert actions(t) == ["train_block", "query_evidence", "read_artifact", "train_block", "close_search"]
    history = t.store.get("orchestrator")["history"]
    assert history[2]["result"]["data"]["text"] == '"Text"'
    assert len(t.store.get("rag_ledger")) == 1
    assert all(c["context"]["estimated"] for c in t.store.get("llm_calls"))


def test_repeated_and_alternating_actions_terminate_without_progress(make_tools):
    t = make_tools(use_llm=False)
    assert Orchestrator(t, Repeater()).run()["reason"] == "no_progress"
    assert actions(t) == ["query_evidence", "close_search"]
    class Alternating:
        index = 0
        def decide(self, observation):
            self.index += 1
            return decision("retrieve", "Consultar", query=str(self.index))
    t = make_tools(use_llm=False, max_no_progress_actions=3)
    Orchestrator(t, Alternating()).run()
    assert actions(t).count("query_evidence") == 3
    assert t.store.get("stop_reason") == "no_progress"


@pytest.mark.parametrize("failure", [RuntimeError("provider down"), {"bad": "json"}])
def test_provider_failure_and_invalid_format_fall_back_with_shared_budget(make_tools, failure):
    t = make_tools([.99], max_retries=1)
    class Client:
        available = True
        last_usage = None
        def complete_json(self, *args):
            if isinstance(failure, Exception):
                raise failure
            return failure
    result = Orchestrator(t, LLMPolicy(Client(), t.cfg, t.store)).run()
    assert result["reason"] == "target_reached"
    assert t.budget.snapshot()["llm_calls"] == 2
    assert t.budget.snapshot()["retries"] == 1
    assert len(t.store.get("llm_calls")) == 2
    assert t.store.get("policy_failures")


def test_timeout_does_not_wait_for_provider_or_retry_in_background(make_tools):
    t = make_tools([.99], llm_timeout_seconds=.02)
    release = threading.Event()
    class Client:
        available = True
        def complete_json(self, *args):
            release.wait(2)
            return {}
    try:
        assert Orchestrator(t, LLMPolicy(Client(), t.cfg, t.store)).run()["reason"] == "target_reached"
        assert t.store.get("llm_calls")[0]["error"] == "TimeoutError"
        assert t.budget.snapshot()["llm_calls"] == 1
    finally:
        release.set()


def test_simulated_client_adapts_to_observations_and_logs_usage(make_tools):
    trajectories = []
    for score in (.99, None):
        t = make_tools([score])
        class Client:
            available = True
            last_usage = {"total_token_count": 12}
            def complete_json(self, system, user):
                o = json.loads(user)["observation"]
                assert "checkpoint" not in user and "test_secret" not in user
                if not o["records"]["count"]:
                    return decision("train", "Explorar", candidates=[{"candidate_id": "one", "candidate_version": 1}], max_attempts=1).to_dict()
                if not o["diagnoses"]["count"]:
                    return decision("diagnose", "Inspeccionar fallo", attempt_id=o["records"]["recent"][-1]["attempt_id"]).to_dict()
                return decision("stop", "Sin modelo", reason="no_model").to_dict()
        t.store.put("test_secret", "never include")
        Orchestrator(t, LLMPolicy(Client(), t.cfg, t.store)).run()
        assert t.store.get("llm_calls")[0]["usage"] == {"total_token_count": 12}
        trajectories.append(actions(t))
    assert trajectories == [["train_block", "close_search"], ["train_block", "diagnose_trial", "close_search"]]


@pytest.mark.parametrize("tool", ["train_block", "close_search"])
def test_crash_after_tool_confirmation_replays_without_effects(make_tools, monkeypatch, tool):
    t = make_tools([.99], use_llm=False)
    call = t.call
    def crash(name, *args, **kwargs):
        result = call(name, *args, **kwargs)
        if name == tool:
            raise SystemExit(17)
        return result
    monkeypatch.setattr(t, "call", crash)
    with pytest.raises(SystemExit):
        Orchestrator(t).run()
    assert t.store.get("orchestrator")["pending"]
    monkeypatch.setattr(t, "call", call)
    assert Orchestrator(t).run()["status"] == "closed"
    assert t.engine.calls == 1
    assert all(task["status"] == "completed" for task in t.store.get("plan")["tasks"])


def test_failed_tool_is_not_completed_and_rules_take_over(make_tools):
    t = make_tools([.99], use_llm=False)
    class Invalid:
        def decide(self, observation):
            return decision("adjust", "Revisión inválida", expected={"candidate_id": "one", "candidate_version": 5},
                            candidate=t.store.get("tool_candidates")["one"])
    assert Orchestrator(t, Invalid()).run()["status"] == "closed"
    assert t.store.get("plan")["tasks"][0]["status"] == "failed"
    assert t.store.get("tool_candidates")["one"]["version"] == 1


def test_diagnosis_delegates_once_and_orchestrator_applies_validated_revision(make_tools):
    t = make_tools([.6, .99])
    original = Candidate.from_dict(t.store.get("tool_candidates")["one"])
    updated = replace(original, version=2, dropout_values=[.2])
    t.engine.curves = {"accuracy": [.95], "val_accuracy": [.6]}

    class SpecialistClient:
        available = True
        calls = 0
        def complete_json(self, system, user):
            self.calls += 1
            context = json.loads(user)["context"]
            assert context["curves"]["summary"]["val_accuracy"]["last"] == .6
            assert context["curves"]["artifact_id"]
            assert "profile" not in context and "checkpoint" not in context
            return {"hypotheses": [{"code": "overfitting", "explanation": "Compatible con la brecha observada."}],
                    "recommendation": {"action": "adjust", "reason": "Aumentar regularización.",
                                       "expected": original.reference.to_dict(), "candidate": updated.to_dict()},
                    "sources": []}

    t.llm = SpecialistClient()
    assert Orchestrator(t, RulesPolicy(t.cfg)).run()["reason"] == "target_reached"
    assert actions(t) == ["train_block", "diagnose_trial", "revise_candidate", "train_block", "close_search"]
    assert t.llm.calls == 1 and t.store.get("specialist_usage") == {"diagnosis": 1}
    assert t.store.get("selection")["candidate_version"] == 2
    assert t.store.get("tool_candidates")["one"]["dropout_values"] == [.2]


def test_tampered_or_stale_specialist_advice_is_ignored(make_tools):
    from automl_agents.specialists.contracts import DiagnosisContext
    from automl_agents.specialists.diagnosis import diagnose_rules

    t = make_tools([.6, .99], use_llm=False)
    c = Candidate.from_dict(t.store.get("tool_candidates")["one"])
    t.engine.run_block([c], 1, action_key="initial")
    record = t.store.records()[0]
    context = DiagnosisContext.build(record, c)
    basic = diagnose_rules(context).to_dict()
    for change in ({"version": 4}, {"batch_sizes": [1]}, {"sources": ["unknown"]}):
        diagnostic = {**basic, "recommendation": {"action": "adjust", "reason": "Invalid advice",
                      "expected": c.reference.to_dict(), "candidate": {**c.to_dict(), "version": 2, **change}}}
        t.store.put("diagnoses", {record["attempt_id"]: diagnostic})
        orchestrator = Orchestrator(t)
        assert orchestrator._specialist_decision(orchestrator.observe()) is None
        assert t.store.get("tool_candidates")[c.id] == c.to_dict()
    diagnostic["recommendation"]["candidate"] = replace(c, version=2).to_dict()
    t.store.put("diagnoses", {record["attempt_id"]: diagnostic})
    t.store.put("candidate_history", {"one:v1": c.to_dict()})
    t.store.put("tool_candidates", {c.id: replace(c, version=2).to_dict()})
    assert Orchestrator(t)._specialist_decision(Orchestrator(t).observe()) is None

def test_live_provider_smoke(make_tools):
    """Opt-in: una búsqueda mínima usa el proveedor configurado y datos sintéticos."""
    import os
    from automl_agents.llm import LLM
    if os.getenv("AUTOML_TEST_LIVE_LLM") != "1":
        pytest.skip("Set AUTOML_TEST_LIVE_LLM=1 to contact the configured provider")
    import time
    from pathlib import Path

    t = make_tools([.99], max_llm_calls=5, max_retries=0)
    client = LLM(t.cfg.llm_model, timeout_seconds=t.cfg.llm_timeout_seconds)
    if not client.available:
        pytest.skip("Proveedor sin credenciales/SDK disponible")
    started = time.monotonic()
    outcome, validated = None, False
    try:
        outcome = Orchestrator(t, LLMPolicy(client, t.cfg, t.store)).run()
        calls = t.store.get("llm_calls", [])
        assert calls
        assert not t.store.get("policy_failures"), t.store.get("policy_failures")
        assert all(c["error"] is None for c in calls)
        assert any(c["usage"] for c in calls)
        assert t.engine.calls >= 1, "Gemini debe solicitar entrenamiento mediante una decisión válida"
        assert outcome["status"] == "closed"
        assert t.store.get("selection") is not None
        assert all(h["verified"] for h in t.store.get("orchestrator")["history"])
        validated = True
    finally:
        report = {"model": t.cfg.llm_model, "timeout_seconds": t.cfg.llm_timeout_seconds,
                  "validated": validated, "outcome": outcome, "seconds": time.monotonic() - started,
                  "training": "simulated", "training_blocks": t.engine.calls,
                  "calls": t.store.get("llm_calls", []),
                  "policy_failures": t.store.get("policy_failures", []),
                  "actions": [h["result"]["tool"] for h in t.store.get("orchestrator", {}).get("history", [])]}
        report_path = os.getenv("AUTOML_LIVE_LLM_REPORT")
        if report_path:
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True))
