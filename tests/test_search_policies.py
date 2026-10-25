"""Integración de las políticas con servicios reales y entrenamiento simulado."""
import pytest

pytest.importorskip("tensorflow")

from test_search_blocks import engine, candidate, simulated_training
from automl_agents.policies import PolicyError
from automl_agents.pipeline import _CountedLLM
from automl_agents.state_store import StateStore
from automl_agents.tools import DomainTools


def test_exhausted_attempt_budget_never_calls_training_and_replay_is_free(engine, monkeypatch):
    engine.store.put("limits", {**engine.store.get("limits"), "max_training_attempts": 1})
    simulated_training(engine, monkeypatch)
    first = engine.run_block([candidate()], 1, action_key="first")
    monkeypatch.setattr(engine, "execute_trial", lambda *a, **kw: pytest.fail("Entrenamiento sin presupuesto"))
    assert engine.run_block([candidate()], 1, action_key="first") == first
    assert engine.run_block([candidate()], 1, action_key="new") == []
    assert len(engine.store.attempts()) == 1
    assert engine.stop_reason == "budget_exhausted"


def test_test_is_unavailable_even_indirectly_through_engine(engine):
    with pytest.raises(PolicyError, match="Test"):
        engine.base_ds("test", candidate())


def test_new_space_cannot_replay_action_for_previous_version(engine, monkeypatch):
    simulated_training(engine, monkeypatch)
    engine.run_block([candidate()], 1, action_key="same")
    changed = candidate()
    changed.lr_values = [0.0003]
    second = engine.run_block([changed], 1, action_key="same")
    assert changed.version == 2
    assert second[0].reference.candidate_version == 2
    assert len(engine.store.attempts()) == 2


def test_oom_retry_reduces_batch_with_same_trial_and_spends_global_retry(engine, monkeypatch):
    c = candidate()
    c.batch_sizes = [16, 32]
    engine._study(c).enqueue_trial({"lr": .001, "dropout": 0., "batch": 32, "augment": False})
    simulated_training(engine, monkeypatch, [MemoryError("OOM"), MemoryError("OOM"), "completed"])
    results = engine.run_block([c], 3, action_key="oom")
    assert [r.parameters["batch"] for r in results[:2]] == [32, 16]
    assert results[0].reference.trial_id == results[1].reference.trial_id
    assert results[0].reference.attempt_id != results[1].reference.attempt_id
    assert results[2].reference.trial_id != results[1].reference.trial_id
    assert engine.budget.snapshot()["retries"] == 1
    assert engine.budget.snapshot()["training_attempts"] == 3


def test_tool_last_decision_can_execute_admitted_training(engine, monkeypatch):
    engine.cfg.img_size_override = 32
    engine.store.put("limits", {**engine.store.get("limits"), "max_decisions": 1})
    c = candidate()
    engine.store.put("tool_candidates", {c.id: c.to_dict()})
    service = DomainTools(engine.cfg, engine.store, engine=engine)
    simulated_training(engine, monkeypatch)
    result = service.call("train_block", {"candidates": [c.reference.to_dict()], "max_attempts": 1},
                          role="orchestrator", action_id="train")
    assert result.status == "completed", result.error
    assert len(result.data["results"]) == 1
    assert engine.budget.snapshot()["decisions"] == 1


def test_llm_failure_is_charged_and_rejected_before_provider_on_resume(engine):
    engine.store.put("limits", {**engine.store.get("limits"), "max_llm_calls": 1})
    class Client:
        available = True
        calls = 0

        def complete(self, *a):
            self.calls += 1
            raise OSError("provider down")
    client = Client()
    counted = _CountedLLM(client, engine.store, engine.cfg)
    with pytest.raises(OSError):
        counted.complete("prompt")
    resumed_store = StateStore(engine.cfg.run_dir)
    counted = _CountedLLM(client, resumed_store, engine.cfg)
    with pytest.raises(PolicyError):
        counted.complete("retry")
    assert client.calls == 1
    resumed_store.close()
