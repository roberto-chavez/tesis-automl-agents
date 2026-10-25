"""Rechazos antes de efectos, presupuestos compartidos y lecturas sin test."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from automl_agents.config import RunConfig
from automl_agents.policies import BudgetManager, PolicyError, SearchCancellation
from automl_agents.events import CancellationToken
from automl_agents.schemas import Candidate
from automl_agents.state_store import StateStore
from automl_agents.tools import DomainTools, REGISTRY


@pytest.fixture
def tools(tmp_path):
    cfg = RunConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), run_id="test",
                    use_llm=False, img_size_override=32, max_decisions=20)
    store = StateStore(cfg.run_dir)
    calls = []
    kb = SimpleNamespace(query=lambda *a: calls.append(a) or [SimpleNamespace(
        id="guide", title="Guide", text="safe evidence", kind="guide", score=1.,
        to_dict=lambda: {"id": "guide", "title": "Guide"})])
    service = DomainTools(cfg, store, kb=kb)
    service.calls = calls
    yield service
    store.close()


def call(tools, name, args=None, role="orchestrator", key="action"):
    return tools.call(name, args or {}, role=role, action_id=key)


def cand():
    return Candidate("c1", "MobileNetV3Small", "test", [], [0.001], [0.0], [16, 32], [False], 32)


@pytest.mark.parametrize("name,args,role,code", [
    ("query_evidence", {"query": ""}, "orchestrator", "invalid_input"),
    ("query_evidence", {"query": "x", "k": True}, "orchestrator", "invalid_input"),
    ("query_evidence", {"query": "x", "path": "test.json"}, "orchestrator", "invalid_input"),
    ("train_block", {"candidates": [], "max_attempts": 1}, "orchestrator", "invalid_input"),
    ("train_block", {}, "diagnosis", "permission_denied"),
    ("close_search", {"reason": "done"}, "configuration", "permission_denied"),
    ("read_artifact", {"artifact_id": "../test.json"}, "orchestrator", "invalid_input"),
    ("unknown", {}, "orchestrator", "invalid_input"),
])
def test_invalid_calls_have_no_effects(tools, name, args, role, code):
    result = call(tools, name, args, role)
    assert result.error.code == code
    assert not tools.calls
    assert tools.store.get("consumed") is None
    assert tools.store.db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 0


def test_phase_permissions_and_permanent_close(tools):
    tools.store.put("phase", "evaluation")
    assert call(tools, "query_evidence", {"query": "x"}, key="after-close").error.code == "phase_denied"
    tools.store.put("phase", "search")
    assert call(tools, "close_search", {"reason": "done"}).status == "completed"
    tools.store.put("phase", "search")  # Un error del host no reabre la búsqueda.
    assert call(tools, "query_evidence", {"query": "x"}, key="new-query").error.code == "phase_denied"
    assert not tools.calls


def test_idempotent_result_and_conflicting_key(tools):
    first = call(tools, "query_evidence", {"query": "x"})
    assert first.status == "completed"
    assert call(tools, "query_evidence", {"query": "x"}) == first
    assert len(tools.calls) == 1
    assert tools.budget.snapshot()["decisions"] == 1
    assert call(tools, "query_evidence", {"query": "other"}).error.code == "idempotency_conflict"
    assert call(tools, "query_evidence", {"query": "x"}, role="configuration").error.code == "idempotency_conflict"


def test_read_only_registered_safe_fragments(tools):
    first = call(tools, "query_evidence", {"query": "x"})
    aid = first.data["documents"][0]["artifact_id"]
    read = call(tools, "read_artifact", {"artifact_id": aid, "length": 5}, role="diagnosis", key="read")
    assert len(read.data["text"]) == 5
    tools.store.put("safe_artifact:secret", {"scope": "test", "roles": ["orchestrator"], "data": "secret"})
    for name in ("secret", "test", "dataset_manifest", "checkpoint"):
        result = call(tools, "read_artifact", {"artifact_id": name}, key=name)
        assert result.error.code in ("permission_denied", "not_found")
    # Una lectura confirmada también revalida sus permisos.
    saved = tools.store.get("safe_artifact:" + aid)
    tools.store.put("safe_artifact:" + aid, {**saved, "scope": "test"})
    assert call(tools, "read_artifact", {"artifact_id": aid, "length": 5}, role="diagnosis", key="read").error.code == "permission_denied"


def test_candidate_revision_validates_sources_and_expected_version(tools):
    original = cand()
    tools.store.put("tool_candidates", {original.id: original.to_dict()})
    updated = replace(original, version=2)
    args = {"expected": original.reference.to_dict(), "candidate": updated.to_dict()}
    invalid = {**args, "candidate": {**updated.to_dict(), "sources": ["invented"]}}
    assert call(tools, "revise_candidate", invalid).error.code == "unknown_source"
    assert tools.store.get("consumed") is None
    assert call(tools, "revise_candidate", args).status == "completed"
    assert call(tools, "revise_candidate", args, key="stale").error.code == "stale_version"
    assert tools.store.get("tool_candidates")[original.id]["version"] == 2


@pytest.mark.parametrize("change", [
    {"architecture": "Invented"}, {"batch_sizes": [1]}, {"dropout_values": [False]},
    {"lr_values": [-.1]}, {"img_size": 999}, {"version": 4}, {"id": "different"},
])
def test_invalid_revision_never_changes_candidate_or_spends(tools, change):
    original = cand()
    tools.store.put("tool_candidates", {original.id: original.to_dict()})
    result = call(tools, "revise_candidate", {"expected": original.reference.to_dict(),
                  "candidate": {**original.to_dict(), "version": 2, **change}})
    assert result.status == "failed"
    assert tools.store.get("tool_candidates") == {original.id: original.to_dict()}
    assert tools.store.get("consumed") is None


@pytest.mark.parametrize("counter", BudgetManager.counters)
def test_atomic_reservations_survive_restart_and_failures(tmp_path, counter):
    cfg = RunConfig(dataset_dir=str(tmp_path), max_training_attempts=1, max_decisions=1,
                    max_llm_calls=1, max_retries=1)
    store = StateStore(tmp_path)
    manager = BudgetManager(store, cfg)
    manager.reserve(**{counter: 1})  # Simular caída sin confirmar.
    store.close()
    second = StateStore(tmp_path)
    resumed = BudgetManager(second, cfg)
    with pytest.raises(PolicyError, match="agotado"):
        resumed.reserve(**{counter: 1})
    assert resumed.snapshot()[counter] == 1
    second.close()


def test_shared_budget_across_roles_and_close_without_budget(tools):
    limits = tools.store.get("limits")
    tools.store.put("limits", {**limits, "max_decisions": 1})
    assert call(tools, "query_evidence", {"query": "x"}, role="configuration").status == "completed"
    result = call(tools, "query_evidence", {"query": "x"}, role="diagnosis", key="second")
    assert result.error.code == "budget_exhausted"
    assert len(tools.calls) == 1
    assert call(tools, "close_search", {"reason": "budget"}, key="close").status == "completed"


def test_wall_clock_deadline_survives_resume_and_cancels_epoch(tools, monkeypatch):
    clock = [100.]
    monkeypatch.setattr("automl_agents.policies.time.time", lambda: clock[0])
    tools.budget.reserve(decisions=1)
    token = SearchCancellation(CancellationToken(), tools.budget)
    assert not token.is_cancelled()
    clock[0] += tools.cfg.max_search_seconds + 1
    assert token.is_cancelled()
    other = StateStore(tools.cfg.run_dir)
    resumed = BudgetManager(other, tools.cfg)
    with pytest.raises(PolicyError, match="Tiempo"):
        resumed.reserve(training_attempts=1)
    assert call(tools, "query_evidence", {"query": "x"}).error.code == "budget_exhausted"
    assert not tools.calls
    other.close()


def test_all_tools_have_serializable_contracts_and_effects():
    for spec in REGISTRY.values():
        assert spec.input.model_json_schema()
        assert spec.output.model_json_schema()
        assert spec.phases and spec.roles and spec.recoverable_errors
        assert isinstance(spec.effects, tuple)


def test_concurrent_reservations_cannot_overspend(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    cfg = RunConfig(dataset_dir=str(tmp_path), max_training_attempts=1)
    initial = StateStore(tmp_path)
    BudgetManager(initial, cfg)
    initial.close()

    def reserve(_):
        store = StateStore(tmp_path)
        try:
            BudgetManager(store, cfg).reserve(training_attempts=1)
            return True
        except PolicyError:
            return False
        finally:
            store.close()
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(reserve, range(2))) == [False, True]
