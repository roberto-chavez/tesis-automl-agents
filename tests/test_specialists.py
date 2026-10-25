"""Delegaciones aisladas, permisos, recomendaciones y consumo durable, sin red."""
from dataclasses import replace
import json

import pytest
from pydantic import ValidationError

from automl_agents.config import RunConfig
from automl_agents.dataset import DatasetProfile
from automl_agents.kb import Doc
from automl_agents.policies import BudgetedLLM
from automl_agents.schemas import Candidate, TrialResult
from automl_agents.specialists import ConfigurationSpecialist, DiagnosisSpecialist
from automl_agents.specialists.contracts import ConfigurationContext, DiagnosisContext, DiagnosisResult
from automl_agents.state_store import StateStore
from automl_agents.tools import DomainTools


class Client:
    available = True
    last_usage = {"total_token_count": 7}

    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.prompts = payload, error, []

    def complete_json(self, system, user):
        self.prompts.append((system, json.loads(user)))
        if self.error:
            raise self.error
        return self.payload


def profile():
    return DatasetProfile("secret_path", ["a", "b"], {"a": 50, "b": 50}, 100,
                          1., (64, 64), 64, scope="train")


def candidate():
    return Candidate("one", "MobileNetV3Small", "test", [], [.001], [0.], [16, 32], [False], 32)


def advice(action="review", **kwargs):
    return {"hypotheses": [], "recommendation": {"action": action, "reason": "Revisar validación.", **kwargs}, "sources": []}


def record(c=None, status="completed", recoverable=True):
    c = c or candidate()
    ref = {**c.reference.to_dict(), "study_id": "study", "trial_id": "trial", "attempt_id": "attempt"}
    result = TrialResult(reference=ref, status=status, seed=42, parameters={"lr": .001},
                         metrics={"accuracy": .6, "f1_macro": .5},
                         curves={"accuracy": [.7, .95], "val_accuracy": [.5, .6]},
                         resources={"seconds": 5., "rss_mb_peak": 20.},
                         validation={"accuracy": .6, "confusion_matrix": [[3, 2], [2, 3]]},
                         error={"code": "out_of_memory", "message": "OOM", "recoverable": recoverable}
                         if status == "failed" else None)
    meta = {**ref, "round": 0, "trial": 0, "architecture": c.architecture}
    return meta, StateStore.make_record(meta, result)


@pytest.fixture
def tools(tmp_path):
    cfg = RunConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), run_id="specialists",
                    img_size_override=32, max_llm_calls=20, max_diagnosis_llm_calls=1)
    store = StateStore(cfg.run_dir)
    service = DomainTools(cfg, store, profile=profile())
    yield service
    service.store.close()


def register(tools, status="completed", recoverable=True):
    c = candidate()
    tools.store.put("tool_candidates", {c.id: c.to_dict()})
    meta, r = record(c, status, recoverable)
    tools.store.begin_attempt("block", meta)
    tools.store.confirm_attempt(r)
    return r


def test_configuration_context_and_results_are_serializable_and_train_only():
    cfg = RunConfig(dataset_dir="secret", img_size_override=32, max_candidates=2)
    docs = [Doc("guide", "Guide", "Transfer learning", "guide", .8)]
    context = ConfigurationContext.build(profile(), docs, cfg)
    assert ConfigurationContext.model_validate_json(context.model_dump_json()) == context
    assert "secret" not in context.model_dump_json() and "test_fraction" not in context.model_dump_json()
    result = ConfigurationSpecialist().run(context)
    assert len(result.candidates) == 2 and result.mode == "reglas"
    assert all(c.version == 1 and c.img_size == 32 for c in result.candidates)
    assert result.validate_context(context) == result
    with pytest.raises(ValidationError):
        ConfigurationContext.build(replace(profile(), scope="all"), docs, cfg)


def test_configuration_reuses_llm_generator_and_validates_catalog():
    context = ConfigurationContext.build(profile(), [], RunConfig(dataset_dir="x", img_size_override=32))
    client = Client({"candidatas": [{"arquitectura": "MobileNetV3Small", "lr": [.001],
                      "dropout": [0.], "batch": [16], "augment": [False]}]})
    result = ConfigurationSpecialist().run(context, client)
    assert result.mode == "llm" and result.candidates[0].architecture == "MobileNetV3Small"
    assert "secret_path" not in json.dumps(client.prompts)
    bad = ConfigurationSpecialist().run(context, Client({"candidatas": [{"arquitectura": "Invented"}]}))
    assert bad.mode == "reglas" and bad.error.code == "candidate_proposal_failed"


def test_diagnosis_context_projects_nested_fields_and_separates_hypotheses():
    _, r = record()
    r["test"] = r["checkpoint"] = "test_secret"
    for name in ("metrics", "validation", "resources", "curves"):
        r["result"][name]["test"] = "test_secret"
    context = DiagnosisContext.build(r, candidate())
    assert "test_secret" not in context.model_dump_json()
    assert DiagnosisContext.model_validate_json(context.model_dump_json()) == context
    result = DiagnosisSpecialist().run(context)
    assert result.mode == "reglas" and result.recommendation.action == "review"
    assert any(f.code == "generalization_gap" for f in result.findings)
    assert result.hypotheses[0].code == "overfitting"
    assert not any("sobreajuste" in f.detail for f in result.findings)
    assert DiagnosisResult.model_validate_json(result.model_dump_json()) == result


def test_llm_can_add_hypotheses_but_cannot_invent_verified_findings():
    _, r = record()
    context = DiagnosisContext.build(r, candidate())
    client = Client({**advice(), "hypotheses": [{"code": "insufficient_capacity", "explanation": "Requiere más observaciones."}]})
    result = DiagnosisSpecialist().run(context, client)
    assert result.mode == "llm" and result.findings == DiagnosisSpecialist().run(context).findings
    assert result.hypotheses[0].code == "insufficient_capacity"
    bad = DiagnosisSpecialist().run(context, Client({**advice(), "findings": ["Causa demostrada"]}))
    assert bad.mode == "reglas" and bad.error.code == "diagnosis_failed"


@pytest.mark.parametrize("change", [{"version": 4}, {"id": "other"}, {"batch_sizes": [1]},
    {"architecture": "VGG16"}, {"img_size": 999}, {"sources": ["invented"]}])
def test_invalid_recommendation_falls_back_without_candidate_changes(tools, change):
    register(tools)
    c = candidate()
    updated = {**c.to_dict(), "version": 2, **change}
    tools.llm = Client(advice("adjust", expected=c.reference.to_dict(), candidate=updated))
    before = tools.store.get("tool_candidates")
    result = tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="diagnose")
    assert result.status == "completed"
    assert result.data["mode"] == "reglas" and result.data["error"]["code"] == "diagnosis_failed"
    assert tools.store.get("tool_candidates") == before
    assert tools.budget.snapshot()["training_attempts"] == 1


@pytest.mark.parametrize("role,name", [(role, name) for role in ("configuration", "diagnosis")
    for name in ("train_block", "revise_candidate", "discard_candidate", "propose_candidates", "diagnose_trial", "close_search")])
def test_specialists_cannot_mutate_train_close_or_redelegate(tools, role, name):
    result = tools.call(name, {}, role=role, action_id="denied")
    assert result.error.code == "permission_denied"
    assert tools.store.get("consumed") is None


def test_role_specific_artifacts_and_validation_tools(tools):
    register(tools)
    artifact = tools._publish({"accuracy": .6}, "validation")
    for role in ("configuration", "diagnosis"):
        read = tools.call("read_artifact", {"artifact_id": artifact}, role=role, action_id=role)
        assert read.status == ("failed" if role == "configuration" else "completed")
        validation = tools.call("evaluate_validation", {"attempt_id": "attempt"}, role=role, action_id=role + "-val")
        assert validation.status == ("failed" if role == "configuration" else "completed")
        tools.store.put("safe_artifact:secret", {"scope": "test", "roles": [role], "data": "secret"})
        assert tools.call("read_artifact", {"artifact_id": "secret"}, role=role, action_id=role + "-test").error.code == "permission_denied"


def test_shared_and_role_limits_survive_resume_and_replay(tools):
    register(tools)
    tools.llm = Client(advice())
    first = tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="first")
    assert first.data["mode"] == "llm"
    assert tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="first") == first
    assert len(tools.llm.prompts) == 1
    tools.store.close()
    tools.store = StateStore(tools.cfg.run_dir)
    tools = DomainTools(replace(tools.cfg, max_diagnosis_llm_calls=10), tools.store, profile=profile(), llm=tools.llm)
    second = tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="second")
    assert second.data["mode"] == "reglas"
    assert len(tools.llm.prompts) == 1
    BudgetedLLM(tools.llm, tools.store, tools.cfg).complete_json("", "{}")
    assert tools.budget.snapshot()["llm_calls"] == 2
    assert tools.store.get("specialist_usage") == {"diagnosis": 1}
    assert tools.store.get("specialist_limits")["diagnosis"] == 1
    assert [c["role"] for c in tools.store.get("llm_calls")] == ["diagnosis", "orchestrator"]
    tools.store.close()


@pytest.mark.parametrize("options", [{"mode": "rules"}, {"use_llm": False}, {"max_llm_calls": 0}])
def test_no_llm_does_not_inspect_injected_provider(tools, options):
    class Forbidden:
        @property
        def available(self):
            raise AssertionError("No debe inspeccionarse el proveedor")
    register(tools)
    tools.cfg = replace(tools.cfg, **options)
    tools.llm = Forbidden()
    result = tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="local")
    assert result.data["mode"] == "reglas" and not tools.store.get("llm_calls")


def test_provider_failure_uses_basic_diagnosis_without_retry(tools):
    register(tools, status="failed", recoverable=False)
    tools.llm = Client(error=RuntimeError("provider down"))
    result = tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="down")
    assert result.data["mode"] == "reglas" and result.data["recommendation"]["action"] == "discard"
    assert tools.budget.snapshot()["llm_calls"] == 1 and tools.budget.snapshot()["retries"] == 0
    assert tools.store.get("specialist:down")["role"] == "diagnosis"


def test_configuration_delegation_is_idempotent_and_has_its_own_budget(tools):
    tools.llm = Client(error=RuntimeError("offline"))
    first = tools.call("propose_candidates", {}, role="orchestrator", action_id="proposal")
    assert first.status == "completed" and tools.store.get("proposal")["mode"] == "reglas"
    assert tools.call("propose_candidates", {}, role="orchestrator", action_id="proposal") == first
    assert tools.store.get("specialist_usage") == {"configuration": 1}
    assert len(tools.llm.prompts) == 1
    assert tools.store.get("specialist:proposal")["context"]["profile"]["scope"] == "train"


def test_global_llm_budget_is_shared_between_configuration_and_diagnosis(tools):
    tools.store.put("limits", {**tools.store.get("limits"), "max_llm_calls": 1})
    tools.llm = Client(error=RuntimeError("offline"))
    assert tools.call("propose_candidates", {}, role="orchestrator", action_id="config").status == "completed"
    # El presupuesto global detiene nuevas herramientas antes de contactar al diagnóstico.
    register(tools)
    denied = tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="diag")
    assert denied.error.code == "budget_exhausted" and len(tools.llm.prompts) == 1
    assert tools.store.get("diagnoses") is None


def test_zero_role_limit_uses_rules_and_keeps_training_budget(tools):
    register(tools)
    tools.cfg = replace(tools.cfg, max_diagnosis_llm_calls=0)
    tools.llm = Client(advice())
    result = tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="zero")
    assert result.data["mode"] == "reglas" and tools.llm.prompts == []
    assert tools.budget.snapshot()["llm_calls"] == 0
    assert tools.budget.snapshot()["decisions"] == 1


def test_host_rejects_invalid_specialist_output_before_persistence(tools, monkeypatch):
    register(tools)
    context = tools._diagnosis_context("attempt")
    basic = DiagnosisSpecialist().run(context)
    fake = basic.model_copy(update={"attempt_id": "another-attempt"})
    monkeypatch.setattr(DiagnosisSpecialist, "run", lambda *args: fake)
    result = tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="invalid")
    assert result.status == "failed" and tools.store.get("diagnoses") is None
    assert tools.store.get("tool_candidates")["one"] == candidate().to_dict()


@pytest.mark.parametrize("field", ["max_configuration_llm_calls", "max_diagnosis_llm_calls"])
def test_role_limits_are_validated_and_config_round_trips(field):
    with pytest.raises(ValidationError):
        RunConfig(dataset_dir="x", **{field: -1})
    cfg = RunConfig(dataset_dir="x", **{field: 0})
    assert RunConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_context_rejects_mismatched_training_counts_or_trial_version():
    cfg = RunConfig(dataset_dir="x")
    with pytest.raises(ValidationError, match="inconsistente"):
        ConfigurationContext.build(replace(profile(), n_images=101), [], cfg)
    _, r = record()
    with pytest.raises(ValidationError, match="no corresponde"):
        DiagnosisContext.build(r, replace(candidate(), version=2))


def test_timeout_charges_one_call_and_returns_rules(tools):
    import threading
    register(tools)
    tools.cfg = replace(tools.cfg, llm_timeout_seconds=.02)
    release = threading.Event()
    class SlowClient:
        available = True
        def complete_json(self, *args):
            release.wait(2)
            return advice()
    tools.llm = SlowClient()
    try:
        result = tools.call("diagnose_trial", {"attempt_id": "attempt"}, role="orchestrator", action_id="timeout")
        assert result.data["mode"] == "reglas" and result.data["error"]["code"] == "diagnosis_failed"
        assert tools.store.get("llm_calls")[0]["error"] == "TimeoutError"
        assert tools.store.get("specialist_usage") == {"diagnosis": 1}
        assert tools.budget.snapshot()["retries"] == 0
    finally:
        release.set()
