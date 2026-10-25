import json
import subprocess
import sys
import types
from dataclasses import replace

import pytest
from automl_agents.state_store import RunBusyError, TrainingBusyError
from pydantic import ValidationError

from automl_agents.cli import main
from automl_agents.config import RunConfig
from automl_agents.schemas import (
    Budget, Candidate, CandidateRef, Decision, DomainError, MetricPolicy, Plan,
    RunState, ToolResult, TrialRef, TrialResult,
)


@pytest.mark.parametrize("overrides", [
    {"val_fraction": 0}, {"test_fraction": -0.1},
    {"val_fraction": 0.5, "test_fraction": 0.5},
    {"val_fraction": float("nan")}, {"test_fraction": float("inf")},
    {"resource_tier": "tiny"}, {"mode": "unknown"},
    {"allowed_architectures": []}, {"allowed_architectures": ["Unknown"]},
    {"allowed_architectures": ["VGG16"]},
    {"allowed_architectures": ["MobileNetV2", "MobileNetV2"]},
    {"max_candidates": 0}, {"rounds": -1}, {"trials_per_round": 0},
    {"max_epochs": True}, {"max_epochs": 1.5}, {"img_size_override": 0},
    {"target_accuracy": 1.01}, {"discard_margin": -0.1}, {"seed": -1},
    {"max_training_attempts": 0}, {"max_search_seconds": float("inf")},
    {"max_search_seconds": 0}, {"max_decisions": 0},
    {"max_llm_calls": -1}, {"max_retries": -1},
    {"metric_policy": {"name": "f1_macro"}},
    {"metric_policy": {"direction": "minimize"}},
    {"metric_policy": {"tie_breaker": "random"}},
])
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        RunConfig(dataset_dir="x", **overrides)


def test_config_roundtrip_and_legacy_accuracy():
    cfg = RunConfig(dataset_dir="x", rounds=3, max_candidates=2,
                    trials_per_round=5, target_accuracy=0.8)
    assert cfg.metric_policy == MetricPolicy(threshold=0.8)
    assert cfg.max_training_attempts == 30
    assert RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    assert Budget.model_validate_json(cfg.initial_budget().model_dump_json()) == cfg.initial_budget()
    assert RunConfig(dataset_dir="x", metric_policy={"threshold": 0.7}).target_accuracy == 0.7


@pytest.mark.parametrize("overrides", [{"use_llm": False}, {"mode": "rules"}, {"max_llm_calls": 0}])
def test_llm_disabled(overrides):
    assert not RunConfig(dataset_dir="x", **overrides).llm_enabled


def candidate(**overrides):
    data = dict(id="c1", architecture="MobileNetV2", rationale="", sources=[],
                lr_values=[1e-4], dropout_values=[0.2], batch_sizes=[16],
                augment_values=[True], img_size=64)
    return Candidate(**(data | overrides))


def test_candidate_roundtrip_and_versioning():
    c = candidate()
    assert Candidate.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    updated = replace(c, version=2, lr_values=[3e-4])
    assert c.reference == CandidateRef(candidate_id="c1", candidate_version=1)
    assert updated.reference.candidate_version == 2


@pytest.mark.parametrize("overrides", [
    {"architecture": "Fake"}, {"lr_values": []}, {"lr_values": [0.5]},
    {"dropout_values": [0.2, 0.2]}, {"batch_sizes": [True]},
    {"augment_values": [1]}, {"version": 0}, {"img_size": 0},
])
def test_invalid_candidate(overrides):
    with pytest.raises(ValueError):
        candidate(**overrides)


def reference(**overrides):
    return TrialRef(**(dict(candidate_id="c1", candidate_version=1,
                           study_id="s1", trial_id="t1", attempt_id="a1") | overrides))


def test_contract_roundtrips():
    error = DomainError(code="oom", message="Sin memoria", recoverable=True)
    contracts = [
        error, reference(), candidate().reference, MetricPolicy(),
        Plan(tasks=[{"task_id": "train", "description": "Entrenar"},
                    {"task_id": "evaluate", "description": "Evaluar", "dependencies": ["train"]}]),
        Decision(decision_id="d1", action="train", reason="Presupuesto disponible",
                 candidate=candidate().reference),
        ToolResult(result_id="r1", tool="train", status="failed", error=error),
        TrialResult(reference=reference(), status="failed", seed=42, error=error),
        TrialResult(reference=reference(attempt_id="a2"), status="completed", seed=42,
                    metrics={"accuracy": 0.8}, checkpoint="models/t1.keras"),
        RunState(run_id="run1", mode="rules", budget=RunConfig(dataset_dir="x").initial_budget(),
                 candidates=[candidate().reference], trials=[reference(), reference(attempt_id="a2")]),
    ]
    for contract in contracts:
        assert type(contract).model_validate_json(json.dumps(contract.to_dict())) == contract


def test_contracts_reject_inconsistent_data():
    with pytest.raises(ValidationError):
        ToolResult(result_id="r", tool="train", status="failed")
    with pytest.raises(ValidationError):
        TrialResult(reference=reference(), status="failed", seed=42)
    with pytest.raises(ValidationError):
        ToolResult(result_id="r", tool="train", status="completed", data={"model": object()})
    with pytest.raises(ValidationError):
        RunState(run_id="run", mode="rules", budget=RunConfig(dataset_dir="x").initial_budget(),
                 trials=[reference()])
    with pytest.raises(ValidationError):
        RunState(run_id="run", mode="rules", budget=RunConfig(dataset_dir="x").initial_budget(),
                 candidates=[candidate().reference], trials=[reference(), reference()])
    with pytest.raises(ValidationError):
        Budget(**(RunConfig(dataset_dir="x").initial_budget().to_dict() | {"llm_calls": 21}))
    for tasks in ([{"task_id": "a", "description": "a", "dependencies": ["missing"]}],
                  [{"task_id": "a", "description": "a", "dependencies": ["a"]}]):
        with pytest.raises(ValidationError):
            Plan(tasks=tasks)


def test_invalid_cli_does_not_import_pipeline():
    # Proceso limpio para comprobar que no se importa TensorFlow ni el pipeline.
    code = """
import sys
from automl_agents.cli import main
try:
    main(['run', '--dataset', 'x', '--rounds', '0'])
except SystemExit as e:
    assert e.code == 2
else:
    raise AssertionError('No se rechazó la configuración')
assert 'automl_agents.pipeline' not in sys.modules
assert 'tensorflow' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_passes_new_options(monkeypatch, capsys):
    captured = []
    pipeline = types.ModuleType("automl_agents.pipeline")
    pipeline.run_pipeline = lambda cfg: captured.append(cfg) or {"ok": True}
    monkeypatch.setitem(sys.modules, "automl_agents.pipeline", pipeline)
    main(["run", "--dataset", "x", "--no-llm", "--mode", "rules",
          "--primary-metric", "accuracy", "--target-accuracy", "0.8",
          "--max-training-attempts", "3", "--max-search-seconds", "60",
          "--max-decisions", "5", "--max-llm-calls", "0", "--max-retries", "0",
          "--llm-timeout-seconds", "5", "--max-stagnant-blocks", "2",
          "--max-configuration-llm-calls", "0", "--max-diagnosis-llm-calls", "2",
          "--rag-max-documents", "7", "--rag-max-tokens", "2500", "--context-read-max-chars", "500",
          "--llm-input-tokens", "3000", "--llm-output-reserve", "1000", "--llm-context-window", "5000",
          "--max-no-progress-actions", "4", "--val-fraction", "0.2", "--test-fraction", "0.1", "--seed", "7"])
    cfg = captured[0]
    assert cfg.llm_timeout_seconds == 5 and cfg.max_stagnant_blocks == 2
    assert cfg.max_no_progress_actions == 4
    assert cfg.max_configuration_llm_calls == 0 and cfg.max_diagnosis_llm_calls == 2
    assert cfg.rag_max_documents == 7 and cfg.rag_max_tokens == 2500 and cfg.context_read_max_chars == 500
    assert cfg.llm_context_budget.to_dict() == {"input_tokens": 3000, "output_tokens": 1000, "window_tokens": 5000}
    assert cfg.target_accuracy == cfg.metric_policy.threshold == 0.8
    assert cfg.max_training_attempts == 3 and cfg.max_search_seconds == 60
    assert cfg.max_decisions == 5 and cfg.max_retries == 0 and cfg.seed == 7
    assert not cfg.llm_enabled and cfg.val_fraction == 0.2
    assert '"ok": true' in capsys.readouterr().out


@pytest.mark.parametrize("error", [ValueError("Dataset incompatible"), OSError("Sin espacio"),
                                  RunBusyError("Proceso activo"), TrainingBusyError("Entrenamiento activo")])
def test_cli_run_reports_operational_errors(monkeypatch, capsys, error):
    pipeline = types.ModuleType("automl_agents.pipeline")
    def fail(cfg):
        raise error
    pipeline.run_pipeline = fail
    monkeypatch.setitem(sys.modules, "automl_agents.pipeline", pipeline)
    with pytest.raises(SystemExit) as stopped:
        main(["run", "--dataset", "x", "--no-llm"])
    assert stopped.value.code == 2
    assert str(error) in capsys.readouterr().err


def test_redirected_cli_output_supports_docker_unicode():
    code = """
import sys, types
sys.stdout.reconfigure(encoding='cp1252')
pipeline = types.ModuleType('automl_agents.pipeline')
pipeline.run_pipeline = lambda cfg: {'docker_log': '\\u2501\\u2713', 'classes': ['\\u5708']}
sys.modules['automl_agents.pipeline'] = pipeline
from automl_agents.cli import main
main(['run', '--dataset', 'x', '--no-llm'])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    payload, _ = json.JSONDecoder().raw_decode(result.stdout.decode("utf-8"))
    assert payload == {"docker_log": "━✓", "classes": ["圈"]}
