"""Regresión de respuestas truncadas por el razonamiento de Gemini, sin red."""
import json
from types import SimpleNamespace

import pytest
from google.genai import types

from automl_agents.config import RunConfig
from automl_agents.context import ModelContextBudget
from automl_agents.dataset import DatasetProfile
from automl_agents.llm import LLM
from automl_agents.policies import BudgetedLLM
from automl_agents.reporting import write_criteria_report
from automl_agents.specialists.configuration import ConfigurationSpecialist
from automl_agents.specialists.contracts import ConfigurationContext
from automl_agents.state_store import StateStore


def client(monkeypatch, model, response):
    from google import genai

    requests = []

    def generate(**kwargs):
        requests.append(kwargs)
        return response

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate)))
    return LLM(model), requests


def response(text, finish="STOP", thoughts=0, output=20):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=text)]),
                                    finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            thoughts_token_count=thoughts, candidates_token_count=output))


@pytest.mark.parametrize("model", ["gemini-3.5-flash", "models/gemini-3.8-flash", "gemini-3.1-pro-preview"])
def test_gemini3_limits_thinking_without_exceeding_persisted_reserve(tmp_path, monkeypatch, model):
    llm, requests = client(monkeypatch, model, response('{"ok":true}'))
    cfg = RunConfig(dataset_dir=str(tmp_path), llm_context_budget=ModelContextBudget(
        input_tokens=4000, output_tokens=1000, window_tokens=6000))
    store = StateStore(tmp_path)
    try:
        assert BudgetedLLM(llm, store, cfg).complete_json("JSON", "{}", max_tokens=3000) == {"ok": True}
        config = requests[0]["config"]
        assert config.thinking_config.thinking_level == "LOW"
        assert config.thinking_config.thinking_budget is None
        assert config.max_output_tokens == 1000
        assert config.response_mime_type == "application/json"
        assert len(requests) == store.status()["budget_consumed"]["llm_calls"] == 1
    finally:
        store.close()


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
def test_gemini25_keeps_its_supported_thinking_budget(monkeypatch, model):
    llm, requests = client(monkeypatch, model, response('{"ok":true}'))
    assert llm.complete_json("JSON", "{}") == {"ok": True}
    assert requests[0]["config"].thinking_config.thinking_budget == 0
    assert requests[0]["config"].thinking_config.thinking_level is None


def test_unknown_model_does_not_receive_model_specific_thinking_options(monkeypatch):
    llm, requests = client(monkeypatch, "custom-model", response('{"ok":true}'))
    llm.complete_json("JSON", "{}")
    assert requests[0]["config"].thinking_config is None


@pytest.mark.parametrize("text", ['', '{"candidatas":[', '{"ok":true}'])
def test_truncated_response_is_rejected_before_parsing_even_if_valid_json(monkeypatch, text):
    llm, requests = client(monkeypatch, "gemini-3.5-flash", response(text, "MAX_TOKENS", 1911, 85))
    with pytest.raises(ValueError, match="MAX_TOKENS; límite=2000, razonamiento=1911, respuesta=85"):
        llm.complete_json("JSON", "{}")
    assert llm.last_usage["thoughts_token_count"] == 1911
    assert len(requests) == 1


def test_specialist_and_report_preserve_truncation_cause(tmp_path, monkeypatch):
    llm, requests = client(monkeypatch, "gemini-3.5-flash", response('{', "MAX_TOKENS", 1911, 85))
    cfg = RunConfig(dataset_dir=str(tmp_path))
    profile = DatasetProfile("", ["a", "b"], {"a": 50, "b": 50}, 100, 1., (64, 64), 64, scope="train")
    context = ConfigurationContext.build(profile, [], cfg)
    store = StateStore(tmp_path)
    try:
        result = ConfigurationSpecialist().run(context, BudgetedLLM(llm, store, cfg, role="configuration"))
        assert result.mode == "reglas" and result.candidates
        assert "MAX_TOKENS" in result.note
        assert "razonamiento=1911" in result.error.message
        assert len(requests) == 1
        assert store.get("llm_calls")[0]["error"] == "ValueError"
        prop = {**result.to_dict(), "candidates": result.candidates, "criteria": {},
                "space": {"total_configs": 100, "candidate_configs": 10, "reduction_pct": 90}}
        write_criteria_report(tmp_path, profile, prop, [])
        report = tmp_path / "reports/01_criterios_acotacion"
        assert "MAX_TOKENS" in report.with_suffix(".md").read_text("utf-8")
        assert json.loads(report.with_suffix(".json").read_text("utf-8"))["error"] == result.error.to_dict()
    finally:
        store.close()
