"""El timeout de la ejecución alcanza al SDK y al wrapper, sin red."""
from types import SimpleNamespace

import pytest

from automl_agents.config import RunConfig
from automl_agents.llm import DEFAULT_LLM_TIMEOUT_SECONDS, LLM
from automl_agents.policies import BudgetedLLM
from automl_agents.state_store import StateStore


def test_configured_timeout_reaches_http_requests_and_wrapper(tmp_path, monkeypatch):
    from google import genai

    requests = []
    options = []

    def generate(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(text='{"ok": true}', usage_metadata=None)

    def client(**kwargs):
        options.append(kwargs["http_options"])
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate))

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(genai, "Client", client)
    llm = LLM("gemini-3.8-flash")
    assert llm.available
    assert options[0].timeout == DEFAULT_LLM_TIMEOUT_SECONDS * 1000
    cfg = RunConfig(dataset_dir=str(tmp_path), llm_timeout_seconds=210.5)
    store = StateStore(tmp_path)
    try:
        counted = BudgetedLLM(llm, store, cfg)
        assert counted.timeout == llm.timeout_seconds == 210.5
        assert counted.complete_json("Return JSON", "Synthetic input") == {"ok": True}
        assert requests[0]["config"].http_options.timeout == 210500
        assert requests[0]["config"].http_options.retry_options.attempts == 1
    finally:
        store.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected_before_initializing_provider(timeout):
    with pytest.raises(ValueError, match="timeout"):
        LLM("gemini-3.8-flash", timeout_seconds=timeout)
