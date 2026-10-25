"""Etapa 8: límites reales, recuperación durable, ACL y prompts pequeños, sin red."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from automl_agents.config import RunConfig
from automl_agents.context import ContextManager, ModelContextBudget, ReferenceStore, token_count
from automl_agents.kb import Doc, KnowledgeBase
from automl_agents.policies import BudgetedLLM, PolicyError
from automl_agents.schemas import Candidate, TrialResult
from automl_agents.state_store import StateStore
from automl_agents.tools import DomainTools, EvidenceOutput


@pytest.fixture
def service(tmp_path):
    cfg = RunConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), run_id="rag",
                    use_llm=False, img_size_override=32, max_decisions=50,
                    rag_max_documents=2, rag_max_tokens=100, context_read_max_chars=100)
    store = StateStore(cfg.run_dir)
    documents = [Doc("guide", "Guide", "batch memory", "guide", source="manual", date="2026-10-21")]
    calls = []
    def query(*args, **kwargs):
        calls.append((args, kwargs))
        return documents
    tools = DomainTools(cfg, store, kb=SimpleNamespace(query=query))
    yield tools, documents, calls
    store.close()


def retrieve(t, key, role="orchestrator", **args):
    return t.call("query_evidence", {"query": "memoria", **args}, role=role, action_id=key)


def test_rag_unique_versions_share_global_limits_and_survive_restart(service):
    t, docs, calls = service
    first = retrieve(t, "one")
    assert first.status == "completed"
    ref = first.data["documents"][0]
    assert ref["source"] == "manual" and ref["version"]
    assert retrieve(t, "one") == first and len(calls) == 1
    for role in ("diagnosis", "configuration"):
        result = retrieve(t, role, role)
        assert result.data["usage"] == first.data["usage"]
        assert result.data["documents"][0]["artifact_id"] == ref["artifact_id"]
    # El mismo ID cambia de contenido: nueva versión; el fragmento anterior es inmutable.
    docs[:] = [Doc("guide", "Guide", "new batch recommendation", "guide")]
    updated = retrieve(t, "updated")
    assert updated.data["usage"]["documents"] == 2
    assert updated.data["documents"][0]["version"] != ref["version"]
    assert t.references.get(ref["artifact_id"], "diagnosis")["data"] == "batch memory"
    other_store = StateStore(t.cfg.run_dir)
    try:
        resumed = DomainTools(replace(t.cfg, rag_max_documents=100, rag_max_tokens=10000), other_store, kb=t.kb)
        docs[:] = [Doc("third", "Third", "memory", "guide")]
        rejected = retrieve(resumed, "third")
        assert rejected.status == "completed" and rejected.data["documents"] == []
        assert rejected.data["limits"] == {"documents": 2, "tokens": 100}
        assert rejected.data["omitted"] == 1
        trace = other_store.get("rag_retrieval:" + updated.data["retrieval_id"])
        assert trace["query"] == "memoria" and trace["documents"] == updated.data["documents"]
    finally:
        other_store.close()


def test_token_limit_filters_duplicates_and_scope_permissions(service):
    t, docs, _ = service
    docs[:] = [Doc("big", "Big", "x" * 1000, "guide"),
               Doc("secret", "Secret", "test metrics", "guide", scope="test"),
               Doc("validation", "Validation", "validation knowledge", "guide", scope="validation"),
               Doc("arch", "Architecture", "architecture", "arch")]
    config = retrieve(t, "config", "configuration", kinds=["guide"])
    assert config.data["documents"] == [] and config.data["usage"]["tokens"] == 0
    diagnosis = retrieve(t, "diag", "diagnosis", kinds=["guide"])
    assert [d["id"] for d in diagnosis.data["documents"]] == ["validation"]
    artifact_id = diagnosis.data["documents"][0]["artifact_id"]
    assert t.call("read_artifact", {"artifact_id": artifact_id}, role="configuration",
                  action_id="denied").error.code == "permission_denied"
    docs[:] = [Doc("guide", "Guide", "batch memory", "guide")] * 3
    result = retrieve(t, "duplicates")
    assert len(result.data["documents"]) == 1
    assert result.data["usage"]["tokens"] <= 100
    assert result.data["usage"]["documents"] == 2


def test_optional_markdown_metadata_and_type_filter(tmp_path):
    (tmp_path / "guide_new.md").write_text(
        "---\nsource: https://example.org/manual\ndate: 2026-10-21\nscope: train\n---\n"
        "# New guide\nReduce el batch si falta memoria.", encoding="utf-8")
    (tmp_path / "arch_old.md").write_text("# Old architecture\nSmall image model", encoding="utf-8")
    kb = KnowledgeBase(backend="hash", knowledge_dir=tmp_path)
    guide = kb.query("memoria", 10, kinds=["guide"])[0]
    assert guide.id == "guide_new" and guide.date == "2026-10-21" and guide.scope == "train"
    assert guide.source == "https://example.org/manual"
    assert kb.query("model", 3, kinds=["arch"])[0].source is None
    assert kb.query("x", kinds=["unknown"]) == []
    assert len(kb.retrieve_many(["memoria"] * 3, max_documents=1)) == 1
    assert kb.retrieve_many(["memoria"], max_tokens=1) == []
    (tmp_path / "guide_new.md").unlink()
    assert kb.build() == 1 and kb.collection.count() == 1


def test_kb_still_works_as_standalone_module_for_colab(tmp_path):
    import importlib.util
    import sys
    import automl_agents.kb as original
    spec = importlib.util.spec_from_file_location("standalone_rag_snapshot", original.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        (tmp_path / "guide_memory.md").write_text("# Memory\nUse a small batch", encoding="utf-8")
        kb = module.KnowledgeBase(backend="hash", knowledge_dir=tmp_path)
        assert kb.retrieve_many(["memory"])[0].version
    finally:
        sys.modules.pop(spec.name, None)


def test_large_context_is_structured_bounded_and_originals_are_recoverable(service):
    t, _, _ = service
    context = ContextManager(t.store, t.cfg)
    history = [{"attempt_id": str(i), "status": "completed", "val_acc": i / 1000,
                "candidate_id": "one", "candidate_version": 1} for i in range(1000)]
    payload = {"observation": {"records": history, "constraints": {"architectures": ["Invented"]}},
               "context": {"curves": {"accuracy": [.8] * 10000, "val_accuracy": [.5] * 10000},
                           "evidence": [{"id": "large", "text": "Long source text. " * 10000}]}}
    t.store.put("plan", {"tasks": [{"task_id": str(i), "status": "completed"} for i in range(300)]})
    prompt, output, audit = context.prepare("System", json.dumps(payload), "orchestrator")
    data = json.loads(prompt)
    assert token_count("System" + prompt)[0] <= t.cfg.llm_context_budget.input_tokens
    assert audit["estimated"] and audit["method"] == "estimated_utf8_bytes"
    assert output == 2000
    assert data["observation"]["records"]["count"] == 1000
    assert len(data["observation"]["records"]["recent"]) <= 5
    assert data["context"]["curves"]["summary"]["accuracy"]["count"] == 10000
    assert "Invented" not in data["authoritative"]["constraints"]["architectures"]
    assert data["authoritative"]["objective"]["metric"] == t.cfg.metric_policy.to_dict()
    assert data["authoritative"]["budget"] == t.budget.snapshot()
    assert data["plan"]["tasks"]["count"] == 300
    artifact_id = data["context"]["curves"]["artifact_id"]
    parts, offset = [], 0
    while True:
        read = t.references.read(artifact_id, "orchestrator", offset, length=8000)
        assert len(read["text"]) <= 100
        parts.append(read["text"])
        if read["next_offset"] is None:
            break
        offset = read["next_offset"]
    assert json.loads("".join(parts)) == payload["context"]["curves"]
    with pytest.raises(PolicyError, match="rol"):
        t.references.read(artifact_id, "configuration")
    with pytest.raises(PolicyError):
        t.references.read(artifact_id, "diagnosis")


def test_role_scope_is_retained_when_text_is_externalized(service):
    t, _, _ = service
    context = ContextManager(t.store, t.cfg)
    prompt, _, _ = context.prepare("", json.dumps({"context": {"text": "validation curve data " * 100}}), "diagnosis")
    aid = json.loads(prompt)["context"]["artifact_id"]
    assert t.references.get(aid, "diagnosis")["scope"] == "validation"
    with pytest.raises(PolicyError):
        t.references.get(aid, "configuration")
    saved = t.store.get("safe_artifact:" + aid)
    t.store.put("safe_artifact:" + aid, {**saved, "scope": "test"})
    with pytest.raises(PolicyError):
        t.references.get(aid, "orchestrator")


def test_bounded_read_text_is_not_truncated_again_and_final_test_is_not_published_as_history(service):
    t, _, _ = service
    read = {"text": "x" * 90, "total_chars": 500, "next_offset": 90, "version": "v"}
    context = ContextManager(t.store, t.cfg)
    prompt, _, _ = context.prepare("", json.dumps({"read": read}), "orchestrator")
    assert json.loads(prompt)["read"] == read
    t.store.put("test_opened", True)
    prompt, _, _ = context.prepare("", json.dumps({"report": "final evaluation"}), "orchestrator")
    aid = json.loads(prompt)["original_artifact_id"]
    for role in ("orchestrator", "configuration", "diagnosis"):
        with pytest.raises(PolicyError):
            t.references.get(aid, role)


def test_impossible_input_preserves_constraints_and_does_not_contact_provider(service):
    t, _, _ = service
    cfg = replace(t.cfg, llm_model="tiny", use_llm=True,
                  llm_model_budgets={"tiny": ModelContextBudget(input_tokens=10, output_tokens=5, window_tokens=20)})
    # Un nuevo store fija esta política, sin alterar límites de una ejecución existente.
    store = StateStore(t.cfg.run_dir / "tiny")
    calls = []
    client = SimpleNamespace(available=True, complete_json=lambda *a: calls.append(a))
    try:
        llm = BudgetedLLM(client, store, cfg)
        with pytest.raises(PolicyError) as exc:
            llm.complete_json("System", "{}")
        assert exc.value.error.code == "context_limit"
        assert calls == [] and llm.budget.snapshot()["llm_calls"] == 0
        assert store.get("context_policy")["constraints"]["architectures"]
    finally:
        store.close()


def test_model_budget_output_reserve_and_context_policy_survive_resume(service):
    t, _, _ = service
    cfg = replace(t.cfg, use_llm=True)
    calls = []
    class Client:
        available = True
        model_name = "custom"
        last_usage = None
        def complete_json(self, system, user, max_tokens=2000):
            calls.append((system, user, max_tokens))
            return {"ok": True}
    store = StateStore(t.cfg.run_dir / "budget")
    try:
        cfg = replace(cfg, llm_model_budgets={"custom": ModelContextBudget(
            input_tokens=4000, output_tokens=37, window_tokens=5000)})
        BudgetedLLM(Client(), store, cfg).complete_json("System", "{}", max_tokens=1000)
        assert calls[0][2] == 37
        assert store.get("llm_calls")[0]["context"]["output_reserve"] == 37
        resumed = ContextManager(store, replace(cfg, allowed_architectures=["MobileNetV2"],
                                                llm_model_budgets={}))
        assert resumed.model_budget("custom").output_tokens == 37
        assert len(resumed.policy["constraints"]["architectures"]) > 1
    finally:
        store.close()


def test_local_tokenizer_is_identified(service):
    t, _, _ = service
    tokenizer = SimpleNamespace(encode=lambda text: list(text))
    prompt, _, audit = ContextManager(t.store, t.cfg).prepare("x", "{}", "diagnosis", tokenizer=tokenizer)
    assert audit["method"] == "tokenizer" and not audit["estimated"]
    assert audit["input_tokens"] == len("x" + prompt)


@pytest.mark.parametrize("backend_failure", [False, True])
def test_diagnosis_recovers_guide_after_oom_and_keeps_original_curves(service, backend_failure):
    t, docs, calls = service
    c = Candidate("one", "MobileNetV3Small", "Reason", [], [.001], [0.], [16], [False], 32)
    ref = {**c.reference.to_dict(), "study_id": "study", "trial_id": "trial", "attempt_id": "failed"}
    trial = TrialResult(reference=ref, status="failed", seed=42, parameters={},
                        curves={"accuracy": [.4] * 1000},
                        error={"code": "out_of_memory", "message": "OOM", "recoverable": True})
    meta = {**ref, "round": 1, "trial": 1, "architecture": c.architecture}
    t.store.put("tool_candidates", {c.id: c.to_dict()})
    t.store.begin_attempt("block", meta)
    t.store.confirm_attempt(t.store.make_record(meta, trial))
    if backend_failure:
        def broken(*args, **kwargs):
            raise RuntimeError("Chroma unavailable")
        t.kb.query = broken
    result = t.call("diagnose_trial", {"attempt_id": "failed"}, role="orchestrator", action_id="diagnose")
    assert result.status == "completed" and result.data["mode"] == "reglas"
    saved = t.store.get("specialist:diagnose")["context"]
    if backend_failure:
        assert t.store.get("specialist_rag:diagnose")["status"] == "failed"
        assert not calls and not saved["evidence"]
    else:
        assert len(calls) == 1 and "OOM" in calls[0][0][0]
        assert calls[0][1]["kinds"] == ["guide"]
        assert saved["evidence"][0]["version"] == docs[0].version
    assert saved["curves"]["accuracy"] == [.4] * 1000
    assert t.budget.snapshot()["decisions"] == 2
    assert t.budget.snapshot()["training_attempts"] == 1


@pytest.mark.parametrize("options", [{"rag_max_documents": 0}, {"rag_max_tokens": -1},
    {"context_read_max_chars": 9000}, {"llm_context_budget": {"input_tokens": 10, "output_tokens": 20, "window_tokens": 20}}])
def test_context_configuration_is_validated(options):
    with pytest.raises(ValidationError):
        RunConfig(dataset_dir="x", **options)


def test_new_configuration_round_trips():
    cfg = RunConfig(dataset_dir="x", llm_model_budgets={"custom": {"input_tokens": 3000,
                    "output_tokens": 50, "window_tokens": 4000}})
    assert RunConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_legacy_evidence_and_confirmed_results_remain_recoverable(tmp_path):
    cfg = RunConfig(dataset_dir=str(tmp_path), use_llm=False)
    store = StateStore(tmp_path)
    old = {"id": "guide", "title": "Guide", "text": "batch memory", "kind": "guide", "score": .8}
    store.put("tool_evidence", {"guide": old})
    try:
        service = DomainTools(cfg, store)
        assert store.get("tool_evidence")["guide"]["version"]
        assert store.status()["rag_budget_consumed"]["documents"] == 1
        assert store.status()["rag_budget_consumed"]["tokens"] > 0
        artifact_id = service._publish(old["text"], "evidence")
        cached = EvidenceOutput.model_validate({"documents": [{"id": "guide", "title": "Guide", "artifact_id": artifact_id}]})
        assert cached.documents[0].id == "guide"
    finally:
        store.close()
