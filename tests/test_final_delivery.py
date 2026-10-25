"""Selección durable, test aislado, reportes recuperables y API exportada real."""
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("tensorflow")
import keras
import tensorflow as tf

from automl_agents import evaluation, packaging, reporting, trainer
from automl_agents.catalog import CATALOG
from automl_agents.config import RunConfig
from automl_agents.events import write_json
from automl_agents.pipeline import Pipeline
from automl_agents.policies import BudgetManager
from automl_agents.schemas import Candidate, TrialResult
from automl_agents.state_store import StateStore, file_sha256


@pytest.fixture
def final_run(tmp_path):
    cfg = RunConfig(dataset_dir=str(tmp_path / "missing-data"), output_dir=str(tmp_path), run_id="final",
                    use_llm=False, pretrained=False, target_accuracy=.8)
    store = StateStore(cfg.run_dir)
    write_json(cfg.run_dir / "config.json", cfg.to_dict())
    c1 = Candidate("one", "MobileNetV2", "inicial", [], [.001], [0.], [16], [False], 8)
    c2 = replace(c1, version=2, rationale="revisada")
    store.put("candidate_history", {"one:v1": c1.to_dict(), "one:v2": c2.to_dict()})
    classes = ["círculo", "cuadrado", "triángulo"]
    inp = keras.Input((8, 8, 3))
    out = keras.layers.Dense(3, activation="softmax", kernel_initializer="zeros",
                            bias_initializer=keras.initializers.Constant(np.log([.1, .7, .2])))(
                                keras.layers.GlobalAveragePooling2D()(inp))
    model = keras.Model(inp, out)
    for index, score in enumerate((.5, .9)):
        path = cfg.run_dir / f"model{index}.keras"
        model.save(path)
        ref = dict(candidate_id="one", candidate_version=index + 1, study_id=f"study{index}",
                   trial_id=f"trial{index}", attempt_id=f"attempt{index}")
        meta = {**ref, "round": index + 1, "trial": index, "architecture": c1.architecture,
                "params": {"lr": .001, "dropout": 0., "batch": 16, "augment": False}, "seed": 42}
        result = TrialResult(reference=ref, status="completed", seed=42, parameters=meta["params"],
                             metrics={"accuracy": score}, checkpoint=str(path),
                             resources={"seconds": 1.}, stop_reason="max_epochs")
        store.begin_attempt(str(index), meta)
        store.confirm_attempt(store.make_record(meta, result))
    profile = dict(n_images=20, n_classes=3, classes=classes, warnings=[], scope="train")
    store.put("training_profile", profile)
    store.put("risk", {"risk_level": "bajo", "notes": [], "disclaimer": "Prueba"})
    store.put("proposal", {"candidates": [c1.to_dict()], "mode": "rules", "space": {
        "total_configs": 20, "candidate_configs": 2, "reduction_pct": 90}})
    store.put("orchestrator", {"history": [{"decision": "cerrar", "reason": "presupuesto"}]})
    BudgetManager(store, cfg).close_search("budget_exhausted")
    ds = tf.data.Dataset.from_tensor_slices((np.zeros((3, 8, 8, 3), "float32"), [0, 1, 2]))
    engine = SimpleNamespace(profile=SimpleNamespace(classes=classes), splits={"test": ([], [0, 1, 2])},
                             base_ds=lambda split, cand: ds)
    yield cfg, store, engine
    store.close()


def test_frozen_selection_and_published_evaluation_recovery(final_run, monkeypatch):
    cfg, store, engine = final_run
    chosen = store.get("selection")
    assert chosen["candidate_version"] == 2
    assert chosen["candidate"]["rationale"] == "revisada"
    assert chosen["checkpoint_sha256"] == file_sha256(chosen["checkpoint"])
    BudgetManager(store, cfg).close_search("changed")
    assert store.get("selection") == chosen
    original = store.confirm_action
    monkeypatch.setattr(store, "confirm_action", lambda *a: (_ for _ in ()).throw(SystemExit(17)))
    with pytest.raises(SystemExit):
        evaluation.evaluate_frozen_selection(store, engine)
    assert store.get("test_opened")
    assert (cfg.run_dir / "final_evaluation.json").exists()
    monkeypatch.setattr(store, "confirm_action", original)
    monkeypatch.setattr(evaluation, "evaluate_model", lambda *a: pytest.fail("Test repetido"))
    saved = evaluation.evaluate_frozen_selection(store, engine)
    assert saved["winner"]["attempt_id"] == chosen["attempt_id"]
    assert saved["metrics"]["classes"] == engine.profile.classes
    assert evaluation.evaluate_frozen_selection(store, engine) == saved
    assert all("test" not in r["result"]["metrics"] for r in store.records())


def test_checkpoint_tampering_fails_before_test(final_run):
    _, store, engine = final_run
    Path(store.get("selection")["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint congelado"):
        evaluation.evaluate_frozen_selection(store, engine)
    assert not store.get("test_opened")


def test_final_metrics_stay_outside_search_observations(final_run):
    from automl_agents.orchestrator import Orchestrator
    from automl_agents.tools import DomainTools
    cfg, store, engine = final_run
    evaluation.evaluate_frozen_selection(store, engine)
    store.put("orchestrator", {"plan": {"tasks": []}, "history": []})
    tools = DomainTools(cfg, store)
    observation = json.dumps(Orchestrator(tools).observe())
    assert "n_test" not in observation and "model_size_mb" not in observation
    assert "final_evaluation" not in observation
    consumed = tools.budget.snapshot()
    denied = tools.call("query_evidence", {"query": "test results"}, role="orchestrator", action_id="after-test")
    assert denied.error.code == "phase_denied"
    assert tools.budget.snapshot() == consumed


def test_test_denied_without_closed_search(final_run):
    _, store, engine = final_run
    store.put("search_closed", False)
    with pytest.raises(RuntimeError, match="selección congelada"):
        evaluation.evaluate_frozen_selection(store, engine)
    assert not store.get("test_opened")


@pytest.mark.parametrize("module", ["mobilenet_v2", "mobilenet_v3", "resnet_v2", "efficientnet"])
@pytest.mark.parametrize("mode,fmt", [("RGB", "PNG"), ("L", "JPEG"), ("RGBA", "PNG"), ("RGB", "BMP"), ("RGB", "GIF")])
def test_training_and_exported_preprocessing_equivalent(tmp_path, module, mode, fmt):
    import importlib.util
    pixels = (np.arange(17 * 29 * 3).reshape(17, 29, 3) % 256).astype("uint8")
    data = io.BytesIO()
    Image.fromarray(pixels).convert(mode).save(data, fmt)
    path = tmp_path / "image"
    path.write_bytes(data.getvalue())
    spec = next(s for s in CATALOG.values() if s.module == module)
    training, _ = next(iter(trainer.load_base_dataset([path], [0], 11, spec, cache=False)))
    exported = tmp_path / "preprocessing.py"
    exported.write_bytes(Path(packaging.__file__).with_name("preprocessing.py").read_bytes())
    loader = importlib.util.spec_from_file_location("exported_preprocessing", exported)
    runtime = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(runtime)
    actual = runtime.decode_and_preprocess(data.getvalue(), 11, module)
    np.testing.assert_array_equal(training, actual)


def test_real_artifact_api_and_persistent_reports(final_run, monkeypatch):
    cfg, store, engine = final_run
    final = evaluation.evaluate_frozen_selection(store, engine)
    monkeypatch.setattr(packaging.shutil, "which", lambda _: None)
    artifact = packaging.build_artifact(cfg.run_dir, final["winner"], engine.profile.classes, final["metrics"])
    checks = json.loads((artifact / "verification.json").read_text("utf-8"))
    assert checks["checks"]["local"]["status"] == "passed", checks
    assert all(c["status"] == "passed" for c in checks["checks"]["local"]["checks"].values())
    assert checks["checks"]["docker"] == {"status": "not_run", "reason": "docker_not_installed"}
    assert checks["status"] == "not_run" and not checks["verified"]
    meta = json.loads((artifact / "metadata.json").read_text("utf-8"))
    assert meta["classes"] == engine.profile.classes
    assert meta["selection"] == final["winner"]
    assert file_sha256(artifact / "model.keras") == store.get("selection")["checkpoint_sha256"]
    assert packaging.build_artifact(cfg.run_dir, final["winner"], engine.profile.classes, final["metrics"]) == artifact
    store.put("verification", checks)
    store.put("delivery", {"status": "completed"})
    store.put("stage_timings", {"evaluar": 1.25, "empaquetar": 2.})
    facts = reporting.write_persistent_reports(store, cfg)
    assert facts["final_evaluation"] == final
    assert facts["candidate_versions"]["one:v2"]["rationale"] == "revisada"
    assert facts["stage_seconds_measured"]["empaquetar"] == 2.
    md = (cfg.run_dir / "reports/06_reporte_final.md").read_text("utf-8")
    assert "revisada" in md and "not_run" in md
    assert "candidate_version" in (cfg.run_dir / "reports/04_consumo_recursos.csv").read_text()
    # Regenerar después de reabrir SQLite no requiere ni datos ni objetos del motor.
    reopened = StateStore(cfg.run_dir)
    try:
        assert reporting.write_persistent_reports(reopened, cfg) == facts
    finally:
        reopened.close()
    (artifact / "app.py").write_text("corrupt")
    failed = packaging.verify_artifact(artifact, docker=False)
    assert failed["status"] == "failed" and not failed["verified"]
    assert failed["checks"]["local"]["status"] == "not_run"


@pytest.mark.parametrize("target,verification,expected", [
    (.8, "passed", "completed"), (.95, "passed", "target_not_reached"),
    (.8, "not_run", "completed"), (.8, "failed", "artifact_failed")])
def test_delivery_states_preserve_final_results(final_run, monkeypatch, target, verification, expected):
    from automl_agents import pipeline
    cfg, store, engine = final_run
    final = evaluation.evaluate_frozen_selection(store, engine)
    final["winner"]["metric_policy"]["threshold"] = target
    store.put("final_evaluation", final)
    art = cfg.run_dir / "artifact"
    art.mkdir()
    write_json(art / "verification.json", {"status": verification, "verified": verification == "passed"})
    monkeypatch.setattr(pipeline, "build_artifact", lambda *a, **k: art)
    pl = Pipeline.__new__(Pipeline)
    pl.cfg, pl.store, pl.log = cfg, store, SimpleNamespace(emit=lambda *a, **k: None)
    summary = pl.n_package({})["summary"]
    assert summary["status"] == expected
    assert summary["test"] and summary["winner"]["checkpoint"]
    assert summary["verification"]["verified"] == (verification == "passed")


def test_partial_deliveries_have_reports(final_run, monkeypatch):
    from automl_agents import pipeline
    cfg, store, engine = final_run
    pl = Pipeline.__new__(Pipeline)
    pl.cfg, pl.store = cfg, store
    selected = store.get("selection")
    store.put("selection", None)
    summary = pl._finish("no_model", "all_failed")
    reporting.write_persistent_reports(store, cfg)
    assert summary["status"] == "no_model"
    assert summary["winner"] is None and summary["verification"]["status"] == "not_run"
    assert (cfg.run_dir / "reports/06_reporte_final.md").exists()
    store.put("selection", selected)
    evaluation.evaluate_frozen_selection(store, engine)
    monkeypatch.setattr(pipeline, "build_artifact", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    summary = pl.n_package({})["summary"]
    reporting.write_persistent_reports(store, cfg)
    assert summary["status"] == "artifact_failed"
    assert summary["test"] and Path(summary["winner"]["checkpoint"]).exists()
    assert summary["verification"]["status"] == "failed"


@pytest.mark.parametrize("scenario,expected", [("missing", "not_run"), ("daemon", "not_run"),
    ("build", "failed"), ("timeout", "failed"), ("success", "passed")])
def test_docker_results_and_cleanup(tmp_path, monkeypatch, scenario, expected):
    calls = []
    monkeypatch.setattr(packaging.shutil, "which", lambda _: None if scenario == "missing" else "docker")
    def command(args, **kw):
        calls.append(args)
        if (scenario == "daemon" and args[1] == "info") or (scenario == "build" and args[1] == "build"):
            return {"status": "failed", "returncode": 1}
        if scenario == "timeout" and args[1] == "run":
            return {"status": "failed", "reason": "timeout"}
        assert kw["timeout"] <= 30
        return {"status": "passed", "returncode": 0}
    monkeypatch.setattr(packaging, "_command", command)
    result = packaging.verify_docker(tmp_path, timeout=30)
    assert result["status"] == expected
    if scenario in ("build", "timeout", "success"):
        assert calls[-2][1:3] == ["rm", "-f"]
        assert calls[-1][1:4] == ["image", "rm", "-f"]


def test_subprocess_timeout_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(a[0], 1)))
    result = packaging._command(["python", "smoke_test.py"], cwd=tmp_path, timeout=1)
    assert result["status"] == "failed" and result["reason"] == "timeout"


@pytest.mark.parametrize("local,docker,expected", [("passed", "passed", "passed"),
    ("passed", "not_run", "not_run"), ("failed", "passed", "failed"),
    ("passed", "failed", "failed"), ("incomplete", "passed", "failed")])
def test_verification_never_claims_unexecuted_or_failed_checks(tmp_path, monkeypatch, local, docker, expected):
    file = tmp_path / "app.py"
    file.write_text("test")
    write_json(tmp_path / "publication.json", {"files": {"app.py": file_sha256(file)}})
    def command(*args, **kwargs):
        checks = {name: {"status": "passed"} for name in
                  ("load_model", "preprocessing", "health", "labels", "predict", "invalid_image")}
        if local == "incomplete":
            checks.pop("invalid_image")
        write_json(tmp_path / "smoke_result.json", {"status": "passed", "checks": checks})
        return {"status": "failed" if local == "failed" else "passed"}
    monkeypatch.setattr(packaging, "_command", command)
    monkeypatch.setattr(packaging, "verify_docker", lambda *a: {"status": docker})
    result = packaging.verify_artifact(tmp_path)
    assert result["status"] == expected
    assert result["verified"] == (expected == "passed")


def test_cli_report_and_verification_refresh_saved_delivery(final_run, monkeypatch, capsys):
    from automl_agents.cli import main
    from automl_agents.jobs import JobQueue
    cfg, store, engine = final_run
    queue = JobQueue(cfg.run_dir.parent)
    queue.enqueue(cfg)
    queue.finish(cfg.run_id, "artifact_failed")
    final = evaluation.evaluate_frozen_selection(store, engine)
    store.put("delivery", {"status": "completed", "artifact": str(cfg.run_dir / "artifact"), "winner": final["winner"]})
    store.put("verification", {"status": "not_run", "verified": False})
    main(["report", cfg.run_id, "--runs-dir", str(cfg.run_dir.parent)])
    assert json.loads(capsys.readouterr().out)["final_evaluation"] == final
    monkeypatch.setattr(packaging, "verify_artifact", lambda *a, **k: {"status": "passed", "verified": True})
    main(["verify-artifact", cfg.run_id, "--runs-dir", str(cfg.run_dir.parent), "--timeout", "2"])
    assert json.loads(capsys.readouterr().out)["verified"]
    assert store.get("delivery")["verification"]["verified"]
    assert store.get("status") == "completed"
    assert queue.get(cfg.run_id)["status"] == "completed"
    assert json.loads((cfg.run_dir / "run_summary.json").read_text("utf-8"))["verification"]["verified"]
    monkeypatch.setattr(packaging, "verify_artifact", lambda *a, **k: {"status": "failed", "verified": False})
    main(["verify-artifact", cfg.run_id, "--runs-dir", str(cfg.run_dir.parent)])
    assert not json.loads(capsys.readouterr().out)["verified"]
    assert store.get("status") == queue.get(cfg.run_id)["status"] == "artifact_failed"


def test_failed_attempts_are_not_counted_as_early_stopping_savings():
    from automl_agents.benchmark import compute_benchmark
    cfg = RunConfig(dataset_dir="missing", max_epochs=10)
    records = [{"status": "completed", "epochs": 10, "pruned": False, "resources": {"seconds": 20.}},
               {"status": "failed", "epochs": 0, "pruned": False, "resources": {"seconds": 3.}},
               {"status": "cancelled", "epochs": 1, "pruned": False, "resources": {"seconds": 2.}}]
    b = compute_benchmark(records, cfg, {"candidate_configs": 3}, {"perfil": .1})
    assert b["saving_from_early_stopping_pct"] == 0
    assert b["seconds_measured"] == 25
    assert b["trial_status_counts"]["failed"] == 1
    assert b["stage_seconds_measured"]["perfil"] == .1


def test_selection_policy_tie_breaks_by_size_and_excludes_failed_missing_checkpoints(tmp_path):
    cfg = RunConfig(dataset_dir="x", output_dir=str(tmp_path), run_id="ties", use_llm=False)
    store = StateStore(cfg.run_dir)
    try:
        for index, (arch, status, score, exists) in enumerate([
                ("MobileNetV2", "completed", .8, True),
                ("MobileNetV3Small", "completed", .8, True),
                ("MobileNetV2", "failed", .99, True),
                ("MobileNetV2", "completed", .99, False)]):
            path = cfg.run_dir / f"{index}.keras"
            if exists:
                path.write_bytes(b"model")
            ref = dict(candidate_id=str(index), candidate_version=1, study_id=f"s{index}",
                       trial_id=f"t{index}", attempt_id=f"a{index}")
            meta = {**ref, "round": 1, "trial": index, "architecture": arch, "params": {}, "seed": 42}
            trial = TrialResult(reference=ref, status=status, seed=42, metrics={"accuracy": score},
                                checkpoint=str(path) if status == "completed" else None,
                                error={"code": "failed", "message": "failed", "recoverable": True}
                                if status == "failed" else None)
            store.begin_attempt(str(index), meta)
            store.confirm_attempt(store.make_record(meta, trial))
        BudgetManager(store, cfg).close_search("budget")
        assert store.get("selection")["candidate_id"] == "1"
        assert store.get("selection")["metric_policy"] == cfg.metric_policy.to_dict()
    finally:
        store.close()
