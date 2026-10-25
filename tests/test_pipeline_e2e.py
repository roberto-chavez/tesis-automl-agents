import json

import pytest

pytest.importorskip("tensorflow")

from automl_agents.config import RunConfig  # noqa: E402
from automl_agents.demo import make_demo_dataset  # noqa: E402
from automl_agents.pipeline import run_pipeline  # noqa: E402


def test_end_to_end_smoke(tmp_path, kb, monkeypatch):
    """Pipeline completo con presupuesto mínimo y sin pesos preentrenados (sin red)."""
    # Docker real se comprueba por separado; instalarlo no debe activar descargas
    # de imágenes/dependencias en una prueba que promete funcionar sin red.
    monkeypatch.setattr("automl_agents.packaging.shutil.which", lambda _: None)
    ds = make_demo_dataset(tmp_path / "ds", per_class=14, size=48)
    cfg = RunConfig(dataset_dir=str(ds), output_dir=str(tmp_path / "runs"), use_case="clasificar formas",
                    pretrained=False, img_size_override=64, allowed_architectures=["MobileNetV3Small"],
                    max_candidates=1, rounds=1, trials_per_round=2, max_epochs=2, use_llm=False)
    s = run_pipeline(cfg, kb=kb)
    rd = cfg.run_dir
    for f in ["reports/01_criterios_acotacion.md", "reports/02_historial_iteraciones.json",
              "reports/03_evaluacion.md", "reports/04_consumo_recursos.csv", "reports/05_benchmark.md",
              "reports/06_reporte_final.md", "artifact/model.keras", "artifact/app.py",
              "artifact/Dockerfile", "run_summary.json", "data/dataset_manifest.json",
              "data/split_manifest.json", "artifact/verification.json", "artifact/preprocessing.py",
              "reports/07_entrega.json", "final_evaluation.json"]:
        assert (rd / f).exists(), f
    assert json.loads((rd / "run_summary.json").read_text())["winner"]["architecture"] == "MobileNetV3Small"
    assert 0.0 <= s["test"]["accuracy"] <= 1.0
    winner = s["winner"]
    assert winner["candidate_version"] == 1
    assert len({winner[key] for key in ("study_id", "trial_id", "attempt_id")}) == 3
    saved_config = json.loads((rd / "config.json").read_text("utf-8"))
    assert saved_config["metric_policy"]["name"] == "accuracy"
    assert saved_config["max_training_attempts"] == 2

    from automl_agents.state_store import StateStore
    store = StateStore(rd)
    try:
        assert store.get("search_closed") and store.get("test_opened")
        assert store.get("selection")["attempt_id"] == winner["attempt_id"]
        assert store.get("selection")["checkpoint"] == winner["checkpoint"]
        from automl_agents.state_store import file_sha256
        assert store.get("selection")["checkpoint_sha256"] == file_sha256(rd / "artifact/model.keras")
        verification = store.get("verification")
        assert verification["checks"]["local"]["status"] == "passed", verification
        assert verification["verified"] == (verification["status"] == "passed")
        assert s["status"] in ("completed", "target_not_reached")
        assert s["verification"] == verification
        assert store.get("stage_timings")["empaquetar"] > 0
        metadata = json.loads((rd / "artifact/metadata.json").read_text("utf-8"))
        assert metadata["selection"] == winner
        assert metadata["classes"] == store.get("final_evaluation")["metrics"]["classes"]
        assert store.get("plan")["tasks"][-1]["status"] == "completed"
        assert store.get("orchestrator")["history"][-1]["result"]["tool"] == "close_search"
    finally:
        store.close()
