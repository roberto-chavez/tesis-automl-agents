import json

import pytest

pytest.importorskip("tensorflow")

from automl_agents.config import RunConfig  # noqa: E402
from automl_agents.demo import make_demo_dataset  # noqa: E402
from automl_agents.pipeline import run_pipeline  # noqa: E402


def test_end_to_end_smoke(tmp_path, kb):
    """Pipeline completo con presupuesto mínimo y sin pesos preentrenados (sin red)."""
    ds = make_demo_dataset(tmp_path / "ds", per_class=14, size=48)
    cfg = RunConfig(dataset_dir=str(ds), output_dir=str(tmp_path / "runs"), use_case="clasificar formas",
                    pretrained=False, img_size_override=64, allowed_architectures=["MobileNetV3Small"],
                    max_candidates=1, rounds=1, trials_per_round=2, max_epochs=2, use_llm=False)
    s = run_pipeline(cfg, kb=kb)
    rd = cfg.run_dir
    for f in ["reports/01_criterios_acotacion.md", "reports/02_historial_iteraciones.json",
              "reports/03_evaluacion.md", "reports/04_consumo_recursos.csv", "reports/05_benchmark.md",
              "reports/06_reporte_final.md", "artifact/model.keras", "artifact/app.py",
              "artifact/Dockerfile", "run_summary.json"]:
        assert (rd / f).exists(), f
    assert json.loads((rd / "run_summary.json").read_text())["winner"]["architecture"] == "MobileNetV3Small"
    assert 0.0 <= s["test"]["accuracy"] <= 1.0
