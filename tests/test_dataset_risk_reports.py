import pytest

from automl_agents.benchmark import compute_benchmark
from automl_agents.config import RunConfig
from automl_agents.dataset import DatasetError, profile_dataset, split_files
from automl_agents.demo import make_demo_dataset
from automl_agents.reporting import wilson
from automl_agents.risk import assess_use_case


def test_profile_and_split(tmp_path):
    make_demo_dataset(tmp_path / "ds", per_class=20, imbalance=True)
    prof, files = profile_dataset(tmp_path / "ds")
    assert prof.n_classes == 3 and prof.imbalance_ratio == 2.0
    assert any("desbalanceadas" in w for w in prof.warnings)
    sp = split_files(files, prof.classes, 0.15, 0.15, 0)
    assert set(sp["train"][0]).isdisjoint(sp["test"][0]) and set(sp["val"][0]).isdisjoint(sp["test"][0])


def test_friendly_dataset_errors(tmp_path):
    (tmp_path / "solo_una").mkdir()
    with pytest.raises(DatasetError, match="al menos 2 clases"):
        profile_dataset(tmp_path)


def test_risk_flags():
    r = assess_use_case("Reconocimiento facial de candidatos en selección de personal")
    assert r["risk_level"].startswith("posible alto") and r["personal_data_flag"] is False or r["notes"]
    low = assess_use_case("Clasificar defectos en piezas metálicas")
    assert low["risk_level"] == "riesgo bajo aparente" and not low["notes"]
    assert assess_use_case("fotos de operarios en la línea")["personal_data_flag"]


def test_benchmark_math():
    cfg = RunConfig(dataset_dir="x", max_epochs=10)
    recs = [{"pruned": True, "epochs": 2, "resources": {"seconds": 2}},
            {"pruned": False, "epochs": 10, "resources": {"seconds": 10}}]
    b = compute_benchmark(recs, cfg, {"candidate_configs": 36})
    assert b["trials_pruned"] == 1 and b["saving_from_early_stopping_pct"] == 40.0
    assert b["baseline_exhaustive_candidate_space"]["epochs"] == 360


def test_wilson_interval():
    lo, hi = wilson(0.9, 100)
    assert 0.82 < lo < 0.9 < hi < 0.96
