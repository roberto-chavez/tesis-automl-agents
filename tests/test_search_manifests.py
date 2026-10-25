"""Integración de los manifiestos con el perfil y el motor, sin entrenar."""
import pytest

pytest.importorskip("tensorflow")

from automl_agents.config import RunConfig
from automl_agents.dataset import DatasetError, prepare_dataset
from automl_agents.demo import make_demo_dataset
from automl_agents.events import RunLogger
from automl_agents.search import SearchEngine


def test_search_reads_saved_splits_and_training_profile(tmp_path, monkeypatch):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    cfg = RunConfig(dataset_dir=str(ds), output_dir=str(tmp_path / "runs"), use_llm=False)
    profile, files, splits = prepare_dataset(ds, cfg.run_dir, .15, .15, cfg.seed)
    monkeypatch.setattr("automl_agents.search.trainer.configure_tf", lambda: {})
    def fail(*args, **kwargs):
        pytest.fail("El motor no debe volver a particionar")
    monkeypatch.setattr("automl_agents.dataset._partition", fail)
    engine = SearchEngine(cfg, profile, files, RunLogger(cfg.run_dir))
    resumed = SearchEngine(cfg, profile, files, RunLogger(cfg.run_dir))
    assert engine.splits == resumed.splits == splits
    assert engine.profile.scope == "train"
    assert engine.profile.n_images == len(splits["train"][0])
    assert engine.records == []


def test_changed_dataset_fails_before_tensorflow_configuration(tmp_path, monkeypatch):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    cfg = RunConfig(dataset_dir=str(ds), output_dir=str(tmp_path / "runs"), use_llm=False)
    profile, files, _ = prepare_dataset(ds, cfg.run_dir, .15, .15, cfg.seed)
    (ds / "circulo" / "000.png").unlink()
    def fail():
        pytest.fail("No debe configurar TensorFlow con datos incompatibles")
    monkeypatch.setattr("automl_agents.search.trainer.configure_tf", fail)
    with pytest.raises(DatasetError) as exc:
        SearchEngine(cfg, profile, files, RunLogger(cfg.run_dir))
    assert exc.value.code == "dataset_changed"
