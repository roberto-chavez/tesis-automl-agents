"""Bloques acotados, publicación inmediata y fallos sin GPU ni descargas."""
import json

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from automl_agents import trainer
from automl_agents.config import RunConfig
from automl_agents.dataset import prepare_dataset
from automl_agents.demo import make_demo_dataset
from automl_agents.events import CancellationToken, RunLogger
from automl_agents.schemas import Candidate, TrialResult
from automl_agents.search import SearchEngine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    cfg = RunConfig(dataset_dir=str(ds), output_dir=str(tmp_path / "runs"),
                    pretrained=False, max_epochs=2, use_llm=False)
    profile, files, _ = prepare_dataset(ds, cfg.run_dir, .15, .15, cfg.seed)
    monkeypatch.setattr(trainer, "configure_tf", lambda: {})
    monkeypatch.setattr("automl_agents.resources.gpu_snapshot", lambda: None)
    return SearchEngine(cfg, profile, files, RunLogger(cfg.run_dir))


def candidate(cid="one", version=1, size=32):
    return Candidate(cid, "MobileNetV3Small", "test", [], [0.001], [0.0], [16], [False], size, version)


class Model:
    def __init__(self, engine):
        self.engine = engine

    def predict(self, ds, verbose=0):
        return np.eye(self.engine.profile.n_classes)[self.engine.splits["val"][1]]

    def save(self, path):
        path.write_text("checkpoint", encoding="utf-8")


def simulated_training(engine, monkeypatch, statuses=None):
    statuses = iter(statuses or [])
    released = []
    monkeypatch.setattr(trainer, "release", lambda: released.append(True))
    monkeypatch.setattr(trainer, "batched", lambda *a: "dataset")
    monkeypatch.setattr(engine, "base_ds", lambda *a: "base")

    def train(*args, progress, on_epoch, **kwargs):
        progress.update(epochs=1, curves={"loss": [1.], "accuracy": [.5],
                                         "val_loss": [.7], "val_accuracy": [1.]})
        on_epoch(progress)
        action = next(statuses, "completed")
        if isinstance(action, BaseException):
            raise action
        if action == "cancelled":
            kwargs["cancellation"].cancel()
        return Model(engine), {**progress, "val_acc": 1., "pruned": action == "pruned",
                               "cancelled": action == "cancelled", "stop_reason": action}
    monkeypatch.setattr(trainer, "train_trial", train)
    return released


def test_block_preserves_failures_versions_and_checkpoints(engine, monkeypatch):
    released = simulated_training(engine, monkeypatch, [MemoryError("OOM"), "completed", "completed"])
    results = engine.run_block([candidate(), candidate("two", 2)], 3)
    assert [r.status for r in results] == ["failed", "completed", "completed"]
    assert results[0].error.code == "out_of_memory"
    assert results[0].error.recoverable
    assert results[1].reference.candidate_version == 2
    assert results[1].validation["f1_macro"] == 1.
    assert results[1].validation["confusion_matrix"]
    assert len({r.checkpoint for r in results[1:]}) == 2
    assert len(released) == 3
    for result in results:
        directory = engine.attempts_dir / result.reference.attempt_id
        assert TrialResult.model_validate_json((directory / "result.json").read_text("utf-8")) == result
        assert json.loads((directory / "curves.json").read_text("utf-8")) == result.curves
        assert result.seed >= engine.cfg.seed
    # El estudio continúa entre bloques para la misma versión.
    more = engine.run_block([candidate()], 1)
    assert more[0].reference.study_id == results[0].reference.study_id
    assert len(engine._studies[("one", 1)].trials) == 3
    assert engine._studies[("one", 1)].trials[0].state.name == "FAIL"


@pytest.mark.parametrize("status", ["pruned", "cancelled", KeyboardInterrupt()])
def test_partial_attempt_is_persisted(engine, monkeypatch, status):
    released = simulated_training(engine, monkeypatch, [status])
    results = engine.run_block([candidate()], 3)
    expected = "interrupted" if isinstance(status, KeyboardInterrupt) else status
    assert results[0].status == expected
    assert results[0].epochs == 1
    assert results[0].checkpoint is None
    assert engine.records[0]["saved_as_best"] is False
    if expected in ("cancelled", "interrupted"):
        assert engine.best_checkpoints == {}
        assert len(results) == 1
        assert len(released) == 1
    else:
        assert len(results) == 3
        assert engine._studies[("one", 1)].trials[0].state.name == "PRUNED"


def test_cancel_before_block_and_between_trials(engine, monkeypatch):
    token = CancellationToken()
    token.cancel()
    assert engine.run_block([candidate()], 3, cancellation=token) == []
    assert not engine._studies
    simulated_training(engine, monkeypatch)
    original = engine.execute_trial

    def execute(*args, **kwargs):
        result = original(*args, **kwargs)
        engine.cancel()
        return result
    monkeypatch.setattr(engine, "execute_trial", execute)
    assert len(engine.run_block([candidate()], 3)) == 1


def test_unexpected_failure_is_recorded_before_propagation(engine, monkeypatch):
    released = simulated_training(engine, monkeypatch, [RuntimeError("bug")])
    with pytest.raises(RuntimeError, match="bug"):
        engine.run_block([candidate()], 2)
    assert engine.results[0].error.recoverable is False
    assert len(list(engine.attempts_dir.glob("*/result.json"))) == 1
    assert released == [True]


def test_cache_is_bounded(engine, monkeypatch):
    monkeypatch.setattr(trainer, "load_base_dataset", lambda *a, **kw: object())
    for index in range(10):
        engine.base_ds("train", candidate(size=32 + index))
        engine.base_ds("val", candidate(size=32 + index))
        assert len(engine._ds) <= 2
        assert engine._ds_bytes <= trainer.CACHE_LIMIT_BYTES


def test_real_block_without_downloads(engine):
    results = engine.run_block([candidate()], 1)
    assert len(results) == 1
    result = results[0]
    assert result.status == "completed", result.error
    assert result.epochs == 2
    assert all(len(result.curves[key]) == 2 for key in ("loss", "accuracy", "val_loss", "val_accuracy"))
    assert result.metrics["accuracy"] == pytest.approx(max(result.curves["val_accuracy"]))
    assert result.checkpoint
    model = trainer.keras.saving.load_model(result.checkpoint)
    ds = trainer.batched(engine.base_ds("val", candidate()), 64, 0, False, 0)
    from automl_agents.evaluation import evaluate_validation
    assert evaluate_validation(model, ds, engine.splits["val"][1], engine.profile.classes) == result.validation
    trainer.release(model)


def test_epoch_callback_cancels_and_preserves_curves():
    token = CancellationToken()
    progress, saved = {}, []
    callback = trainer.ProgressCallback(progress, token, lambda p: saved.append(p["epochs"]))
    model = type("Model", (), {"stop_training": False})()
    callback.set_model(model)
    token.cancel()
    callback.on_epoch_end(0, {"loss": .7, "accuracy": .5, "val_loss": .6, "val_accuracy": .75})
    assert model.stop_training
    assert progress["cancelled"]
    assert progress["curves"]["val_accuracy"] == [.75]
    assert saved == [1]


def test_epoch_callback_reports_accuracy_and_prunes():
    class Trial:
        def report(self, value, step):
            self.reported = (value, step)

        def should_prune(self):
            return True

    trial = Trial()
    callback = trainer.PruneCallback(trial)
    model = type("Model", (), {"stop_training": False})()
    callback.set_model(model)
    callback.on_epoch_end(2, {"val_accuracy": .8, "val_loss": .3})
    assert trial.reported == (.8, 2)
    assert callback.pruned and model.stop_training


def test_pipeline_cancel_finishes_without_evaluation(engine, monkeypatch):
    from automl_agents.pipeline import Pipeline
    pipeline = Pipeline.__new__(Pipeline)
    from dataclasses import replace
    pipeline.cfg, pipeline.engine = replace(engine.cfg, img_size_override=32), engine
    pipeline.store, pipeline.profile = engine.store, engine.profile
    pipeline.kb, pipeline.llm = None, None
    pipeline.candidates = [candidate()]
    pipeline.store.put("tool_candidates", {c.id: c.to_dict() for c in pipeline.candidates})
    simulated_training(engine, monkeypatch, ["cancelled"])
    monkeypatch.setattr(pipeline, "n_profile", lambda s: {})
    monkeypatch.setattr(pipeline, "n_retrieve", lambda s: {})
    monkeypatch.setattr(pipeline, "n_propose", lambda s: {"round": 1})

    def forbidden(state):
        pytest.fail("Cancelar debe finalizar antes de control o evaluación")

    monkeypatch.setattr(pipeline, "n_control", forbidden)
    monkeypatch.setattr(pipeline, "n_evaluate", forbidden)
    state = pipeline.build().invoke({})
    assert state["summary"]["status"] == "cancelled"
    assert state["stop"]
    assert json.loads((engine.cfg.run_dir / "run_summary.json").read_text("utf-8"))["status"] == "cancelled"


def test_nonfinite_epoch_remains_serializable():
    progress = {"epochs": 1, "curves": {"loss": [.5]}}
    callback = trainer.ProgressCallback(progress)
    with pytest.raises(FloatingPointError):
        callback.on_epoch_end(1, {"loss": float("nan")})
    assert progress == {"epochs": 1, "curves": {"loss": [.5]}}
    json.dumps(progress, allow_nan=False)


def test_numerical_failure_allows_next_attempt(engine, monkeypatch):
    simulated_training(engine, monkeypatch, [FloatingPointError("NaN"), "completed"])
    results = engine.run_block([candidate()], 2)
    assert results[0].error.code == "numerical_error"
    assert results[0].error.recoverable
    assert results[1].status == "completed"
