"""Caídas antes/después de publicar, confirmar y avanzar el grafo."""
from dataclasses import replace
import json
import subprocess
import sys

import pytest

from automl_agents.state_store import StateStore, run_lock


def metadata():
    return {"candidate_id": "one", "candidate_version": 1, "study_id": "study",
            "trial_id": "trial", "attempt_id": "attempt", "round": 1,
            "trial": 0, "architecture": "MobileNetV3Small", "params": {}, "seed": 42}


@pytest.mark.parametrize("published,confirmed", [(False, False), (True, False), (True, True)])
def test_process_death_reconciles_without_losing_consumption(tmp_path, published, confirmed):
    # os._exit evita finally, cierre de conexiones y liberación explícita del lock.
    script = """
import os, sys
from pathlib import Path
from automl_agents.state_store import StateStore, run_lock
from automl_agents.schemas import TrialResult
from automl_agents.events import write_json
directory = Path(sys.argv[1])
meta = __import__('json').loads(sys.argv[2])
with run_lock(directory):
    store = StateStore(directory)
    store.begin_attempt('slot', meta)
    if sys.argv[3] == 'True':
        model = directory / 'model.keras'
        model.write_text('complete')
        result = TrialResult(reference={k: meta[k] for k in ('candidate_id', 'candidate_version', 'study_id', 'trial_id', 'attempt_id')}, status='completed', seed=42, checkpoint=str(model), metrics={'accuracy': .8})
        write_json(directory / 'attempts' / 'attempt' / 'result.json', result.to_dict())
        if sys.argv[4] == 'True':
            store.confirm_attempt(store.make_record(meta, result))
    os._exit(17)
"""
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path),
                              json.dumps(metadata()), str(published), str(confirmed)])
    assert process.returncode == 17
    with run_lock(tmp_path):
        store = StateStore(tmp_path)
        store.reconcile()
        store.reconcile()  # Reconciliar también es idempotente.
        assert len(store.records()) == 1
        assert store.records()[0]["status"] == ("completed" if published else "interrupted")
        assert store.status()["budget_consumed"]["training_attempts"] == 1
        store.close()


def test_action_confirmation_is_immutable_and_lock_excludes_workers(tmp_path):
    store = StateStore(tmp_path)
    assert store.confirm_action("key", {"result": 1}) == {"result": 1}
    assert store.confirm_action("key", {"result": 2}) == {"result": 1}
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="proceso activo"):
            with run_lock(tmp_path):
                pytest.fail("Dos workers entraron en la misma ejecución")
    store.close()


def test_resume_cannot_clear_cancellation_of_an_active_run(engine):
    from automl_agents.pipeline import run_pipeline
    from automl_agents.state_store import RunBusyError
    marker = engine.cfg.run_dir / "cancel.requested"
    marker.touch()
    before = engine.store.status()["budget_consumed"]
    with run_lock(engine.cfg.run_dir):
        with pytest.raises(RunBusyError):
            run_pipeline(engine.cfg, resume=True)
    assert marker.exists()
    assert engine.store.status()["budget_consumed"] == before


# Reutiliza simulaciones de entrenamiento, pero SQLite, Optuna y LangGraph son reales.
from test_search_blocks import engine, candidate, simulated_training  # noqa: E402,F401


def restart(engine):
    from automl_agents.search import SearchEngine
    engine.store.reconcile()
    return SearchEngine(engine.cfg, engine.profile, engine.files, engine.log)


def test_two_confirmed_trials_and_one_interrupted_resume(engine, monkeypatch):
    simulated_training(engine, monkeypatch, ["completed", "completed", SystemExit(17)])
    with pytest.raises(SystemExit):
        engine.run_block([candidate()], 3, action_key="block")
    assert len(engine.records) == 2
    resumed = restart(engine)
    assert [r["status"] for r in resumed.records] == ["completed", "completed", "interrupted"]
    simulated_training(resumed, monkeypatch)
    results = resumed.run_block([candidate()], 3, action_key="block")
    assert len(results) == 3
    assert len(resumed.store.attempts()) == 4
    interrupted, retried = resumed.records[-2:]
    assert interrupted["trial_id"] == retried["trial_id"]
    assert interrupted["attempt_id"] != retried["attempt_id"]
    assert interrupted["result"]["seed"] == retried["result"]["seed"]
    from automl_agents.reporting import write_resources_csv
    from automl_agents.benchmark import compute_benchmark
    write_resources_csv(resumed.cfg.run_dir, resumed.records)
    assert compute_benchmark(resumed.records, resumed.cfg, {"candidate_configs": 1})["trials_total"] == 4
    assert resumed.store.status()["budget_consumed"]["retries"] == 1
    assert all(t.state.is_finished() for t in resumed._study(candidate()).trials)
    monkeypatch.setattr(resumed, "execute_trial", lambda *a, **kw: pytest.fail("Acción confirmada repetida"))
    assert resumed.run_block([candidate()], 3, action_key="block") == results


@pytest.mark.parametrize("boundary", ["publication", "confirmation", "optuna"])
def test_crash_at_result_boundaries(engine, monkeypatch, boundary):
    simulated_training(engine, monkeypatch)
    from automl_agents import search
    if boundary == "publication":
        original = search.write_json

        def crash(path, data):
            if path.name == "result.json":
                raise SystemExit(17)
            original(path, data)
        monkeypatch.setattr(search, "write_json", crash)
    elif boundary == "confirmation":
        monkeypatch.setattr(engine.store, "confirm_attempt", lambda r: (_ for _ in ()).throw(SystemExit(17)))
    else:
        study = engine._study(candidate())
        monkeypatch.setattr(study, "tell", lambda *a, **kw: (_ for _ in ()).throw(SystemExit(17)))
    with pytest.raises(SystemExit):
        engine.run_block([candidate()], 1, action_key="block")
    monkeypatch.undo()
    resumed = restart(engine)
    simulated_training(resumed, monkeypatch)
    if boundary != "publication":
        monkeypatch.setattr(resumed, "execute_trial", lambda *a, **kw: pytest.fail("Resultado publicado repetido"))
    resumed.run_block([candidate()], 1, action_key="block")
    assert len(resumed.store.attempts()) == (2 if boundary == "publication" else 1)
    assert resumed.records[-1]["status"] == "completed"
    assert all(t.state.is_finished() for t in resumed._study(candidate()).trials)
    assert resumed._study(candidate()).trials[-1].value == 1.


def test_configuration_fingerprint_reuses_result_for_same_seed_and_data(engine, monkeypatch):
    simulated_training(engine, monkeypatch)
    result = engine.run_block([candidate()], 1)[0]
    study = engine._study(candidate())
    trial = study.ask()
    meta = engine.store.attempts()[0]["metadata"]
    monkeypatch.setattr("automl_agents.trainer.train_trial", lambda *a, **kw: pytest.fail("Configuración repetida"))
    assert engine.execute_trial(candidate(), study, trial, retry=meta) == result
    assert len(engine.store.attempts()) == 1
    study.tell(trial, result.metrics["accuracy"])


def test_studies_survive_restart_and_incompatible_categories(engine):
    first = engine._study(candidate())
    trial = first.ask()
    trial.suggest_categorical("batch", [16])
    first.tell(trial, .5)
    resumed = restart(engine)
    assert resumed._study(candidate()).trials[0].value == .5
    changed = candidate()
    changed.batch_sizes = [16, 32]
    new = resumed._study(changed)
    assert changed.version == 2
    assert new.study_name != first.study_name
    again = candidate()
    again.batch_sizes = [16, 32]
    assert resumed._study(again).study_name == new.study_name


@pytest.mark.parametrize("boundary", ["training", "node_confirmation", "cancelled", "cancel_marker"])
def test_graph_resume_rebuilds_context_and_skips_confirmed_nodes(engine, monkeypatch, kb, boundary):
    from automl_agents.pipeline import Pipeline, run_pipeline
    from automl_agents.search import SearchEngine
    cfg = replace(engine.cfg, max_candidates=1, rounds=1, trials_per_round=3,
                  allowed_architectures=["MobileNetV3Small"], img_size_override=32)
    actions = (["completed", "completed", SystemExit(17), "completed"] if boundary == "training"
               else ["cancelled"] if boundary in ("cancelled", "cancel_marker") else [])
    simulated_training(engine, monkeypatch, actions)
    monkeypatch.setattr(SearchEngine, "base_ds", lambda *a: "base")
    calls = []

    def retrieve(self, state):
        calls.append("retrieve")
        self.docs = []
        return {}

    def package(self, state):
        calls.append("package")
        return {"summary": {"run_id": self.cfg.run_id, "attempts": len(self.engine.records)}}

    monkeypatch.setattr(Pipeline, "n_retrieve", retrieve)
    monkeypatch.setattr(Pipeline, "n_evaluate", lambda *a: {})
    monkeypatch.setattr(Pipeline, "n_benchmark", lambda *a: {})
    monkeypatch.setattr(Pipeline, "n_report", lambda *a: {})
    monkeypatch.setattr(Pipeline, "n_package", package)
    if boundary == "node_confirmation":
        original = StateStore.confirm_action

        def confirm_then_crash(self, key, value):
            result = original(self, key, value)
            if key == "node:rag":
                raise SystemExit(17)
            return result
        monkeypatch.setattr(StateStore, "confirm_action", confirm_then_crash)
    if boundary in ("cancelled", "cancel_marker"):
        assert run_pipeline(cfg, kb=kb)["status"] == "cancelled"
    else:
        with pytest.raises(SystemExit):
            run_pipeline(cfg, kb=kb)
    if boundary == "node_confirmation":
        monkeypatch.setattr(StateStore, "confirm_action", original)
    # Reiniciar sin conservar ningún objeto de Pipeline/SearchEngine.
    if boundary == "cancel_marker":
        from automl_agents.pipeline import resume_pipeline
        marker = cfg.run_dir / "cancel.requested"
        marker.touch()
        monkeypatch.setattr("automl_agents.pipeline.KnowledgeBase", lambda *a: kb)
        result = resume_pipeline(cfg.run_id, cfg.output_dir)
        assert not marker.exists()
    else:
        result = run_pipeline(cfg, kb=kb)
    expected = 3 if boundary == "node_confirmation" else 4
    assert result["attempts"] == expected
    assert calls == ["retrieve", "package"]
    assert run_pipeline(cfg, kb=kb) == result
    assert calls == ["retrieve", "package"]
    store = StateStore(cfg.run_dir)
    assert store.status()["status"] == "completed"
    assert store.status()["budget_consumed"]["training_attempts"] == expected
    store.close()


def test_cli_status_and_resume_use_same_runs_directory(tmp_path, monkeypatch, capsys):
    from automl_agents.cli import main
    from automl_agents.config import RunConfig
    from automl_agents.events import write_json
    from automl_agents import pipeline
    cfg = RunConfig(dataset_dir=str(tmp_path / "ds"), output_dir=str(tmp_path), run_id="one")
    write_json(cfg.run_dir / "config.json", cfg.to_dict())
    store = StateStore(cfg.run_dir)
    store.put("status", "interrupted")
    store.close()
    main(["status", "one", "--runs-dir", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["status"] == "interrupted"
    seen = []
    def resume(cfg, *, resume=False):
        assert resume
        seen.append(cfg)
        return {"ok": True}
    monkeypatch.setattr(pipeline, "run_pipeline", resume)
    main(["resume", "one", "--output", str(tmp_path)])
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert seen[0] == cfg
    monkeypatch.setenv("AUTOML_RUNS_DIR", str(tmp_path))
    main(["resume", "one"])
    assert seen[-1] == cfg
    with pytest.raises(SystemExit):
        main(["status", "../one", "--runs-dir", str(tmp_path)])
