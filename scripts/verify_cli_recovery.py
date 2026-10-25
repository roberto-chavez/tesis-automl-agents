"""Recorrido CLI real con muerte del proceso durante el segundo intento.

Sin LLM ni pesos externos; Docker, si está disponible, puede descargar imágenes
y dependencias. Ejecutar desde la raíz del repositorio:
    python scripts/verify_cli_recovery.py runs/stage11/recovery
Conserva logs y verification.json; nunca interrumpe procesos ajenos al script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from automl_agents.demo import make_demo_dataset
from automl_agents.state_store import StateStore, file_sha256


def read_state(directory):
    store = StateStore(directory)
    try:
        return store.status(), store.attempts()
    finally:
        store.close()


def stop_process_tree(process):
    if process.poll() is None:
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
    process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="runs/integration-recovery")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runs = root / "jobs"
    previous = set(runs.iterdir()) if runs.exists() else set()
    dataset = make_demo_dataset(root / "images", per_class=10, size=40)
    env = {**os.environ, "AUTOML_EMBEDDING": "hash", "PYTHONIOENCODING": "utf-8"}
    cli = [sys.executable, "-m", "automl_agents.cli"]

    def command(name, *arguments):
        with (root / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run([*cli, *arguments], env=env, stdout=log,
                           stderr=subprocess.STDOUT, timeout=600, check=True)

    process = None
    with (root / "interrupted.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen([*cli, "run", "--dataset", str(dataset),
                "--runs-dir", str(runs), "--mode", "rules", "--no-llm", "--no-pretrained",
                "--archs", "MobileNetV3Small", "--max-candidates", "1", "--rounds", "1",
                "--trials", "3", "--epochs", "2", "--img-size", "32",
                "--max-training-attempts", "4", "--max-search-seconds", "600",
                "--docker-timeout-seconds", "300"], env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 480
            directory, confirmed = None, []
            while time.monotonic() < deadline and process.poll() is None:
                fresh = [p for p in runs.iterdir() if p not in previous and
                         (p / "state.sqlite").is_file()] if runs.exists() else []
                if fresh:
                    directory = fresh[0]
                    _, attempts = read_state(directory)
                    confirmed = [a for a in attempts if a["status"] == "completed"]
                    if confirmed and any(a["status"] == "running" for a in attempts):
                        break
                time.sleep(.05)
            else:
                raise AssertionError("No se observó un intento activo después del primer resultado; consulte interrupted.log")
        finally:
            if process is not None:
                stop_process_tree(process)

    before, attempts = read_state(directory)
    confirmed_ids = [a["id"] for a in confirmed]
    active = next(a for a in attempts if a["status"] == "running")
    command("status-before", "status", directory.name, "--runs-dir", str(runs))
    command("resume", "resume", directory.name, "--runs-dir", str(runs))
    after, recovered = read_state(directory)
    assert after["status"] in ("completed", "target_not_reached"), after
    assert all(next(a for a in recovered if a["id"] == i)["status"] == "completed" for i in confirmed_ids)
    assert next(a for a in recovered if a["id"] == active["id"])["status"] == "interrupted"
    retries = [a for a in recovered if a["metadata"]["trial_id"] == active["metadata"]["trial_id"]
               and a["id"] != active["id"]]
    assert len(retries) == 1 and retries[0]["status"] == "completed"
    assert retries[0]["metadata"]["seed"] == active["metadata"]["seed"]
    assert retries[0]["metadata"]["params"] == active["metadata"]["params"]
    assert after["budget_consumed"]["training_attempts"] == len(recovered) == 4
    assert after["budget_consumed"]["retries"] == 1
    command("resume-again", "resume", directory.name, "--runs-dir", str(runs))
    replay, replay_attempts = read_state(directory)
    assert recovered == replay_attempts
    assert replay["budget_consumed"] == after["budget_consumed"]
    command("report", "report", directory.name, "--runs-dir", str(runs))
    command("verify-artifact", "verify-artifact", directory.name, "--runs-dir", str(runs))
    command("status-after", "status", directory.name, "--runs-dir", str(runs))
    state, _ = read_state(directory)
    summary = json.loads((directory / "run_summary.json").read_text("utf-8"))
    metadata = json.loads((directory / "artifact/metadata.json").read_text("utf-8"))
    facts = json.loads((directory / "reports/07_entrega.json").read_text("utf-8"))
    winner = summary["winner"]
    assert metadata["selection"] == facts["final_evaluation"]["winner"] == winner
    assert state["selection"]["checkpoint"] == winner["checkpoint"]
    assert file_sha256(directory / "artifact/model.keras") == file_sha256(winner["checkpoint"]) == winner["checkpoint_sha256"]
    assert state["verification"]["checks"]["local"]["status"] == "passed"
    result = {"run_id": directory.name, "status": state["status"], "process_killed": True,
              "confirmed_before": confirmed_ids, "interrupted_attempt": active["id"],
              "retry_attempt": retries[0]["id"], "attempts": len(recovered),
              "budget_before": before["budget_consumed"], "budget_after": state["budget_consumed"],
              "replay_without_training": True, "checkpoint_consistent": True,
              "verification": state["verification"]}
    (root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
