"""Reinicio real del servidor durante búsqueda; sin LLM ni pesos externos.

python scripts/verify_web_restart.py runs/integration-web-restart --port 8766
El servidor y sus jobs quedan aislados; sólo se eliminan sus propios procesos.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx

from verify_cli_recovery import stop_process_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="runs/integration-web-restart")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "WEB_SMOKE_ROOT": str(root / "server"),
           "WEB_SMOKE_PORT": str(args.port), "PYTHONIOENCODING": "utf-8"}
    server_script = Path(__file__).with_name("serve_web_smoke.py")
    base = f"http://127.0.0.1:{args.port}"
    process = None
    logs = []

    def start(name):
        nonlocal process
        log = (root / (name + ".log")).open("w", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen([sys.executable, "-u", str(server_script)], env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and process.poll() is None:
            try:
                if http.get("/").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(.1)
        raise AssertionError("El servidor de prueba no inició; consulte los logs")

    def wait(check, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = http.get(f"/api/runs/{run_id}")
            response.raise_for_status()
            data = response.json()
            if check(data):
                return data
            time.sleep(.1)
        raise AssertionError("No se alcanzó el estado esperado")

    try:
        with httpx.Client(base_url=base, timeout=15, trust_env=False) as http:
            # Un puerto ocupado podría corresponder a otro proceso del usuario.
            try:
                http.get("/")
            except httpx.TransportError:
                pass
            else:
                raise ValueError("El puerto ya está en uso; seleccione otro con --port")
            start("before-restart")
            with (root / "server/images.zip").open("rb") as dataset:
                response = http.post("/api/runs", files={"file": ("images.zip", dataset, "application/zip")},
                                     data={"budget": "rapido", "mode": "rules", "pretrained": "false"})
            response.raise_for_status()
            run_id = response.json()["run_id"]
            active = wait(lambda d: d["progress"]["current_trial"] is not None)
            attempt_id = active["progress"]["current_trial"]["attempt_id"]
            stop_process_tree(process)
            start("after-restart")
            interrupted = wait(lambda d: d["status"] == "interrupted", timeout=30)
            assert interrupted["capabilities"]["resume"]
            assert interrupted["budget"]["consumed"]["training_attempts"] == active["budget"]["consumed"]["training_attempts"]
            http.post(f"/api/runs/{run_id}/resume").raise_for_status()
            finished = wait(lambda d: d["status"] in (
                "completed", "target_not_reached", "failed", "artifact_failed", "no_model"))
            assert finished["status"] in ("completed", "target_not_reached"), finished
            assert finished["summary"]["verification"]["checks"]["local"]["status"] == "passed"
            assert finished["budget"]["consumed"]["retries"] == 1
            assert finished["progress"]["attempts_started"] <= finished["progress"]["attempts_limit"]
            for url in finished["downloads"].values():
                http.get(url).raise_for_status()
            result = {"run_id": run_id, "status": finished["status"],
                      "process_killed": True, "interrupted_attempt": attempt_id,
                      "after_restart": interrupted, "finished": finished}
            (root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"run_id": run_id, "status": finished["status"], "restart": "passed"}))
    finally:
        if process is not None:
            stop_process_tree(process)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
