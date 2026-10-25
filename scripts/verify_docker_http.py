"""Construye el artefacto y prueba Uvicorn por HTTP real en un contenedor local.

python scripts/verify_docker_http.py <artifact> <directorio-resultados>
Publica sólo en 127.0.0.1 y elimina su propio contenedor e imagen al terminar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import httpx

from automl_agents.packaging import _check_integrity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _check_integrity(artifact)
    metadata = json.loads((artifact / "metadata.json").read_text("utf-8"))
    tag = "automl-http-verify-" + uuid4().hex
    result = {"status": "failed", "architecture": metadata["architecture"], "checks": {}}

    def docker(*arguments, timeout=30):
        return subprocess.run(["docker", *arguments], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, check=True).stdout

    try:
        with (output / "build.log").open("w", encoding="utf-8") as log:
            subprocess.run(["docker", "build", "-t", tag, str(artifact)], stdout=log,
                           stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
        result["checks"]["build"] = "passed"
        docker("run", "-d", "--name", tag, "-p", "127.0.0.1::8000", tag)
        port = docker("port", tag, "8000/tcp").strip()
        assert port.startswith("127.0.0.1:") and "\n" not in port, port
        with httpx.Client(base_url="http://" + port, timeout=30, trust_env=False) as client:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if docker("inspect", "--format", "{{.State.Running}}", tag).strip() != "true":
                    raise AssertionError("El contenedor terminó antes de responder")
                time.sleep(.5)
            else:
                raise AssertionError("Uvicorn no respondió en el plazo de inicio")
            assert response.json() == {"status": "ok", "architecture": metadata["architecture"]}
            result["checks"]["health"] = "passed"
            labels = client.get("/labels")
            assert labels.status_code == 200 and labels.json()["classes"] == metadata["classes"]
            result["checks"]["labels"] = "passed"
            response = client.post("/predict?top_k=999", files={"file": (
                "verification.png", (artifact / "verification.png").read_bytes(), "image/png")})
            assert response.status_code == 200, response.text
            predictions = response.json()["predictions"]
            assert len(predictions) == len(metadata["classes"])
            assert {p["class"] for p in predictions} == set(metadata["classes"])
            assert all(0 <= p["confidence"] <= 1 for p in predictions)
            assert abs(sum(p["confidence"] for p in predictions) - 1) < .001
            result["predictions"] = predictions
            result["checks"]["predict"] = "passed"
            invalid = client.post("/predict", files={"file": ("bad.png", b"invalid", "image/png")})
            assert invalid.status_code == 400, invalid.text
            result["checks"]["invalid_image"] = "passed"
        result["versions"] = json.loads(docker("exec", tag, "python", "-c",
            "import json,sys,tensorflow,keras; print(json.dumps({'python':sys.version.split()[0],"
            "'tensorflow':tensorflow.__version__,'keras':keras.__version__}))"))
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            logs = subprocess.run(["docker", "logs", tag], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=30)
            (output / "container.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        except (subprocess.SubprocessError, OSError):
            pass
        cleanup = {}
        for name, arguments in (("container", ("rm", "-f", tag)),
                                ("image", ("image", "rm", "-f", tag))):
            try:
                docker(*arguments)
                cleanup[name] = "removed"
            except (subprocess.SubprocessError, OSError) as exc:
                cleanup[name] = str(exc)
        result["cleanup"] = cleanup
        (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
