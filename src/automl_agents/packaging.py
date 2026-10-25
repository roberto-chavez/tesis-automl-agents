"""Artefacto final listo para despliegue (R3.3): API FastAPI + Dockerfile + modelo."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from uuid import uuid4
from pathlib import Path

from .catalog import CATALOG
from .events import write_json
from .state_store import file_sha256, fingerprint

APP_PY = '''"""API de clasificación de imágenes generada automáticamente."""
import json
from pathlib import Path

import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from preprocessing import decode_and_preprocess

HERE = Path(__file__).parent
META = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
MODEL = keras.saving.load_model(HERE / "model.keras", compile=False)
SIZE = META["img_size"]
if tuple(MODEL.input_shape[1:]) != (SIZE, SIZE, 3) or MODEL.output_shape[-1] != len(META["classes"]):
    raise ValueError("El modelo no corresponde a las entradas/clases de metadata.json")

app = FastAPI(title="Clasificador de imágenes", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "architecture": META["architecture"]}


@app.get("/labels")
def labels():
    return {"classes": META["classes"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    try:
        x = decode_and_preprocess(await file.read(), SIZE, META["preprocess_module"])[None]
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida (use JPG o PNG).")
    probs = MODEL.predict(x, verbose=0)[0]
    if not np.isfinite(probs).all() or len(probs) != len(META["classes"]):
        raise HTTPException(status_code=500, detail="Predicciones inválidas del modelo")
    idx = probs.argsort()[::-1][: max(1, min(top_k, len(probs)))]
    return {"predictions": [{"class": META["classes"][i], "confidence": round(float(probs[i]), 4)} for i in idx]}
'''

SMOKE_PY = '''"""Comprueba carga, transformación y contratos HTTP: python smoke_test.py."""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
checks = {name: {"status": "not_run"} for name in
          ("load_model", "preprocessing", "health", "labels", "predict", "invalid_image")}

def check(name, fn):
    try:
        fn()
        checks[name] = {"status": "passed"}
        return True
    except Exception as exc:
        checks[name] = {"status": "failed", "reason": str(exc)}
        return False

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def load():
    global api, client
    import app as api
    from fastapi.testclient import TestClient
    client = TestClient(api.app)

if check("load_model", load):
    data = (HERE / "verification.png").read_bytes()
    reference = np.load(HERE / "verification_input.npy", allow_pickle=False)

    def preprocessing():
        actual = api.decode_and_preprocess(data, api.SIZE, api.META["preprocess_module"])
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-5)

    def health():
        r = client.get("/health")
        require(r.status_code == 200 and r.json() ==
                {"status": "ok", "architecture": api.META["architecture"]}, r.text)

    def labels():
        r = client.get("/labels")
        require(r.status_code == 200 and r.json()["classes"] == api.META["classes"], r.text)

    def predict():
        r = client.post("/predict?top_k=99999", files={"file": ("x.png", data, "image/png")})
        require(r.status_code == 200, r.text)
        predictions = r.json()["predictions"]
        expected = api.MODEL.predict(reference[None], verbose=0)[0]
        require(len(predictions) == len(api.META["classes"]), "Número incorrecto de clases")
        require(len({p["class"] for p in predictions}) == len(predictions), "Clases repetidas")
        for p, i in zip(predictions, expected.argsort()[::-1]):
            require(p["class"] == api.META["classes"][i], "Orden/etiqueta de predicción incorrecto")
            require(abs(p["confidence"] - float(expected[i])) <= 0.00006, "Confianza incorrecta")
        require(np.isfinite(expected).all() and abs(float(expected.sum()) - 1) < .001,
                "Probabilidades inválidas")

    def invalid():
        r = client.post("/predict", files={"file": ("bad.png", b"invalid image", "image/png")})
        require(r.status_code == 400, r.text)

    for name, fn in (("preprocessing", preprocessing), ("health", health), ("labels", labels),
                     ("predict", predict), ("invalid_image", invalid)):
        check(name, fn)

result = {"checks": checks, "status": "passed" if all(
    c["status"] == "passed" for c in checks.values()) else "failed"}
(HERE / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
sys.exit(0 if result["status"] == "passed" else 1)
'''

DOCKERFILE = """FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

REQS = "tensorflow-cpu>=2.16\nfastapi>=0.110\nuvicorn>=0.29\npillow>=10\nnumpy>=1.26\npython-multipart>=0.0.9\nhttpx>=0.27\n"

README = """# Artefacto del modelo

Contiene el modelo entrenado (`model.keras`) y una API lista para desplegar.

## Con Docker
```
docker build -t clasificador .
docker run -p 8000:8000 clasificador
```
## Sin Docker
```
pip install -r requirements.txt
uvicorn app:app --port 8000
```
## Uso
```
curl -F "file=@foto.jpg" http://localhost:8000/predict
```
Documentación interactiva en http://localhost:8000/docs. Clases: __CLASSES__.
Prueba automática: `python smoke_test.py`.
Los resultados automáticos están en `verification.json`: passed (aprobado),
failed (fallido) y not_run (no ejecutado). Sólo passed permite verified=true.
"""


def build_artifact(run_dir: Path, winner: dict, classes: list[str], metrics: dict, *,
                   smoke_timeout=120, docker_timeout=300) -> Path:
    from tempfile import TemporaryDirectory

    published = run_dir / "artifact"
    identity = fingerprint({"winner": winner, "classes": classes, "metrics": metrics})
    marker = published / "publication.json"
    if marker.exists() and json.loads(marker.read_text("utf-8")).get("identity") == identity:
        _check_integrity(published)
        if not (published / "verification.json").exists():
            verify_artifact(published, smoke_timeout=smoke_timeout, docker_timeout=docker_timeout)
        return published
    if published.exists():
        raise ValueError("El directorio artifact existente no corresponde a una publicación completa de esta selección")
    # El directorio definitivo sólo se hace visible cuando contiene todos los archivos.
    with TemporaryDirectory(prefix=".artifact-", dir=run_dir) as staging:
        out = Path(staging) / "artifact"
        out.mkdir()
        _write_artifact(out, run_dir, winner, classes, metrics)
        write_json(out / "publication.json", {"identity": identity,
                   "files": {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()}})
        out.replace(published)
    verify_artifact(published, smoke_timeout=smoke_timeout, docker_timeout=docker_timeout)
    return published


def _write_artifact(out: Path, run_dir: Path, winner: dict, classes: list[str], metrics: dict):
    checkpoint = winner.get("checkpoint") or run_dir / "models" / f"{winner['candidate_id']}.keras"
    digest = file_sha256(checkpoint)
    if winner.get("checkpoint_sha256", digest) != digest:
        raise ValueError("El checkpoint seleccionado cambió antes del empaquetado")
    shutil.copy(checkpoint, out / "model.keras")
    spec = CATALOG[winner["architecture"]]
    meta = {"architecture": winner["architecture"], "preprocess_module": spec.module,
            "img_size": winner["img_size"], "classes": classes, "params": winner["params"],
            "test_metrics": {k: metrics[k] for k in ("accuracy", "f1_macro", "n_test")},
            "selection": winner, "model_sha256": digest}
    from .preprocessing import PREPROCESSING, decode_and_preprocess
    import numpy as np
    from PIL import Image
    meta["preprocessing"] = PREPROCESSING
    shutil.copy(Path(__file__).with_name("preprocessing.py"), out / "preprocessing.py")
    # Patrón no uniforme y rectangular: detecta diferencias de resize/interpolación.
    pixels = (np.arange(37 * 53 * 3).reshape(37, 53, 3) % 256).astype("uint8")
    Image.fromarray(pixels).save(out / "verification.png")
    reference = decode_and_preprocess((out / "verification.png").read_bytes(), winner["img_size"], spec.module)
    np.save(out / "verification_input.npy", np.asarray(reference), allow_pickle=False)
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "app.py").write_text(APP_PY, encoding="utf-8")
    (out / "smoke_test.py").write_text(SMOKE_PY, encoding="utf-8")
    (out / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (out / "requirements.txt").write_text(REQS, encoding="utf-8")
    (out / "README.md").write_text(README.replace("__CLASSES__", ", ".join(classes)), encoding="utf-8")


def _check_integrity(path):
    manifest = json.loads((path / "publication.json").read_text("utf-8"))
    if not manifest.get("files"):
        raise ValueError("La publicación carece de un manifiesto de integridad")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or not (path / name).is_file() or file_sha256(path / name) != digest:
            raise ValueError(f"El artefacto publicado cambió: {name}")


def _command(args, *, cwd, timeout):
    started = time.perf_counter()
    try:
        result = subprocess.run(args, cwd=cwd, timeout=timeout, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        return {"status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "output": (result.stdout + result.stderr)[-6000:],
                "seconds": time.perf_counter() - started}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "timeout", "seconds": time.perf_counter() - started}
    except OSError as exc:
        return {"status": "failed", "reason": str(exc), "seconds": time.perf_counter() - started}


def verify_docker(path: Path, timeout=300):
    """Build y smoke HTTP dentro de un contenedor; plazo global y limpieza acotada."""
    docker = shutil.which("docker")
    if docker is None:
        return {"status": "not_run", "reason": "docker_not_installed"}
    started = time.monotonic()
    info = _command([docker, "info"], cwd=path, timeout=min(10, timeout))
    if info["status"] != "passed":
        return {"status": "not_run", "reason": "docker_unavailable", "probe": info}
    tag = "automl-verify-" + uuid4().hex
    steps = {}
    try:
        for name, args in (("build", [docker, "build", "-t", tag, "."]),
                           ("smoke", [docker, "run", "--rm", "--name", tag, tag, "python", "smoke_test.py"])):
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                steps[name] = {"status": "failed", "reason": "timeout"}
            else:
                steps[name] = _command(args, cwd=path, timeout=remaining)
            if steps[name]["status"] != "passed":
                return {"status": "failed", "steps": steps}
        return {"status": "passed", "steps": steps}
    finally:
        _command([docker, "rm", "-f", tag], cwd=path, timeout=10)
        _command([docker, "image", "rm", "-f", tag], cwd=path, timeout=10)


def verify_artifact(path: Path, *, docker=True, smoke_timeout=120, docker_timeout=300):
    """Publica hechos de verificación; una omisión jamás equivale a aprobación."""
    if smoke_timeout <= 0 or docker_timeout <= 0:
        raise ValueError("Los timeouts de verificación deben ser positivos")
    path = Path(path).resolve()
    checks = {}
    try:
        _check_integrity(path)
        checks["integrity"] = {"status": "passed"}
    except (ValueError, OSError) as exc:
        checks["integrity"] = {"status": "failed", "reason": str(exc)}
    if checks["integrity"]["status"] == "passed":
        result_path = path / "smoke_result.json"
        result_path.unlink(missing_ok=True)  # Nunca reutilizar un resultado anterior.
        checks["local"] = _command([sys.executable, "smoke_test.py"], cwd=path, timeout=smoke_timeout)
        if checks["local"]["status"] == "passed":
            try:
                smoke = json.loads(result_path.read_text("utf-8"))
                required = {"load_model", "preprocessing", "health", "labels", "predict", "invalid_image"}
                if smoke["status"] != "passed" or set(smoke["checks"]) != required or any(
                        c["status"] != "passed" for c in smoke["checks"].values()):
                    raise ValueError("Comprobaciones incompletas o fallidas")
                checks["local"]["checks"] = smoke["checks"]
            except (ValueError, OSError, KeyError, TypeError) as exc:
                checks["local"] = {"status": "failed", "reason": str(exc)}
        elif result_path.exists():
            try:
                checks["local"]["checks"] = json.loads(result_path.read_text("utf-8"))["checks"]
            except (ValueError, OSError, KeyError):
                pass
    else:
        checks["local"] = {"status": "not_run", "reason": "integrity_failed"}
    checks["docker"] = (verify_docker(path, docker_timeout)
                        if docker and checks["local"]["status"] == "passed" else
                        {"status": "not_run", "reason": "disabled" if not docker else "local_not_passed"})
    statuses = [c["status"] for c in checks.values()]
    status = "failed" if "failed" in statuses else "not_run" if "not_run" in statuses else "passed"
    result = {"status": status, "verified": status == "passed", "checks": checks}
    write_json(path / "verification.json", result)
    return result
