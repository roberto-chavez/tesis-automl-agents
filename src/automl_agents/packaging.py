"""Artefacto final listo para despliegue (R3.3): API FastAPI + Dockerfile + modelo."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .catalog import CATALOG

APP_PY = '''"""API de clasificación de imágenes generada automáticamente."""
import importlib
import io
import json
from pathlib import Path

import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

HERE = Path(__file__).parent
META = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
MODEL = keras.saving.load_model(HERE / "model.keras")
PREPROCESS = importlib.import_module("keras.applications." + META["preprocess_module"]).preprocess_input
SIZE = META["img_size"]

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
        img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((SIZE, SIZE))
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida (use JPG o PNG).")
    x = PREPROCESS(np.asarray(img, dtype="float32")[None])
    probs = MODEL.predict(x, verbose=0)[0]
    idx = probs.argsort()[::-1][: max(1, min(top_k, len(probs)))]
    return {"predictions": [{"class": META["classes"][i], "confidence": round(float(probs[i]), 4)} for i in idx]}
'''

SMOKE_PY = '''"""Prueba funcional rápida (smoke test) del artefacto: python smoke_test.py [imagen.jpg]"""
import io
import sys

from fastapi.testclient import TestClient
from PIL import Image

from app import META, app

client = TestClient(app)
assert client.get("/health").json()["status"] == "ok"
assert client.get("/labels").json()["classes"] == META["classes"]
if len(sys.argv) > 1:
    data = open(sys.argv[1], "rb").read()
else:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 60, 200)).save(buf, "PNG")
    data = buf.getvalue()
r = client.post("/predict", files={"file": ("x.png", data, "image/png")})
assert r.status_code == 200, r.text
print("OK", r.json())
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
Prueba rápida: `python smoke_test.py foto.jpg`.
"""


def build_artifact(run_dir: Path, winner: dict, classes: list[str], metrics: dict) -> Path:
    out = run_dir / "artifact"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "models" / f"{winner['candidate_id']}.keras", out / "model.keras")
    spec = CATALOG[winner["architecture"]]
    meta = {"architecture": winner["architecture"], "preprocess_module": spec.module,
            "img_size": winner["img_size"], "classes": classes, "params": winner["params"],
            "test_metrics": {k: metrics[k] for k in ("accuracy", "f1_macro", "n_test")}}
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "app.py").write_text(APP_PY, encoding="utf-8")
    (out / "smoke_test.py").write_text(SMOKE_PY, encoding="utf-8")
    (out / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (out / "requirements.txt").write_text(REQS, encoding="utf-8")
    (out / "README.md").write_text(README.replace("__CLASSES__", ", ".join(classes)), encoding="utf-8")
    return out
