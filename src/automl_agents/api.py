"""Plataforma web + API (R3.1): carga de dataset, ejecución del pipeline y descarga de resultados."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from .config import RunConfig
from .dataset import DatasetError, profile_dataset
from .events import RunLogger

MAX_UNZIPPED = 2_000_000_000
MAX_MEMBERS = 50_000
PRESETS = {  # presupuestos de búsqueda comprensibles para no especialistas
    "rapido": dict(max_candidates=2, rounds=2, trials_per_round=3, max_epochs=5),
    "estandar": dict(max_candidates=3, rounds=2, trials_per_round=4, max_epochs=8),
    "exhaustivo": dict(max_candidates=4, rounds=3, trials_per_round=5, max_epochs=12),
}
_ID = re.compile(r"^[0-9a-z-]{8,40}$")

app = FastAPI(title="AutoML multiagente para clasificación de imágenes", version="0.1.0")
_pool = ThreadPoolExecutor(max_workers=1)  # una sola ejecución a la vez (GPU compartida)
_lock = threading.Lock()


def runs_dir() -> Path:
    p = Path(os.getenv("AUTOML_RUNS_DIR", "runs"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_status(rd: Path, status: str, **extra):
    with _lock:
        (rd / "status.json").write_text(json.dumps({"status": status, "updated": time.time(), **extra},
                                                   ensure_ascii=False), encoding="utf-8")


def _get_status(rd: Path) -> dict:
    p = rd / "status.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {"status": "desconocido"}


def _safe_extract(zf: zipfile.ZipFile, dest: Path):
    infos = zf.infolist()
    if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_UNZIPPED:
        raise DatasetError("El archivo ZIP es demasiado grande (máx. 2 GB descomprimido).")
    root = dest.resolve()
    for i in infos:
        target = (dest / i.filename).resolve()
        if root not in target.parents and target != root:
            raise DatasetError("El ZIP contiene rutas no permitidas.")
    zf.extractall(dest)


def _execute(cfg: RunConfig):
    from .pipeline import run_pipeline
    rd = cfg.run_dir
    _set_status(rd, "running")
    try:
        run_pipeline(cfg)
        _set_status(rd, "done")
    except DatasetError as e:
        _set_status(rd, "error", message=str(e))
    except Exception as e:
        _set_status(rd, "error", message=f"Error interno: {type(e).__name__}. Ver events.jsonl.")


@app.post("/api/runs")
async def create_run(file: UploadFile = File(...), use_case: str = Form(""), tier: str = Form("low"),
                     budget: str = Form("estandar")):
    if budget not in PRESETS or tier not in ("low", "medium", "high"):
        raise HTTPException(400, "Parámetros de presupuesto o recursos no válidos.")
    cfg = RunConfig(dataset_dir="", use_case=use_case[:1000], output_dir=str(runs_dir()),
                    resource_tier=tier, **PRESETS[budget])
    cfg.kb_persist_dir = str(runs_dir() / ".chroma")
    rd = cfg.run_dir
    (rd / "dataset").mkdir(parents=True)
    tmp = rd / "upload.zip"
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        with zipfile.ZipFile(tmp) as zf:
            _safe_extract(zf, rd / "dataset")
        prof, _ = profile_dataset(rd / "dataset")  # validación inmediata y con mensajes claros
    except zipfile.BadZipFile:
        shutil.rmtree(rd, ignore_errors=True)
        raise HTTPException(400, "El archivo no es un ZIP válido. Comprima la carpeta de su dataset en formato .zip.")
    except DatasetError as e:
        shutil.rmtree(rd, ignore_errors=True)
        raise HTTPException(400, str(e))
    finally:
        tmp.unlink(missing_ok=True)
    cfg.dataset_dir = prof.root
    _set_status(rd, "queued")
    _pool.submit(_execute, cfg)
    return {"run_id": cfg.run_id, "dataset": prof.to_dict()}


def _run_path(run_id: str) -> Path:
    if not _ID.match(run_id):
        raise HTTPException(404, "Ejecución no encontrada")
    rd = runs_dir() / run_id
    if not rd.is_dir():
        raise HTTPException(404, "Ejecución no encontrada")
    return rd


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    rd = _run_path(run_id)
    st = _get_status(rd)
    events = RunLogger.read(rd)
    out = {**st, "events": [{k: e[k] for k in ("stage", "message", "ts")} for e in events[-40:]]}
    if (rd / "run_summary.json").exists():
        out["summary"] = json.loads((rd / "run_summary.json").read_text("utf-8"))
    return out


@app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
def get_report(run_id: str):
    p = _run_path(run_id) / "reports" / "06_reporte_final.md"
    if not p.exists():
        raise HTTPException(404, "El reporte aún no está disponible.")
    return p.read_text("utf-8")


@app.get("/api/runs/{run_id}/download")
def download(run_id: str):
    rd = _run_path(run_id)
    if _get_status(rd).get("status") != "done":
        raise HTTPException(409, "La ejecución aún no termina.")
    out = rd / f"resultado-{run_id}.zip"
    if not out.exists():
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for sub in ("reports", "artifact"):
                for f in (rd / sub).rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(rd))
            z.write(rd / "run_summary.json", "run_summary.json")
    return FileResponse(out, filename=out.name, media_type="application/zip")


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "web" / "index4.html").read_text("utf-8")
