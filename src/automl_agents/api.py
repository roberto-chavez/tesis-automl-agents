"""Web accesible y API de jobs persistentes para clasificación de imágenes."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from .config import RunConfig
from .dataset import DatasetError, prepare_dataset
from .events import RunLogger
from .jobs import JobQueue, JobWorker, RESUMABLE, TERMINAL
from .state_store import StateStore

MAX_UPLOAD = 512_000_000
MAX_UNZIPPED = 2_000_000_000
MAX_MEMBERS = 50_000
PRESETS = {
    "rapido": dict(max_candidates=2, rounds=2, trials_per_round=3, max_epochs=5, max_search_seconds=600),
    "estandar": dict(max_candidates=3, rounds=2, trials_per_round=4, max_epochs=8, max_search_seconds=1800),
    "exhaustivo": dict(max_candidates=4, rounds=3, trials_per_round=5, max_epochs=12, max_search_seconds=3600),
}
_ID = re.compile(r"^[0-9a-z-]{8,40}$")
WEB = Path(__file__).parent / "web"


def runs_dir() -> Path:
    root = Path(os.getenv("AUTOML_RUNS_DIR", "runs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def queue():
    return JobQueue(runs_dir())


@asynccontextmanager
async def lifespan(app):
    worker = JobWorker(queue())
    worker.start()
    try:
        yield
    finally:
        worker.stop()


app = FastAPI(title="AutoML · Clasificador de imágenes", version="0.2.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


def _error(status, message, field=None, code="invalid_request"):
    raise HTTPException(status, {"message": message, "field": field, "code": code})


def _safe_extract(zf, dest):
    infos = zf.infolist()
    if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_UNZIPPED:
        raise DatasetError("El ZIP excede 2 GB descomprimido o 50 000 archivos.")
    root = dest.resolve()
    for info in infos:
        target = (dest / info.filename).resolve()
        if (root not in target.parents and target != root) or stat.S_ISLNK(info.external_attr >> 16):
            raise DatasetError("El ZIP contiene rutas no permitidas.")
        if info.flag_bits & 1:
            raise DatasetError("Quite la contraseña del ZIP antes de subirlo.")
    zf.extractall(dest)


@app.post("/api/runs")
def create_run(file: UploadFile = File(...), use_case: str = Form(""), tier: str = Form("low"),
               budget: str = Form("estandar"), mode: str = Form("rules"), pretrained: bool = Form(True)):
    if budget not in PRESETS:
        _error(400, "Seleccione un límite de ejecución válido.", "budget")
    if tier not in ("low", "medium", "high"):
        _error(400, "Seleccione un equipo válido.", "tier")
    if mode not in ("rules", "agentic"):
        _error(400, "Seleccione un modo válido.", "mode")
    if not file.filename or not file.filename.lower().endswith(".zip"):
        _error(400, "Seleccione un archivo .zip.", "file")
    root = runs_dir()
    cfg = RunConfig(dataset_dir="", use_case=use_case[:1000], output_dir=str(root),
                    resource_tier=tier, mode=mode, use_llm=mode == "agentic", pretrained=pretrained,
                    embedding_backend="hash", kb_persist_dir=str(root / ".chroma"), **PRESETS[budget])
    rd = cfg.run_dir
    (rd / "dataset").mkdir(parents=True)
    temporary = rd / "upload.zip"
    try:
        size = 0
        with temporary.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise DatasetError("El ZIP supera el máximo de 512 MB.", "upload_too_large")
                out.write(chunk)
        with zipfile.ZipFile(temporary) as zf:
            _safe_extract(zf, rd / "dataset")
        profile, _, _ = prepare_dataset(rd / "dataset", rd, cfg.val_fraction, cfg.test_fraction, cfg.seed)
        # prepare_dataset devuelve el perfil de entrenamiento, con raíz absoluta.
        cfg.dataset_dir = str(Path(profile.root).resolve())
        # The web presets constrain wall time, not an explicit epoch request.
        # Tiny-image CNNs need more cheap epochs than a frozen ImageNet head.
        if max(profile.median_size) <= 32:
            cfg.max_epochs = max(cfg.max_epochs, 20)
        from .events import write_json
        write_json(rd / "upload.json", {"filename": file.filename, "profile": profile.to_dict(),
                                       "budget": budget, "use_case": cfg.use_case})
        queue().enqueue(cfg)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError) as exc:
        shutil.rmtree(rd)
        _error(400, "El archivo no es un ZIP válido o usa una compresión no compatible.", "file", "invalid_zip")
    except DatasetError as exc:
        shutil.rmtree(rd)
        _error(400, str(exc), "file", exc.code)
    finally:
        temporary.unlink(missing_ok=True)
        file.file.close()
    return {"run_id": cfg.run_id, "status": "queued", "dataset": profile.to_dict()}


def _job(run_id):
    if not _ID.fullmatch(run_id):
        _error(404, "Ejecución no encontrada.")
    job = queue().get(run_id)
    if not job:
        _error(404, "Ejecución no encontrada.")
    return runs_dir() / run_id, job


def _read(path, default=None):
    return json.loads(path.read_text("utf-8")) if path.is_file() else default


def _snapshot(rd):
    if not (rd / "state.sqlite").exists():
        return {}, [], {}, []
    store = StateStore(rd)
    try:
        return (store.status(), store.attempts(), store.get("orchestrator", {}),
                store.get("training_profile", {}).get("warnings", []))
    finally:
        store.close()


def _downloads(rd, job, state):
    if job["status"] not in TERMINAL:
        return {}
    result = {}
    base = f"/api/runs/{job['run_id']}"
    if any((rd / "reports").glob("*")):
        result["reports"] = base + "/download?kind=reports"
    selection = state.get("selection") or {}
    checkpoint = Path(selection.get("checkpoint") or rd / "missing")
    if checkpoint.is_file() and checkpoint.resolve().is_relative_to(rd.resolve()):
        result["model"] = base + "/download?kind=model"
    verification = state.get("verification") or {}
    local = verification.get("checks", {}).get("local", {}).get("status")
    if (job["status"] in ("completed", "target_not_reached") and local == "passed"
            and (rd / "artifact/model.keras").is_file()):
        result["artifact"] = base + "/download?kind=artifact"
    if result:
        result["all"] = base + "/download"
    return result


@app.get("/api/runs")
def list_runs():
    return {"runs": queue().list()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    rd, job = _job(run_id)
    state, attempts, orchestrator, warnings = _snapshot(rd)
    records = [a["record"] for a in attempts if a["record"] and a["status"] == "completed"]
    best = max(records, key=lambda r: r["val_acc"], default=None)
    active = next((a for a in reversed(attempts) if a["status"] == "running"), None)
    current = None
    cfg = json.loads(job["config"])
    if active:
        curves = _read(rd / "attempts" / active["id"] / "curves.json", {})
        current = {"attempt_id": active["id"], "architecture": active["metadata"]["architecture"],
                   "epochs_completed": max((len(v) for v in curves.values()), default=0),
                   "epochs_limit": cfg["max_epochs"], "curves": curves}
    closed = state.get("phase") in ("evaluation", "packaging", "done") or state.get("selection") is not None
    can_cancel = job["status"] in ("queued", "running") and not closed
    summary = _read(rd / "run_summary.json") if job["status"] in TERMINAL else None
    return {"run_id": run_id, "status": job["status"], "message": job["message"],
            "created": job["created"], "updated": job["updated"],
            "execution": state, "plan": state.get("plan", {"tasks": []}),
            "active_task": next((t for t in state.get("plan", {}).get("tasks", []) if t["status"] == "running"), None),
            "stage": state.get("node"),
            "progress": {"attempts_finished": sum(a["status"] != "running" for a in attempts),
                         "attempts_started": len(attempts), "attempts_limit": cfg["max_training_attempts"],
                         "current_trial": current},
            "budget": {"consumed": state.get("budget_consumed", {}),
                       "limits": state.get("budget_limits") or RunConfig.from_dict(cfg).initial_budget().to_dict()},
            "best_validation": {"accuracy": best["val_acc"], "architecture": best["architecture"],
                                "metrics": best["result"]["metrics"]} if best else None,
            "stop_reason": (summary or {}).get("reason") or state.get("stop_reason"),
            "decisions": [{"reason": h.get("decision", {}).get("reason", ""),
                           "action": h.get("decision", {}).get("action"), "verified": h.get("verified")}
                          for h in orchestrator.get("history", [])[-12:]],
            "events": [{k: e.get(k) for k in ("stage", "message", "ts")} for e in RunLogger.read(rd)[-40:]],
            "summary": summary, "warnings": warnings, "dataset": _read(rd / "upload.json", {}),
            "capabilities": {"cancel": can_cancel, "resume": job["status"] in RESUMABLE},
            "downloads": _downloads(rd, job, state)}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    _job(run_id)
    if not get_run(run_id)["capabilities"]["cancel"]:
        _error(409, "La ejecución está cerrando resultados o no admite cancelación.")
    try:
        return {"run_id": run_id, "status": queue().cancel(run_id)}
    except ValueError as exc:
        _error(409, str(exc))


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str):
    _job(run_id)
    try:
        queue().resume(run_id)
    except ValueError as exc:
        _error(409, str(exc))
    return {"run_id": run_id, "status": "queued"}


@app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
def get_report(run_id: str):
    rd, job = _job(run_id)
    p = rd / "reports/06_reporte_final.md"
    if job["status"] not in TERMINAL or not p.is_file():
        _error(409, "El reporte aún no está disponible.")
    return p.read_text("utf-8")


@app.get("/api/runs/{run_id}/download")
def download(run_id: str, kind: str = "all"):
    rd, job = _job(run_id)
    state, _, _, _ = _snapshot(rd)
    available = _downloads(rd, job, state)
    if kind not in available:
        _error(409, "Esta descarga no está disponible en el estado actual.")
    if kind == "model":
        return FileResponse(state["selection"]["checkpoint"], filename=f"modelo-{run_id}.keras")
    with tempfile.NamedTemporaryFile(dir=rd, prefix="download-", suffix=".zip", delete=False) as temp:
        out = Path(temp.name)
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            for sub in ("reports", "artifact"):
                if sub not in available or kind not in ("all", sub):
                    continue
                for path in sorted((rd / sub).rglob("*")):
                    if path.is_file() and path.resolve().is_relative_to(rd.resolve()):
                        archive.write(path, path.relative_to(rd))
            if kind == "all":
                if "model" in available:
                    archive.write(state["selection"]["checkpoint"], "model.keras")
                if (rd / "run_summary.json").is_file():
                    archive.write(rd / "run_summary.json", "run_summary.json")
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    return FileResponse(out, filename=f"{kind}-{run_id}.zip", media_type="application/zip",
                        background=BackgroundTask(out.unlink, missing_ok=True))


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text("utf-8")
