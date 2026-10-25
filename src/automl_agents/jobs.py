"""Cola durable de la web. SQLite conserva intenciones; el SO elige un worker.

El bloqueo global se mantiene durante toda la ejecución. Un segundo proceso puede
servir consultas, pero no reconciliar ni entrenar mientras otro posee el bloqueo.
Los pendientes arrancan automáticamente; los interrumpidos requieren reanudación.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time

from .config import RunConfig
from .events import write_json
from .state_store import StateStore, run_lock, TrainingBusyError

TERMINAL = {"completed", "target_not_reached", "no_model", "artifact_failed", "cancelled", "interrupted", "failed"}
RESUMABLE = {"cancelled", "interrupted", "failed"}


class JobQueue:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                run_id TEXT PRIMARY KEY, config TEXT NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL, message TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0)""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "jobs.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, cfg):
        # Publicar la configuración antes de hacer visible el job al worker.
        write_json(cfg.run_dir / "config.json", cfg.to_dict())
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES (?,?, 'queued',?,?,NULL,0)",
                       (cfg.run_id, json.dumps(cfg.to_dict()), time.time(), time.time()))

    def get(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list(self):
        with self.connect() as db:
            rows = db.execute("SELECT run_id,status,created,updated,message FROM jobs ORDER BY created DESC").fetchall()
        return [dict(row) for row in rows]

    def finish(self, run_id, status, message=None):
        if status not in TERMINAL:
            raise ValueError("Estado de job no válido")
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,updated=?,message=? WHERE run_id=?",
                       (status, time.time(), message, run_id))

    def cancel(self, run_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone()
            if not row or row["status"] not in ("queued", "running", "cancelling"):
                raise ValueError("Esta ejecución no se puede cancelar en su estado actual.")
            rd = self.root / run_id
            # El marcador se conserva ante reinicios y lo lee el callback por época.
            (rd / "cancel.requested").touch()
            status = "cancelled" if row["status"] == "queued" else "cancelling"
            db.execute("UPDATE jobs SET status=?,cancel_requested=1,updated=? WHERE run_id=?",
                       (status, time.time(), run_id))
        return status

    def resume(self, run_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone()
            if not row or row["status"] not in RESUMABLE:
                raise ValueError("Sólo se puede reanudar una ejecución cancelada, interrumpida o con error.")
            (self.root / run_id / "cancel.requested").unlink(missing_ok=True)
            db.execute("UPDATE jobs SET status='queued',cancel_requested=0,message=NULL,updated=? WHERE run_id=?",
                       (time.time(), run_id))

    def reconcile(self):
        """Sólo bajo el bloqueo global: ningún job puede seguir activo."""
        for job in self.list():
            if job["status"] not in ("running", "cancelling"):
                continue
            rd = self.root / job["run_id"]
            status = "interrupted"
            if (rd / "state.sqlite").exists():
                # No reconciliar un journal que una CLI aún tenga abierto.
                try:
                    with run_lock(rd):
                        store = StateStore(rd)
                        try:
                            store.reconcile()
                            saved = store.get("status")
                            status = saved if saved in TERMINAL else "interrupted"
                            store.put("status", status)
                        finally:
                            store.close()
                except RuntimeError:
                    continue
            self.finish(job["run_id"], status, "El servidor se reinició. Puede continuar la ejecución." if status == "interrupted" else None)

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY updated,created LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET status='running',updated=? WHERE run_id=?", (time.time(), row["run_id"]))
        return RunConfig.from_dict(json.loads(row["config"]))

    def defer(self, run_id):
        with self.connect() as db:
            db.execute("""UPDATE jobs SET status=CASE WHEN cancel_requested=1 THEN 'cancelled'
                ELSE 'queued' END, updated=?, message='Esperando a que termine otro entrenamiento.'
                WHERE run_id=?""", (time.time(), run_id))


def execute_job(cfg):
    from .pipeline import run_pipeline
    return run_pipeline(cfg)


class JobWorker:
    def __init__(self, queue, runner=None):
        self.queue = queue
        self.runner = runner or execute_job
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="automl-jobs")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.thread.join(timeout=2)

    def tick(self):
        with ExitStack() as stack:
            try:
                stack.enter_context(run_lock(self.queue.root / ".web-worker"))
            except RuntimeError:
                return
            self.queue.reconcile()
            cfg = self.queue.claim()
            if cfg is None:
                return
            try:
                summary = self.runner(cfg)
                status = summary.get("status", "failed")
                self.queue.finish(cfg.run_id, status)
            except TrainingBusyError:
                self.queue.defer(cfg.run_id)
            except Exception as exc:
                self.queue.finish(cfg.run_id, "failed", str(exc) if isinstance(exc, ValueError)
                                  else f"Error interno: {type(exc).__name__}. Consulte el registro de eventos.")

    def run(self):
        while not self.stopping.is_set():
            try:
                self.tick()
            except Exception:
                # Un fallo transitorio de almacenamiento no debe matar el worker.
                logging.getLogger(__name__).exception("No se pudo avanzar la cola de jobs")
            self.stopping.wait(.4)
