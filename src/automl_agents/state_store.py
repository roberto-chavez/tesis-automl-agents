"""Journal SQLite y recuperación de publicaciones atómicas por ejecución.

El resultado en disco se publica antes de confirmarlo en SQLite. Una caída entre
ambas operaciones se recupera sin volver a ejecutar el trabajo de entrenamiento.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from .events import write_json
from .schemas import TrialResult


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunBusyError(RuntimeError):
    """La ejecución está protegida por el bloqueo de otro proceso."""


@contextmanager
def run_lock(run_dir):
    """Bloqueo del SO: se libera también si el proceso muere abruptamente."""
    path = Path(run_dir) / ".run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RunBusyError("La ejecución ya tiene un proceso activo") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def load_run(run_id, runs_dir=None):
    from .config import RunConfig
    root = Path(runs_dir or os.getenv("AUTOML_RUNS_DIR", "runs")).resolve()
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id inválido")
    data = json.loads((root / run_id / "config.json").read_text("utf-8"))
    return RunConfig.from_dict({**data, "output_dir": str(root), "run_id": run_id})


class TrainingBusyError(RuntimeError):
    """Otro pipeline usa los recursos del directorio de ejecuciones."""


@contextmanager
def training_lock(root):
    from contextlib import ExitStack
    with ExitStack() as stack:
        try:
            stack.enter_context(run_lock(Path(root) / ".training"))
        except RuntimeError as exc:
            raise TrainingBusyError("Hay otro entrenamiento activo. Espere a que termine.") from exc
        yield


class StateStore:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.run_dir / "state.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS actions (
                key TEXT PRIMARY KEY, result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, slot TEXT NOT NULL, status TEXT NOT NULL,
                metadata TEXT NOT NULL, record TEXT, started REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS attempts_slot ON attempts(slot);
        """)

    def close(self):
        self.db.close()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value, allow_nan=False)))

    def action(self, key):
        row = self.db.execute("SELECT result FROM actions WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def confirm_action(self, key, value):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO actions VALUES (?,?)", (key, json.dumps(value, allow_nan=False)))
        return self.action(key)

    def attempts(self, slot=None):
        rows = self.db.execute("SELECT * FROM attempts" + (" WHERE slot=?" if slot else "") +
                               " ORDER BY rowid", (slot,) if slot else ()).fetchall()
        return [{**dict(r), "metadata": json.loads(r["metadata"]),
                 "record": json.loads(r["record"]) if r["record"] else None} for r in rows]

    def begin_attempt(self, slot, metadata):
        with self.db:
            self.db.execute("INSERT INTO attempts VALUES (?,?, 'running',?,NULL,?)",
                            (metadata["attempt_id"], slot, json.dumps(metadata), time.time()))

    def confirm_attempt(self, record):
        result = TrialResult.model_validate(record["result"])
        with self.db:
            existing = self.db.execute("SELECT status,record FROM attempts WHERE id=?",
                                       (result.reference.attempt_id,)).fetchone()
            if existing is None:
                raise ValueError("No existe una reserva para este intento")
            if existing["status"] != "running":
                return json.loads(existing["record"])
            self.db.execute("UPDATE attempts SET status=?,record=? WHERE id=?",
                            (result.status, json.dumps(record, allow_nan=False), result.reference.attempt_id))
        return record

    def reconcile(self):
        """Sólo bajo run_lock: recuperar resultados completos y cerrar los activos."""
        for attempt in self.attempts():
            if attempt["status"] != "running":
                continue
            meta = attempt["metadata"]
            directory = self.run_dir / "attempts" / attempt["id"]
            path = directory / "result.json"
            if path.exists():
                result = TrialResult.model_validate_json(path.read_text("utf-8"))
                if result.reference.to_dict() != {k: meta[k] for k in result.reference.to_dict()}:
                    raise ValueError("El resultado no corresponde al intento reservado")
                if result.status == "completed" and (not result.checkpoint or not Path(result.checkpoint).is_file()):
                    raise ValueError("El resultado completado carece de checkpoint publicado")
            else:
                curves_path = directory / "curves.json"
                curves = json.loads(curves_path.read_text("utf-8")) if curves_path.exists() else {}
                resources_path = directory / "resources.json"
                resources = {"seconds": 0., "cpu_percent_avg": None, "rss_mb_peak": None,
                             "gpu_util_avg": None, "gpu_mem_mb_peak": None}
                if resources_path.exists():
                    resources.update(json.loads(resources_path.read_text("utf-8")))
                result = TrialResult(reference={k: meta[k] for k in (
                    "candidate_id", "candidate_version", "study_id", "trial_id", "attempt_id")},
                    status="interrupted", parameters=meta["params"], seed=meta["seed"],
                    epochs=max((len(v) for v in curves.values()), default=0), curves=curves,
                    resources=resources,
                    stop_reason="process_interrupted")
                write_json(path, result.to_dict())
            self.confirm_attempt(self.make_record(meta, result))

    @staticmethod
    def make_record(meta, result):
        return {**result.reference.to_dict(), "round": meta["round"],
                "architecture": meta["architecture"], "trial": meta["trial"],
                "params": result.parameters, "val_acc": result.metrics.get("accuracy", 0.),
                "epochs": result.epochs, "pruned": result.status == "pruned", "status": result.status,
                "resources": result.resources, "saved_as_best": result.status == "completed",
                "checkpoint": result.checkpoint, "result": result.to_dict()}

    def records(self):
        return [a["record"] for a in self.attempts() if a["record"]]

    def status(self):
        attempts = self.attempts()
        started = self.get("search_started")
        return {"status": self.get("status", "pending"), "node": self.get("node"),
                "phase": self.get("phase", "profile"), "stop_reason": self.get("stop_reason"),
                "closing_deadline": self.get("closing_deadline"),
                "plan": self.get("plan", {"tasks": []}),
                "selection": self.get("selection"),
                "delivery": self.get("delivery"),
                "verification": self.get("verification"),
                "stage_timings": self.get("stage_timings", {}),
                "budget_limits": self.get("limits", {}),
                "rag_budget_limits": self.get("rag_limits", {}),
                "rag_budget_consumed": {"documents": len(self.get("rag_ledger", {})),
                                        "tokens": sum(self.get("rag_ledger", {}).values()),
                                        "token_count_method": "estimated_utf8_bytes"},
                "last_context": {role: self.get("last_context:" + role) for role in
                                 ("orchestrator", "configuration", "diagnosis")},
                "budget_consumed": {"training_attempts": len(attempts), "decisions": 0, "llm_calls": 0,
                    "retries": len(attempts) - len({a["metadata"]["trial_id"] for a in attempts}),
                    "search_seconds": sum(a["record"]["resources"].get("seconds", 0)
                                          for a in attempts if a["record"]),
                    **self.get("consumed", {}),
                    **({"search_seconds": max(0., self.get("search_finished", time.time()) - started)} if started else {})},
                "attempts": [{"attempt_id": a["id"], "status": a["status"],
                              "trial_id": a["metadata"]["trial_id"]} for a in attempts]}
