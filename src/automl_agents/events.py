"""Registro de eventos de la ejecución (JSONL). Lo consume la API para mostrar progreso."""
from __future__ import annotations

import json
import logging
import threading
import time
import os
from tempfile import NamedTemporaryFile
from pathlib import Path

log = logging.getLogger("automl_agents")


class CancellationToken:
    """Solicitud cooperativa, comprobada entre trials y al finalizar cada época."""

    def __init__(self, run_dir: Path | None = None):
        self._event = threading.Event()
        self.path = Path(run_dir) / "cancel.requested" if run_dir else None

    def cancel(self):
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set() or bool(self.path and self.path.exists())


class PipelineCancelled(Exception):
    """Pausa cooperativa solicitada antes de un nodo del pipeline."""


def write_json(path: Path, data) -> None:
    """Publica un artefacto completo sin exponer un JSON escrito a medias."""
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def write_text(path: Path, text: str) -> None:
    """Publicación atómica en el mismo volumen, con datos vaciados antes de renombrar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RunLogger:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, stage: str, message: str, **data) -> None:
        rec = {"ts": round(time.time(), 2), "stage": stage, "message": message, **data}
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        log.info("[%s] %s", stage, message)

    @staticmethod
    def read(run_dir: Path) -> list[dict]:
        p = Path(run_dir) / "events.jsonl"
        if not p.exists():
            return []
        events = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # El escritor puede estar publicando la última línea.
                continue
        return events
