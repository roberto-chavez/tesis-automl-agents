"""Registro de eventos de la ejecución (JSONL). Lo consume la API para mostrar progreso."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("automl_agents")


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
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
