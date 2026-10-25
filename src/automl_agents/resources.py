"""Instrumentación del consumo de recursos por configuración evaluada (R4.2)."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil


def gpu_snapshot() -> dict | None:
    """Utilización y memoria de GPU vía nvidia-smi; None si no hay GPU NVIDIA."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout.strip().splitlines()[0]
        util, mem = (float(x) for x in out.split(","))
        return {"util": util, "mem_mb": mem}
    except Exception:
        return None


@dataclass
class ResourceUsage:
    seconds: float = 0.0
    cpu_percent_avg: float = 0.0
    rss_mb_peak: float = 0.0
    gpu_util_avg: float | None = None
    gpu_mem_mb_peak: float | None = None
    _cpu: list = field(default_factory=list, repr=False)
    _gpu_u: list = field(default_factory=list, repr=False)

    def as_dict(self) -> dict:
        return {
            "seconds": round(self.seconds, 2),
            "cpu_percent_avg": round(self.cpu_percent_avg, 1),
            "rss_mb_peak": round(self.rss_mb_peak, 1),
            "gpu_util_avg": None if self.gpu_util_avg is None else round(self.gpu_util_avg, 1),
            "gpu_mem_mb_peak": self.gpu_mem_mb_peak,
        }


class ResourceMonitor:
    """Context manager que muestrea CPU/RAM/GPU en un hilo mientras se entrena."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.usage = ResourceUsage()
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def _sample(self):
        u = self.usage
        u._cpu.append(self._proc.cpu_percent(interval=None))
        u.rss_mb_peak = max(u.rss_mb_peak, self._proc.memory_info().rss / 1e6)
        g = gpu_snapshot()
        if g:
            u._gpu_u.append(g["util"])
            u.gpu_mem_mb_peak = max(u.gpu_mem_mb_peak or 0.0, g["mem_mb"])

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except (psutil.Error, OSError):
                pass

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=4)
        try:
            self._sample()
        except (psutil.Error, OSError):
            pass
        u = self.usage
        u.seconds = time.perf_counter() - self._t0
        u.cpu_percent_avg = sum(u._cpu) / len(u._cpu) if u._cpu else 0.0
        u.gpu_util_avg = sum(u._gpu_u) / len(u._gpu_u) if u._gpu_u else None
        return False
