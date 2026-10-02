from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


@dataclass
class RunConfig:
    dataset_dir: str
    use_case: str = ""                       # descripción libre del usuario (R3.1)
    output_dir: str = field(default_factory=lambda: os.getenv("AUTOML_RUNS_DIR", "runs"))
    run_id: str = field(default_factory=_default_run_id)
    resource_tier: str = "low"               # low | medium | high
    allowed_architectures: list[str] | None = None
    max_candidates: int = 3
    rounds: int = 2
    trials_per_round: int = 4
    max_epochs: int = 8
    target_accuracy: float = 0.99            # parada anticipada de la búsqueda (R4.1)
    discard_margin: float = 0.08             # margen para descartar candidatos rezagados (R2.3)
    pretrained: bool = True                  # pesos ImageNet (False = smoke tests sin red)
    img_size_override: int | None = None
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    embedding_backend: str = field(default_factory=lambda: os.getenv("AUTOML_EMBEDDING", "auto"))
    kb_persist_dir: str | None = None
    llm_model: str = field(default_factory=lambda: os.getenv("AUTOML_LLM_MODEL", "claude-sonnet-5"))
    use_llm: bool = True

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id

    def to_dict(self) -> dict:
        return asdict(self)
