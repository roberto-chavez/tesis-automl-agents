from __future__ import annotations

import os
import uuid
from dataclasses import asdict, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from dotenv import load_dotenv

from .catalog import CATALOG, TIER_ALLOWED
from .llm import DEFAULT_LLM_TIMEOUT_SECONDS
from .context import ModelContextBudget
from .schemas import Budget, MetricPolicy, NonNegativeInt, PositiveInt

load_dotenv()  # carga variables desde .env (si existe) hacia os.environ, sin sobrescribir las ya definidas


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


@dataclass(config=ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True))
class RunConfig:
    dataset_dir: str
    use_case: str = ""                       # descripción libre del usuario (R3.1)
    output_dir: str = field(default_factory=lambda: os.getenv("AUTOML_RUNS_DIR", "runs"))
    run_id: str = field(default_factory=_default_run_id)
    resource_tier: Literal["low", "medium", "high"] = "low"
    allowed_architectures: list[str] | None = None
    max_candidates: PositiveInt = 3
    rounds: PositiveInt = 2
    trials_per_round: PositiveInt = 4
    max_epochs: PositiveInt = 8
    target_accuracy: Annotated[float, Field(ge=0, le=1)] = 0.99
    discard_margin: Annotated[float, Field(ge=0, le=1)] = 0.08
    pretrained: bool = True                  # pesos ImageNet (False = smoke tests sin red)
    img_size_override: PositiveInt | None = None
    val_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.15
    test_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.15
    seed: NonNegativeInt = 42
    embedding_backend: Literal["auto", "hash", "sentence-transformers"] = field(default_factory=lambda: os.getenv("AUTOML_EMBEDDING", "auto"))
    kb_persist_dir: str | None = None
    llm_model: str = field(default_factory=lambda: os.getenv("AUTOML_LLM_MODEL", "gemini-3.8-flash"))
    use_llm: bool = True
    mode: Literal["rules", "agentic"] = "agentic"
    metric_policy: MetricPolicy | None = None
    # Tiempo exclusivo de búsqueda; el cierre dispone de una reserva separada.
    max_training_attempts: PositiveInt | None = None
    max_search_seconds: Annotated[float, Field(gt=0)] = 3600.0
    closing_reserve_seconds: Annotated[float, Field(gt=0)] = 60.0
    artifact_smoke_timeout_seconds: Annotated[float, Field(gt=0)] = 120.0
    docker_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0
    max_decisions: PositiveInt = 100
    max_llm_calls: NonNegativeInt = 20
    max_configuration_llm_calls: NonNegativeInt = 1
    max_diagnosis_llm_calls: NonNegativeInt = 3
    max_retries: NonNegativeInt = 1
    llm_timeout_seconds: Annotated[float, Field(gt=0)] = DEFAULT_LLM_TIMEOUT_SECONDS
    max_stagnant_blocks: PositiveInt = 3
    max_no_progress_actions: PositiveInt = 6
    rag_max_documents: PositiveInt = 32
    rag_max_tokens: PositiveInt = 32000
    context_read_max_chars: Annotated[int, Field(ge=1, le=8000)] = 4000
    llm_context_budget: ModelContextBudget = field(default_factory=ModelContextBudget)
    llm_model_budgets: dict[str, ModelContextBudget] = field(default_factory=dict)

    def __post_init__(self):
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction debe ser menor que 1")
        if self.allowed_architectures is not None:
            if not self.allowed_architectures or len(set(self.allowed_architectures)) != len(self.allowed_architectures):
                raise ValueError("allowed_architectures debe ser una lista no vacía sin duplicados")
            if any(name not in CATALOG for name in self.allowed_architectures):
                raise ValueError("allowed_architectures contiene arquitecturas fuera del catálogo")
            if not any(CATALOG[name].tier in TIER_ALLOWED[self.resource_tier] for name in self.allowed_architectures):
                raise ValueError("Ninguna arquitectura permitida es compatible con resource_tier")
        if self.metric_policy is None:
            self.metric_policy = MetricPolicy(threshold=self.target_accuracy)
        else:
            # La política es la fuente de verdad; target_accuracy mantiene la API previa.
            self.target_accuracy = self.metric_policy.threshold
        if self.max_training_attempts is None:
            self.max_training_attempts = self.max_candidates * self.rounds * self.trials_per_round

    @property
    def llm_enabled(self) -> bool:
        return self.use_llm and self.mode == "agentic" and self.max_llm_calls > 0

    def initial_budget(self) -> Budget:
        return Budget(max_training_attempts=self.max_training_attempts,
                      max_search_seconds=self.max_search_seconds, max_decisions=self.max_decisions,
                      max_llm_calls=self.max_llm_calls, max_retries=self.max_retries)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id

    def to_dict(self) -> dict:
        result = asdict(self)
        result["metric_policy"] = self.metric_policy.to_dict()
        result["llm_context_budget"] = self.llm_context_budget.to_dict()
        result["llm_model_budgets"] = {k: v.to_dict() for k, v in self.llm_model_budgets.items()}
        return result

    @classmethod
    def from_dict(cls, data: dict) -> RunConfig:
        return cls(**data)
