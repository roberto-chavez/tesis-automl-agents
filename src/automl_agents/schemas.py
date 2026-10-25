"""Contratos de dominio: sólo datos JSON, sin modelos ML ni clientes externos.

Los IDs de estudio, trial e intento son opacos y distintos: un reintento
conserva trial_id, pero recibe un attempt_id nuevo. La versión identifica
el espacio de una candidata y forma parte de toda referencia de ejecución.
"""
from __future__ import annotations

from typing import Annotated, Literal

from dataclasses import asdict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from .catalog import AUGMENT_GRID, BATCH_GRID, CATALOG, DROPOUT_GRID, LR_GRID

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Identifier = Annotated[str, Field(min_length=1)]


@dataclass(config=ConfigDict(extra="forbid", allow_inf_nan=False))
class Candidate:
    id: Identifier
    architecture: str
    rationale: str
    sources: list[str]
    lr_values: list[Annotated[float, Field(strict=True)]]
    dropout_values: list[Annotated[float, Field(strict=True)]]
    batch_sizes: list[PositiveInt]
    augment_values: list[StrictBool]
    img_size: PositiveInt
    version: PositiveInt = 1

    def __post_init__(self):
        if self.architecture not in CATALOG:
            raise ValueError("Arquitectura fuera del catálogo")
        for name, grid in (("lr_values", LR_GRID), ("dropout_values", DROPOUT_GRID),
                           ("batch_sizes", BATCH_GRID), ("augment_values", AUGMENT_GRID)):
            values = getattr(self, name)
            if not values or any(value not in grid for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} debe ser un subconjunto no vacío de la rejilla, sin duplicados")

    @property
    def reference(self) -> CandidateRef:
        return CandidateRef(candidate_id=self.id, candidate_version=self.version)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Candidate:
        return TypeAdapter(cls).validate_python(data)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class MetricPolicy(Contract):
    # Ampliar sólo cuando entrenamiento, poda y selección soporten la métrica.
    name: Literal["accuracy"] = "accuracy"
    direction: Literal["maximize"] = "maximize"
    threshold: Annotated[float, Field(ge=0, le=1)] = 0.99
    tie_breaker: Literal["smaller_model"] = "smaller_model"


class DomainError(Contract):
    code: Identifier
    message: Identifier
    recoverable: bool


class ImageEntry(Contract):
    path: Identifier
    label: NonNegativeInt
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: PositiveInt
    width: PositiveInt
    height: PositiveInt


class DatasetManifest(Contract):
    version: Literal[1] = 1
    classes: list[Identifier]
    images: list[ImageEntry]


class SplitManifest(Contract):
    version: Literal[1] = 1
    dataset_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    seed: NonNegativeInt
    val_fraction: Annotated[float, Field(gt=0, lt=1)]
    test_fraction: Annotated[float, Field(gt=0, lt=1)]
    train: list[Identifier]
    val: list[Identifier]
    test: list[Identifier]


class CandidateRef(Contract):
    candidate_id: Identifier
    candidate_version: PositiveInt


class TrialRef(CandidateRef):
    study_id: Identifier
    trial_id: Identifier
    attempt_id: Identifier


class Budget(Contract):
    max_training_attempts: PositiveInt
    max_search_seconds: Annotated[float, Field(gt=0)]
    max_decisions: PositiveInt
    max_llm_calls: NonNegativeInt
    max_retries: NonNegativeInt
    training_attempts: NonNegativeInt = 0
    search_seconds: Annotated[float, Field(ge=0)] = 0
    decisions: NonNegativeInt = 0
    llm_calls: NonNegativeInt = 0
    retries: NonNegativeInt = 0

    @model_validator(mode="after")
    def within_limits(self):
        # El tiempo real puede superar el plazo por la cancelación cooperativa
        # al final de una época; conservarlo evita ocultar ese exceso.
        for used, limit in (("training_attempts", "max_training_attempts"),
                            ("decisions", "max_decisions"), ("llm_calls", "max_llm_calls"),
                            ("retries", "max_retries")):
            if getattr(self, used) > getattr(self, limit):
                raise ValueError(f"{used} supera {limit}")
        return self


class PlanTask(Contract):
    task_id: Identifier
    description: Identifier
    dependencies: list[Identifier] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    result_id: Identifier | None = None


class Plan(Contract):
    tasks: list[PlanTask] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_dependencies(self):
        tasks = {task.task_id: task for task in self.tasks}
        if len(tasks) != len(self.tasks):
            raise ValueError("IDs de tarea duplicados")
        visited, active = set(), set()

        def visit(key):
            if key not in tasks or key in active:
                raise ValueError("Dependencia desconocida o cíclica")
            if key in visited:
                return
            active.add(key)
            for dep in tasks[key].dependencies:
                visit(dep)
            active.remove(key)
            visited.add(key)

        for key in tasks:
            visit(key)
        return self


class Decision(Contract):
    decision_id: Identifier
    action: Literal["retrieve", "propose", "train", "adjust", "discard", "diagnose", "continue", "read", "validate", "stop"]
    reason: Identifier
    candidate: CandidateRef | None = None
    sources: list[str] = Field(default_factory=list)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResult(Contract):
    result_id: Identifier
    tool: Identifier
    status: Literal["completed", "failed", "cancelled"]
    data: dict[str, JsonValue] = Field(default_factory=dict)
    error: DomainError | None = None

    @model_validator(mode="after")
    def error_matches_status(self):
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("Un resultado fallido requiere error; los demás no lo admiten")
        return self


class TrialResult(Contract):
    reference: TrialRef
    status: Literal["completed", "pruned", "failed", "cancelled", "interrupted"]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    seed: NonNegativeInt
    epochs: NonNegativeInt = 0
    checkpoint: str | None = None
    error: DomainError | None = None
    curves: dict[str, list[float]] = Field(default_factory=dict)
    resources: dict[str, JsonValue] = Field(default_factory=dict)
    stop_reason: str | None = None
    validation: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def failure_has_error(self):
        if self.status == "failed" and self.error is None:
            raise ValueError("Un trial fallido requiere un error estructurado")
        if self.status == "completed" and self.error is not None:
            raise ValueError("Un trial completado no admite error")
        return self


class RunState(Contract):
    run_id: Identifier
    status: Literal["pending", "running", "completed", "target_not_reached", "no_model",
                    "artifact_failed", "failed", "cancelled", "interrupted"] = "pending"
    phase: Literal["profile", "search", "evaluation", "packaging", "done"] = "profile"
    mode: Literal["rules", "agentic"]
    metric_policy: MetricPolicy = Field(default_factory=MetricPolicy)
    budget: Budget
    plan: Plan = Field(default_factory=Plan)
    candidates: list[CandidateRef] = Field(default_factory=list)
    trials: list[TrialRef] = Field(default_factory=list)
    error: DomainError | None = None

    @model_validator(mode="after")
    def consistent_references(self):
        keys = {(c.candidate_id, c.candidate_version) for c in self.candidates}
        if len(keys) != len(self.candidates):
            raise ValueError("Referencia de candidata duplicada")
        attempts, trials, studies = set(), {}, {}
        for ref in self.trials:
            key = (ref.candidate_id, ref.candidate_version)
            if key not in keys:
                raise ValueError("El trial referencia una versión de candidata desconocida")
            if ref.attempt_id in attempts:
                raise ValueError("ID de intento duplicado")
            attempts.add(ref.attempt_id)
            if studies.setdefault(ref.study_id, key) != key:
                raise ValueError("Un estudio pertenece a una única versión de candidata")
            if trials.setdefault(ref.trial_id, (ref.study_id, key)) != (ref.study_id, key):
                raise ValueError("Un trial pertenece a un único estudio y candidata")
        return self
