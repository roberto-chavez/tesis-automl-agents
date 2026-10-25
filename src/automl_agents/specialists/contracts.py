"""Contratos y proyecciones de especialistas: sólo entrenamiento y validación."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..schemas import Candidate, CandidateRef, Contract, DomainError, PositiveInt, TrialRef


class TrainingProfile(Contract):
    scope: Literal["train"]
    n_images: PositiveInt
    n_classes: PositiveInt
    classes: list[str]
    counts: dict[str, PositiveInt]
    imbalance_ratio: Annotated[float, Field(ge=1)]
    median_size: tuple[PositiveInt, PositiveInt]
    min_side: PositiveInt

    @model_validator(mode="after")
    def consistent_counts(self):
        if (len(set(self.classes)) != self.n_classes or len(self.classes) != self.n_classes or
                set(self.counts) != set(self.classes) or sum(self.counts.values()) != self.n_images):
            raise ValueError("Perfil de entrenamiento inconsistente")
        return self


class ConfigurationConstraints(Contract):
    resource_tier: Literal["low", "medium", "high"]
    pretrained: bool
    max_candidates: PositiveInt
    img_size_override: PositiveInt | None = None
    use_case: str = ""


class Architecture(Contract):
    params_m: float
    img_size: PositiveInt


class Evidence(Contract):
    id: str
    title: str
    text: str
    kind: str | None = None
    score: float | None = None
    source: str | None = None
    date: str | None = None
    scope: str | None = None
    version: str = ""


class ConfigurationContext(Contract):
    profile: TrainingProfile
    constraints: ConfigurationConstraints
    catalog: dict[str, Architecture]
    evidence: list[Evidence]

    @classmethod
    def build(cls, profile, docs, cfg):
        from ..candidates import _allowed, _img_size
        data = profile.to_dict()
        return cls(profile={k: data[k] for k in TrainingProfile.model_fields},
                   constraints={k: getattr(cfg, k) for k in ConfigurationConstraints.model_fields},
                   catalog={n: {"params_m": s.params_m, "img_size": _img_size(s, cfg)}
                            for n, s in _allowed(cfg)[0].items()},
                   evidence=[Evidence(**{k: getattr(d, k) for k in Evidence.model_fields}) for d in docs])


class ConfigurationResult(Contract):
    candidates: Annotated[list[Candidate], Field(min_length=1)]
    mode: Literal["llm", "reglas"]
    note: str = ""
    error: DomainError | None = None
    discarded: list[dict[str, str]] = Field(default_factory=list)

    def validate_context(self, context):
        ids, architectures = set(), set()
        if len(self.candidates) > context.constraints.max_candidates:
            raise ValueError("Demasiadas candidatas")
        for c in self.candidates:
            c = Candidate.from_dict(c.to_dict())
            if c.id in ids or c.architecture in architectures or c.version != 1:
                raise ValueError("Candidatas iniciales duplicadas o con versión inválida")
            if c.architecture not in context.catalog or c.img_size != context.catalog[c.architecture].img_size:
                raise ValueError("Candidata fuera del catálogo permitido")
            if not set(c.sources) <= {d.id for d in context.evidence}:
                raise ValueError("Fuente desconocida")
            ids.add(c.id)
            architectures.add(c.architecture)
        return self


class ValidationMetrics(Contract):
    accuracy: float | None = None
    f1_macro: float | None = None


class ValidationSummary(Contract):
    accuracy: float | None = None
    f1_macro: float | None = None
    confusion_matrix: list[list[int]] = Field(default_factory=list)


class Curves(Contract):
    loss: list[float] = Field(default_factory=list)
    val_loss: list[float] = Field(default_factory=list)
    accuracy: list[float] = Field(default_factory=list)
    val_accuracy: list[float] = Field(default_factory=list)


class Resources(Contract):
    seconds: float | None = None
    cpu_percent_avg: float | None = None
    rss_mb_peak: float | None = None
    gpu_util_avg: float | None = None
    gpu_mem_mb_peak: float | None = None


class DiagnosisContext(Contract):
    reference: TrialRef
    status: Literal["completed", "pruned", "failed", "cancelled", "interrupted"]
    candidate: Candidate | None = None
    curves: Curves
    metrics: ValidationMetrics
    validation: ValidationSummary
    resources: Resources
    error: DomainError | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def candidate_matches_trial(self):
        if self.candidate is not None and (self.candidate.id, self.candidate.version) != (
                self.reference.candidate_id, self.reference.candidate_version):
            raise ValueError("La candidata no corresponde al intento observado")
        return self

    @classmethod
    def build(cls, record, candidate=None, evidence=()):
        result = record["result"]
        def project(name, contract):
            return {k: v for k, v in result.get(name, {}).items() if k in contract.model_fields}
        return cls(reference={k: record[k] for k in TrialRef.model_fields}, status=record["status"],
                   candidate=candidate, curves=project("curves", Curves),
                   metrics=project("metrics", ValidationMetrics), validation=project("validation", ValidationSummary),
                   resources=project("resources", Resources), error=result.get("error"), evidence=list(evidence))


class Observation(Contract):
    code: Literal["status", "error", "validation", "resources", "generalization_gap"]
    detail: str


class Hypothesis(Contract):
    code: Literal["overfitting", "insufficient_capacity", "resource_pressure", "other"]
    explanation: Annotated[str, Field(min_length=1, max_length=1000)]
    confidence: Literal["low", "medium"] = "low"


class Recommendation(Contract):
    action: Literal["continue", "review", "adjust", "discard"]
    reason: Annotated[str, Field(min_length=1, max_length=1000)]
    expected: CandidateRef | None = None
    candidate: Candidate | None = None

    @model_validator(mode="after")
    def action_fields(self):
        if self.action in ("adjust", "discard") and self.expected is None:
            raise ValueError("La recomendación requiere versión esperada")
        if (self.action == "adjust") != (self.candidate is not None):
            raise ValueError("Sólo un ajuste contiene una candidata revisada")
        return self

    def validate_context(self, context):
        if self.expected is not None and (context.candidate is None or
                                         self.expected != context.candidate.reference):
            raise ValueError("Recomendación para otra candidata o versión")
        if self.candidate is not None:
            c = Candidate.from_dict(self.candidate.to_dict())
            old = context.candidate
            if (c.id, c.version, c.architecture, c.img_size) != (
                    old.id, old.version + 1, old.architecture, old.img_size):
                raise ValueError("El diagnóstico sólo puede revisar la rejilla de la siguiente versión")
            if not set(c.sources) <= {d.id for d in context.evidence}:
                raise ValueError("Fuente desconocida")
        return self


class DiagnosisAdvice(Contract):
    # El proveedor sólo añade hipótesis; los hechos los calcula el host.
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    recommendation: Recommendation
    sources: list[str] = Field(default_factory=list)


class DiagnosisResult(DiagnosisAdvice):
    attempt_id: str
    status: Literal["completed", "pruned", "failed", "cancelled", "interrupted"]
    findings: list[Observation]
    mode: Literal["llm", "reglas"]
    error: DomainError | None = None

    def validate_context(self, context):
        if self.attempt_id != context.reference.attempt_id or self.status != context.status:
            raise ValueError("Diagnóstico para otro intento")
        if not set(self.sources) <= {d.id for d in context.evidence}:
            raise ValueError("Fuente desconocida")
        self.recommendation.validate_context(context)
        return self
