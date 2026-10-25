"""Herramientas de dominio tipadas, con validación previa y permisos por rol.

Los IDs de artefacto apuntan a proyecciones JSON seguras en SQLite, nunca a
archivos elegidos por el solicitante (ni manifests, checkpoints o reportes mixtos).
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, JsonValue, ValidationError

from . import candidates as candidate_service
from .policies import BudgetManager, BudgetedLLM, PolicyError
from .schemas import Candidate, CandidateRef, Contract, DomainError, ToolResult, TrialRef
from .state_store import fingerprint, run_lock
from .specialists import ConfigurationSpecialist, DiagnosisSpecialist
from .specialists.contracts import ConfigurationContext, ConfigurationResult, DiagnosisContext, DiagnosisResult, Evidence
from .specialists.diagnosis import diagnose_rules
from .context import ContextManager
from .kb import EvidenceManager

Role = Literal["orchestrator", "configuration", "diagnosis"]
BoundedID = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]


class EvidenceInput(Contract):
    query: Annotated[str, Field(min_length=1, max_length=2000)]
    k: Annotated[int, Field(strict=True, ge=1, le=10)] = 3
    kinds: Annotated[list[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")]],
                     Field(min_length=1, max_length=10)] | None = None


class ProposeInput(Contract):
    pass


class ReviseInput(Contract):
    expected: CandidateRef
    candidate: Candidate


class TrainInput(Contract):
    candidates: Annotated[list[CandidateRef], Field(min_length=1)]
    max_attempts: Annotated[int, Field(strict=True, ge=1)]


class CandidateInput(Contract):
    expected: CandidateRef


DiagnosisOutput = DiagnosisResult


class DiscardOutput(Contract):
    candidate: CandidateRef
    discarded: bool


class ValidationInput(Contract):
    attempt_id: BoundedID


class ReadInput(Contract):
    artifact_id: BoundedID
    offset: Annotated[int, Field(strict=True, ge=0)] = 0
    length: Annotated[int, Field(strict=True, ge=1, le=8000)] = 2000


class CloseInput(Contract):
    reason: Annotated[str, Field(min_length=1, max_length=1000)]


class EvidenceReference(Contract):
    id: str
    title: str
    kind: str | None = None
    score: float | None = None
    artifact_id: BoundedID
    version: str = ""
    source: str | None = None
    date: str | None = None
    scope: str | None = None


class EvidenceOutput(Contract):
    documents: list[EvidenceReference]
    # Los resultados confirmados antes de la etapa 8 sólo contenían documents.
    retrieval_id: str = ""
    evidence_revision: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    limits: dict[str, int] = Field(default_factory=dict)
    omitted: int = 0
    token_count_method: str = "estimated_utf8_bytes"


class CandidatesOutput(Contract):
    candidates: list[Candidate]


class SafeTrialResult(Contract):
    reference: TrialRef
    status: Literal["completed", "pruned", "failed", "cancelled", "interrupted"]
    metrics: dict[str, float]
    error: DomainError | None = None
    artifact_id: BoundedID


class TrainOutput(Contract):
    results: list[SafeTrialResult]
    stop_reason: str | None = None


class ValidationOutput(Contract):
    metrics: dict[str, float]
    validation: dict[str, JsonValue]


class ReadOutput(Contract):
    text: str
    total_chars: int
    next_offset: int | None = None
    version: str = ""


class CloseOutput(Contract):
    reason: str
    phase: Literal["evaluation"] = "evaluation"


@dataclass(frozen=True)
class ToolSpec:
    input: type[Contract]
    output: type[Contract]
    phases: tuple[str, ...]
    roles: tuple[str, ...]
    effects: tuple[str, ...]
    recoverable_errors: tuple[str, ...] = ("invalid_input", "permission_denied", "phase_denied",
        "budget_exhausted", "stale_version", "unknown_source", "not_found", "idempotency_conflict")


ALL_ROLES = ("orchestrator", "configuration", "diagnosis")
REGISTRY = {
    "query_evidence": ToolSpec(EvidenceInput, EvidenceOutput, ("profile", "search"), ALL_ROLES, ("persist_evidence",)),
    "propose_candidates": ToolSpec(ProposeInput, CandidatesOutput, ("search",), ("orchestrator",), ("persist_candidates", "llm_call")),
    "revise_candidate": ToolSpec(ReviseInput, CandidatesOutput, ("search",), ("orchestrator",), ("version_candidate",)),
    "train_block": ToolSpec(TrainInput, TrainOutput, ("search",), ("orchestrator",), ("training_attempts", "checkpoints")),
    "discard_candidate": ToolSpec(CandidateInput, DiscardOutput, ("search",), ("orchestrator",), ("discard_candidate",)),
    "diagnose_trial": ToolSpec(ValidationInput, DiagnosisOutput, ("search",), ("orchestrator",), ("persist_diagnosis", "llm_call")),
    "evaluate_validation": ToolSpec(ValidationInput, ValidationOutput, ("search",), ("orchestrator", "diagnosis"), ()),
    "read_artifact": ToolSpec(ReadInput, ReadOutput, ("profile", "search", "evaluation"), ALL_ROLES, ()),
    "close_search": ToolSpec(CloseInput, CloseOutput, ("search", "evaluation"), ("orchestrator",), ("close_search_permanently",)),
}


class DomainTools:
    def __init__(self, cfg, store, *, kb=None, profile=None, engine=None, llm=None, lock_held=False):
        self.cfg, self.store = cfg, store
        self.lock_held = lock_held  # Sólo el host bajo run_lock puede establecerlo.
        self.kb, self.profile, self.engine, self.llm = kb, profile, engine, llm
        self.budget = BudgetManager(store, cfg)
        self.evidence = EvidenceManager(store, cfg)
        self.references = ContextManager(store, cfg).refs

    def _specialist_llm(self, role):
        if not self.cfg.llm_enabled or self.llm is None:
            return None
        # Un único wrapper por llamada; nunca anidar reservas ni exponer estado al especialista.
        client = self.llm.client if isinstance(self.llm, BudgetedLLM) else self.llm
        llm = BudgetedLLM(client, self.store, self.cfg, role=role)
        llm.budget.admitted = True  # La decisión de la herramienta ya está reservada.
        return llm

    def _diagnosis_context(self, attempt_id):
        record = next(r for r in self.store.records() if r["attempt_id"] == attempt_id)
        data = self.store.get("candidate_history", {}).get(
            f"{record['candidate_id']}:v{record['candidate_version']}")
        current = self.store.get("tool_candidates", {}).get(record["candidate_id"])
        if data is None and current and current["version"] == record["candidate_version"]:
            data = current
        evidence = [Evidence.model_validate(d) for d in self.store.get("tool_evidence", {}).values()]
        return DiagnosisContext.build(record, Candidate.from_dict(data) if data else None, evidence)

    def validate_recommendation(self, diagnosis):
        """Revalidar contra el estado actual justo antes de aplicar un consejo."""
        result = DiagnosisResult.model_validate(diagnosis)
        context = self._diagnosis_context(result.attempt_id)
        result.validate_context(context)
        if result.findings != diagnose_rules(context).findings:
            raise PolicyError("invalid_input", "Los hallazgos no coinciden con las observaciones")
        rec = result.recommendation
        if rec.expected:
            self._candidate(rec.expected)
        if rec.candidate:
            self._validate_candidate(rec.candidate)
        return rec

    def _log_specialist(self, role, action_id, context, result):
        # El contexto guardado es la misma proyección segura enviada al especialista.
        self.store.put("specialist:" + action_id, {"role": role, "context": context.to_dict(),
                                                   "result": result.to_dict()})

    def _candidate(self, ref):
        data = self.store.get("tool_candidates", {}).get(ref.candidate_id)
        if data is None:
            raise PolicyError("not_found", "Candidata desconocida")
        if data["version"] != ref.candidate_version:
            raise PolicyError("stale_version", "La versión esperada ya no es actual")
        return Candidate.from_dict(data)

    def _validate_candidate(self, cand):
        # Revalidar incluso dataclasses mutadas después de su construcción.
        cand = Candidate.from_dict(cand.to_dict())
        allowed, _ = candidate_service._allowed(self.cfg)
        if cand.architecture not in allowed:
            raise PolicyError("invalid_input", "Arquitectura no permitida por recursos/configuración")
        if cand.img_size != candidate_service._img_size(allowed[cand.architecture], self.cfg):
            raise PolicyError("invalid_input", "Resolución fuera de la configuración permitida")
        if not set(cand.sources) <= set(self.store.get("tool_evidence", {})):
            raise PolicyError("unknown_source", "Fuente no recuperada en esta ejecución")
        signature = fingerprint({k: cand.to_dict()[k] for k in (
            "architecture", "img_size", "lr_values", "dropout_values", "batch_sizes", "augment_values")})
        registered = self.store.get("spaces", {}).get(cand.id, {}).get(str(cand.version))
        if registered is not None and registered != signature:
            raise PolicyError("stale_version", "El espacio cambió sin crear una versión nueva")
        return cand

    def _artifact(self, artifact_id, role):
        return self.references.get(artifact_id, role)

    def _publish(self, data, scope):
        return self.references.publish(data, scope)

    def call(self, name: str, arguments: dict, *, role: Role, action_id: str) -> ToolResult:
        """El rol lo suministra el host, nunca los argumentos del agente.

        Serialización por ejecución; el journal de entrenamiento recupera bloques
        interrumpidos. Repeticiones confirmadas no vuelven a gastar presupuesto.
        """
        try:
            with (nullcontext() if self.lock_held else run_lock(self.cfg.run_dir)):
                return self._call(name, arguments, role, action_id)
        except (PolicyError, ValidationError, ValueError) as exc:
            error = exc.error if isinstance(exc, PolicyError) else {
                "code": "invalid_input", "message": str(exc) or "Entrada inválida", "recoverable": True}
            return ToolResult(result_id=uuid4().hex, tool=name or "unknown", status="failed", error=error)
        except Exception as exc:
            return ToolResult(result_id=uuid4().hex, tool=name or "unknown", status="failed",
                              error={"code": "tool_failed", "message": str(exc) or type(exc).__name__,
                                     "recoverable": False})

    def _call(self, name, arguments, role, action_id):
        spec = REGISTRY.get(name)
        if spec is None:
            raise PolicyError("invalid_input", "Herramienta desconocida")
        if role not in spec.roles:
            raise PolicyError("permission_denied", "Rol sin permiso para esta herramienta")
        if not action_id or len(action_id) > 128:
            raise PolicyError("invalid_input", "action_id requerido y acotado")
        args = spec.input.model_validate(arguments)
        signature = fingerprint({"tool": name, "args": args.to_dict(), "role": role})
        key = "tool:" + action_id
        old = self.store.action(key)
        if old:
            if old["signature"] != signature:
                raise PolicyError("idempotency_conflict", "action_id reutilizado con otra entrada")
            # Revalidar ACL incluso al devolver un resultado almacenado.
            if name == "read_artifact":
                self._artifact(args.artifact_id, role)
            return ToolResult.model_validate(old["result"])
        phase = self.store.get("phase", "search")
        if phase not in spec.phases or (self.store.get("search_closed", False) and name not in ("close_search", "read_artifact")):
            raise PolicyError("phase_denied", "Herramienta no permitida en esta fase")
        # Toda precondición se verifica antes de reservar o ejecutar servicios.
        if name == "discard_candidate":
            self._candidate(args.expected)
        if name == "diagnose_trial" and not any(r["attempt_id"] == args.attempt_id for r in self.store.records()):
            raise PolicyError("not_found", "Intento desconocido")
        if name == "revise_candidate":
            self._candidate(args.expected)
            cand = self._validate_candidate(args.candidate)
            if cand.id != args.expected.candidate_id or cand.version != args.expected.candidate_version + 1:
                raise PolicyError("stale_version", "La revisión debe crear la siguiente versión del mismo ID")
        elif name == "train_block":
            for ref in args.candidates:
                if ref.candidate_id in self.store.get("discarded_candidates", []):
                    raise PolicyError("invalid_input", "Candidata descartada")
                self._validate_candidate(self._candidate(ref))
        elif name == "read_artifact":
            self._artifact(args.artifact_id, role)
        elif name == "evaluate_validation":
            records = [r for r in self.store.records() if r["attempt_id"] == args.attempt_id]
            if not records or records[0]["status"] != "completed":
                raise PolicyError("not_found", "No existe validación completada para este intento")
        elif name == "propose_candidates" and self.store.get("tool_candidates", {}):
            raise PolicyError("invalid_input", "Las candidatas existentes deben revisarse con versión esperada")
        # Cerrar siempre es posible, incluso después de agotar decisiones o tiempo.
        reservation = None if name == "close_search" else self.budget.reserve(
            decisions=1, search=phase in ("profile", "search"))
        try:
            data = self._execute(name, args, role, action_id)
            output = spec.output.model_validate(data)
            result = ToolResult(result_id=uuid4().hex, tool=name, status="completed", data=output.to_dict())
            if not (name == "train_block" and any(
                    r["status"] in ("cancelled", "interrupted") for r in output.to_dict()["results"])):
                self.store.confirm_action(key, {"signature": signature, "result": result.to_dict()})
            return result
        finally:
            if reservation:
                self.budget.confirm(reservation)

    def _execute(self, name, args, role, action_id):
        if name == "query_evidence":
            return self.evidence.retrieve(self.kb, args.query, args.k, role=role,
                                          action_id=action_id, kinds=args.kinds)
        if name == "propose_candidates":
            from .kb import Doc
            docs = [Doc(**d) for d in self.store.get("tool_evidence", {}).values()
                    if d.get("scope") != "validation"]
            context = ConfigurationContext.build(self.profile, docs, self.cfg)
            result = ConfigurationSpecialist().run(context, self._specialist_llm("configuration"))
            result = ConfigurationResult.model_validate(result.to_dict()).validate_context(context)
            candidates = [self._validate_candidate(c) for c in result.candidates]
            from .catalog import full_space_size, space_size
            total, reduced = full_space_size(), space_size(candidates)
            proposal = {**result.to_dict(), "candidates": candidates,
                        "space": {"total_configs": total, "candidate_configs": reduced,
                                  "reduction_pct": round(100 * (1 - reduced / total), 1)},
                        "criteria": {"dataset_bucket": self.profile.size_bucket(),
                                     "resource_tier": self.cfg.resource_tier,
                                     "pretrained": self.cfg.pretrained, "retrieved_docs": []}}
            if self.store.get("tool_candidates", {}):
                raise PolicyError("invalid_input", "Las candidatas existentes deben revisarse con versión esperada")
            self.store.put("candidate_history", {f"{c.id}:v{c.version}": c.to_dict() for c in candidates})
            self.store.put("proposal", {**proposal, "candidates": [c.to_dict() for c in candidates]})
            self.store.put("tool_candidates", {c.id: c.to_dict() for c in candidates})
            self._log_specialist("configuration", action_id, context, result)
            return {"candidates": [c.to_dict() for c in candidates]}
        if name == "revise_candidate":
            candidates = self.store.get("tool_candidates", {})
            history = self.store.get("candidate_history", {})
            old = self._candidate(args.expected)
            history[f"{old.id}:v{old.version}"] = old.to_dict()
            history[f"{args.candidate.id}:v{args.candidate.version}"] = args.candidate.to_dict()
            self.store.put("candidate_history", history)
            candidates[args.candidate.id] = args.candidate.to_dict()
            self.store.put("tool_candidates", candidates)
            return {"candidates": [args.candidate.to_dict()]}
        if name == "train_block":
            self.engine.budget.admitted = True
            try:
                results = self.engine.run_block([self._candidate(r) for r in args.candidates],
                                                 args.max_attempts, action_key="tool-block:" + action_id)
            finally:
                self.engine.budget.admitted = False
            # No exponer rutas de checkpoint ni parámetros con acceso a archivos.
            safe = []
            for r in results:
                view = {"reference": r.reference.to_dict(), "status": r.status,
                        "metrics": r.metrics, "error": r.error.to_dict() if r.error else None}
                view["artifact_id"] = self._publish({"curves": r.curves, "validation": r.validation}, "validation")
                safe.append(view)
            return {"results": safe, "stop_reason": getattr(self.engine, "stop_reason", None)}
        if name == "discard_candidate":
            discarded = self.store.get("discarded_candidates", [])
            if args.expected.candidate_id not in discarded:
                discarded.append(args.expected.candidate_id)
            self.store.put("discarded_candidates", discarded)
            return {"candidate": args.expected.to_dict(), "discarded": True}
        if name == "diagnose_trial":
            context = self._diagnosis_context(args.attempt_id)
            if self.kb is not None:
                query = ("memoria GPU OOM reducir tamaño de lote recursos limitados"
                         if context.error and context.error.code == "out_of_memory"
                         else "diagnóstico curvas validación sobreajuste regularización dropout tasa aprendizaje")
                # Recuperación de un nivel bajo el mismo lock y presupuesto de decisiones.
                # Agotar RAG no impide producir el diagnóstico básico.
                try:
                    rag = self._call("query_evidence", {"query": query, "k": 2, "kinds": ["guide"]},
                                     "diagnosis", "diagnosis-rag-" + fingerprint(action_id))
                    self.store.put("specialist_rag:" + action_id, rag.to_dict())
                except Exception as exc:
                    self.store.put("specialist_rag:" + action_id, {"query": query, "role": "diagnosis",
                        "status": "failed", "error": type(exc).__name__, "message": str(exc)[:400]})
                context = self._diagnosis_context(args.attempt_id)
            result = DiagnosisSpecialist().run(context, self._specialist_llm("diagnosis"))
            diagnosis = DiagnosisResult.model_validate(result.to_dict()).validate_context(context).to_dict()
            self.validate_recommendation(diagnosis)
            diagnoses = self.store.get("diagnoses", {})
            diagnoses[args.attempt_id] = diagnosis
            self.store.put("diagnoses", diagnoses)
            self._log_specialist("diagnosis", action_id, context, result)
            return diagnosis
        if name == "evaluate_validation":
            # execute_trial ya evaluó exactamente el checkpoint seleccionado.
            result = next(r["result"] for r in self.store.records() if r["attempt_id"] == args.attempt_id)
            return {"metrics": result["metrics"], "validation": result["validation"]}
        if name == "read_artifact":
            return self.references.read(args.artifact_id, role, args.offset, args.length)
        if name == "close_search":
            self.budget.close_search(args.reason)
            return {"reason": args.reason, "phase": "evaluation"}
        raise AssertionError(name)
