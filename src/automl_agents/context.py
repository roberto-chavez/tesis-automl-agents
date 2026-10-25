"""Contextos deterministas y referencias inmutables a proyecciones autorizadas.

El conteo sin tokenizer es una estimación por bytes UTF-8, no consumo del proveedor.
Los datos críticos nunca se recortan: si no caben, se rechaza la llamada.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import re
from pydantic import model_validator

from .schemas import Contract, PositiveInt
from .state_store import fingerprint


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def token_count(text, tokenizer=None):
    if tokenizer is not None:
        return len(tokenizer.encode(text)), "tokenizer"
    return math.ceil(len(text.encode("utf-8")) / 3), "estimated_utf8_bytes"


class ModelContextBudget(Contract):
    input_tokens: PositiveInt = 16000
    output_tokens: PositiveInt = 2000
    window_tokens: PositiveInt = 24000

    @model_validator(mode="after")
    def fits_window(self):
        if self.input_tokens + self.output_tokens > self.window_tokens:
            raise ValueError("Entrada y reserva de salida exceden la ventana de contexto")
        return self


SCOPE_ROLES = {"evidence": ("orchestrator", "configuration", "diagnosis"),
               "configuration": ("orchestrator", "configuration"),
               "validation": ("orchestrator", "diagnosis"), "history": ("orchestrator",)}


class ReferenceStore:
    def __init__(self, store, max_read_chars=4000):
        self.store, self.max_read_chars = store, max_read_chars

    def publish(self, data, scope):
        # El ID incluye el alcance: publicar otro contexto nunca amplía un ACL.
        version = fingerprint(data)
        artifact_id = fingerprint({"scope": scope, "version": version})
        key = "safe_artifact:" + artifact_id
        if self.store.get(key) is None:
            self.store.put(key, {"data": data, "scope": scope,
                                 "roles": list(SCOPE_ROLES.get(scope, ())), "version": version})
        return artifact_id

    def get(self, artifact_id, role):
        from .policies import PolicyError
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", artifact_id):
            raise PolicyError("invalid_input", "Referencia inválida")
        artifact = self.store.get("safe_artifact:" + artifact_id)
        if artifact is None:
            raise PolicyError("not_found", "Artefacto desconocido")
        if role not in artifact["roles"] or role not in SCOPE_ROLES.get(artifact["scope"], ()):
            raise PolicyError("permission_denied", "Artefacto no accesible para este rol")
        return artifact

    def read(self, artifact_id, role, offset=0, length=2000):
        from .policies import PolicyError
        if type(offset) is not int or offset < 0 or type(length) is not int or length < 1:
            raise PolicyError("invalid_input", "Fragmento inválido")
        artifact = self.get(artifact_id, role)
        text = json.dumps(artifact["data"], ensure_ascii=False)
        end = min(len(text), offset + min(length, self.max_read_chars))
        return {"text": text[offset:end], "total_chars": len(text),
                "next_offset": end if end < len(text) else None,
                "version": artifact.get("version", fingerprint(artifact["data"]))}


def curve_summary(curves):
    return {name: {"count": len(values), "first": values[0], "last": values[-1],
                   "min": min(values), "max": max(values)}
            for name, values in curves.items() if values}


class ContextManager:
    def __init__(self, store, cfg):
        self.store, self.cfg = store, cfg
        if store.get("context_policy") is None:
            from .candidates import _allowed
            from .catalog import LR_GRID, DROPOUT_GRID, BATCH_GRID, AUGMENT_GRID
            store.put("context_policy", {
                "objective": {"use_case": cfg.use_case, "metric": cfg.metric_policy.to_dict()},
                "constraints": {"architectures": list(_allowed(cfg)[0]), "resource_tier": cfg.resource_tier,
                    "pretrained": cfg.pretrained, "img_size_override": cfg.img_size_override,
                    "max_candidates": cfg.max_candidates, "max_attempts_per_block": cfg.trials_per_round,
                    "grid": dict(lr_values=LR_GRID, dropout_values=DROPOUT_GRID,
                                 batch_sizes=BATCH_GRID, augment_values=AUGMENT_GRID),
                    "search_data_scope": "train_validation_only"},
                "default_model_budget": cfg.llm_context_budget.to_dict(),
                "model_budgets": {k.removeprefix("models/"): v.to_dict() for k, v in cfg.llm_model_budgets.items()},
                "max_read_chars": cfg.context_read_max_chars})
        self.policy = store.get("context_policy")
        self.refs = ReferenceStore(store, self.policy["max_read_chars"])

    def model_budget(self, model=None):
        model = (model or self.cfg.llm_model).removeprefix("models/")
        return ModelContextBudget.model_validate(self.policy["model_budgets"].get(
            model, self.policy["default_model_budget"]))

    def _reference(self, value, role):
        scope = {"configuration": "configuration", "diagnosis": "validation",
                 "orchestrator": "history"}[role]
        if self.store.get("test_opened", False):
            # El reporte final también usa el wrapper; nunca publicar sus datos como historial de búsqueda.
            scope = "test"
        return self.refs.publish(value, scope)

    def _compact(self, value, role, key="", terse=False):
        if key == "curves" and isinstance(value, dict):
            return {"summary": curve_summary(value), "artifact_id": self._reference(value, role)}
        if isinstance(value, dict):
            if "text" in value and "total_chars" in value and "next_offset" in value:
                # Una lectura ya viene acotada. Recortarla otra vez rompería su paginación.
                return value
            if key in ("diagnoses", "recent_results"):
                return {"count": len(value), "artifact_id": self._reference(value, role),
                        "recent": self._compact(list(value.values())[-2:], role, terse=terse)}
            result = {}
            for k, v in value.items():
                if k in ("text", "texto") and isinstance(v, str):
                    result[k] = v[:(160 if terse else 600)]
                    result["artifact_id"] = self._reference(v, role)
                    result["text_truncated"] = len(v) > len(result[k])
                elif k == "confusion_matrix":
                    result[k] = {"rows": len(v), "artifact_id": self._reference(v, role)}
                else:
                    result[k] = self._compact(v, role, k, terse)
            return result
        if isinstance(value, list):
            if key in ("records", "tasks", "history", "recent_results"):
                limit = 2 if terse else 5
                summary = {"count": len(value), "status_counts": dict(Counter(
                    str(v.get("status", "unknown")) for v in value if isinstance(v, dict))),
                    "recent": [self._compact(v, role, terse=terse) for v in value[-limit:]],
                    "artifact_id": self._reference(value, role)}
                if key == "records":
                    completed = [v for v in value if v.get("status") == "completed"]
                    summary["best_validation"] = max(completed, key=lambda r: r.get("val_acc", -1), default=None)
                    summary["candidates"] = {}
                    for record in value:
                        candidate_id = record.get("candidate_id", "unknown")
                        own = summary["candidates"].setdefault(candidate_id,
                            {"attempts": 0, "failures": 0, "best_val_acc": None})
                        own["attempts"] += 1
                        own["failures"] += record.get("status") == "failed"
                        own["latest_attempt_id"] = record.get("attempt_id")
                        own["latest_version"] = record.get("candidate_version")
                        if record.get("status") == "completed":
                            score = record.get("val_acc", 0.)
                            own["best_val_acc"] = max(score, own["best_val_acc"]) if own["best_val_acc"] is not None else score
                return summary
            return [self._compact(v, role, terse=terse) for v in value]
        if isinstance(value, str) and len(value) > (200 if terse else 1000):
            return {"summary": value[:(200 if terse else 1000)], "artifact_id": self._reference(value, role)}
        return value

    def prepare(self, system, user, role, *, model=None, output_tokens=None, tokenizer=None):
        from .policies import BudgetManager, PolicyError
        budget = self.model_budget(model)
        if output_tokens is not None and (type(output_tokens) is not int or output_tokens < 1):
            raise PolicyError("invalid_input", "El límite de salida debe ser un entero positivo")
        output = min(output_tokens or budget.output_tokens, budget.output_tokens)
        try:
            payload = json.loads(user)
        except (ValueError, TypeError):
            payload = {"request": user}
        # El host proporciona sólo proyecciones seguras; originales y resúmenes tienen el mismo ACL.
        reference = self._reference(payload, role)
        authoritative = {"objective": self.policy["objective"], "constraints": self.policy["constraints"],
            "budget": BudgetManager(self.store, self.cfg).snapshot(), "limits": self.store.get("limits"),
            "phase": self.store.get("phase", "search"),
            "search_closed": self.store.get("search_closed", False)}
        for terse in (False, True):
            compact = self._compact(payload, role, terse=terse)
            # Conservar las claves de la petición para mantener los contratos de los prompts.
            data = compact if isinstance(compact, dict) else {"request": compact}
            data = {**data, "authoritative": authoritative,
                    "plan": self._compact(self.store.get("plan", {"tasks": []}), role, terse=terse),
                    "original_artifact_id": reference,
                    "context_accounting": {"method": "tokenizer" if tokenizer else "estimated_utf8_bytes",
                                           "estimated": tokenizer is None,
                                           "input_limit": budget.input_tokens, "output_reserve": output}}
            prompt = dumps(data)
            tokens, method = token_count(system + prompt, tokenizer)
            if tokens <= budget.input_tokens and tokens + output <= budget.window_tokens:
                audit = {"role": role, "model": model or self.cfg.llm_model, "input_tokens": tokens,
                         "method": method, "estimated": tokenizer is None, "output_reserve": output,
                         "artifact_id": reference}
                self.store.put("last_context:" + role, audit)
                return prompt, output, audit
        raise PolicyError("context_limit", "Las restricciones y el contexto mínimo exceden el presupuesto de entrada")
