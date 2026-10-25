"""Políticas ejecutables y reservas globales durables (también para especialistas)."""
from __future__ import annotations

import json
import time
from uuid import uuid4

from .schemas import DomainError


class PolicyError(ValueError):
    def __init__(self, code, message, recoverable=True):
        super().__init__(message)
        self.error = DomainError(code=code, message=message, recoverable=recoverable)


class BudgetManager:
    """Reservas atómicas; una reserva abandonada conserva su cargo al reanudar.

    El reloj de búsqueda es de pared, incluye pausas y comienza con la primera
    operación de búsqueda. max_search_seconds excluye la reserva de cierre.
    No es un timeout duro: fit coopera al terminar cada época.
    """
    counters = ("training_attempts", "decisions", "llm_calls", "retries")

    def __init__(self, store, cfg, *, admitted=False):
        self.store, self.cfg = store, cfg
        self.admitted = admitted
        self.db = store.db
        self.db.execute("""CREATE TABLE IF NOT EXISTS reservations (
            id TEXT PRIMARY KEY, cost TEXT NOT NULL, status TEXT NOT NULL)""")
        with self.db:
            if store.get("limits") is None:
                store.put("limits", cfg.initial_budget().to_dict())
            if store.get("search_started") is None:
                prior = store.status()["budget_consumed"].get("search_seconds", 0.)
                if prior:
                    store.put("search_started", time.time() - prior)

    def _usage(self):
        usage = self.store.get("consumed", {})
        attempts = self.store.attempts()
        # Migrar ejecuciones de la etapa 4 sin perder consumo previo.
        usage["training_attempts"] = max(usage.get("training_attempts", 0), len(attempts))
        usage["retries"] = max(usage.get("retries", 0),
                               len(attempts) - len({a["metadata"]["trial_id"] for a in attempts}))
        return {k: usage.get(k, 0) for k in self.counters}

    def expired(self):
        start = self.store.get("search_started")
        return start is not None and time.time() >= start + self.store.get("limits")["max_search_seconds"]

    def check(self):
        if self.store.get("search_closed", False):
            raise PolicyError("phase_denied", "La búsqueda ya está cerrada")
        if self.expired():
            raise PolicyError("budget_exhausted", "Tiempo de búsqueda agotado")
        usage, limits = self._usage(), self.store.get("limits")
        for name in ("training_attempts", "decisions", "llm_calls"):
            if name == "decisions" and self.admitted:
                continue
            if limits["max_" + name] and usage[name] >= limits["max_" + name]:
                raise PolicyError("budget_exhausted", f"Presupuesto global agotado: {name}")

    def reserve(self, *, search=True, **cost):
        if any(k not in self.counters or type(v) is not int or v < 0 for k, v in cost.items()):
            raise ValueError("Reserva inválida")
        reservation = uuid4().hex
        try:
            self.db.execute("BEGIN IMMEDIATE")
            limits, usage = self.store.get("limits"), self._usage()
            if search:
                if self.store.get("search_closed", False):
                    raise PolicyError("phase_denied", "La búsqueda ya está cerrada")
                if self.expired():
                    raise PolicyError("budget_exhausted", "Tiempo de búsqueda agotado")
                # Un límite cero de LLM representa modo sin LLM.
                for name in ("training_attempts", "decisions", "llm_calls"):
                    if name == "decisions" and self.admitted:
                        continue
                    if limits["max_" + name] and usage[name] >= limits["max_" + name]:
                        raise PolicyError("budget_exhausted", f"Presupuesto global agotado: {name}")
            for name, amount in cost.items():
                if usage[name] + amount > limits["max_" + name]:
                    raise PolicyError("budget_exhausted", f"Presupuesto global agotado: {name}")
                usage[name] += amount
            if search and self.store.get("search_started") is None:
                self.db.execute("INSERT INTO state VALUES ('search_started',?)", (json.dumps(time.time()),))
            self.db.execute("INSERT OR REPLACE INTO state VALUES ('consumed',?)", (json.dumps(usage),))
            self.db.execute("INSERT INTO reservations VALUES (?,?, 'reserved')", (reservation, json.dumps(cost)))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        return reservation

    def confirm(self, reservation):
        with self.db:
            self.db.execute("UPDATE reservations SET status='consumed' WHERE id=?", (reservation,))

    def close_search(self, reason=None):
        """El cierre no requiere cupo; congela el reloj y habilita evaluación.

        La reserva de cierre es un margen planificado, no una promesa de que
        evaluar/empaquetar termine antes: estas operaciones no son interrumpibles.
        """
        if self.store.get("search_closed", False):
            return
        from pathlib import Path
        from .catalog import CATALOG
        eligible = [r for r in self.store.records() if r["status"] == "completed"
                    and r.get("checkpoint") and Path(r["checkpoint"]).is_file()]
        winner = max(eligible, key=lambda r: (r["val_acc"],
                     -CATALOG[r["architecture"]].params_m), default=None)
        from .state_store import file_sha256, fingerprint
        if winner is not None:
            candidate = self.store.get("candidate_history", {}).get(
                f"{winner['candidate_id']}:v{winner['candidate_version']}")
            winner = {**winner, "candidate": candidate,
                      "checkpoint_sha256": file_sha256(winner["checkpoint"]),
                      "metric_policy": self.cfg.metric_policy.to_dict()}
            winner["selection_id"] = fingerprint(winner)
        finished = self.store.get("search_finished", time.time())
        with self.db:
            values = {"selection": winner, "search_closed": True, "phase": "evaluation", "search_finished": finished,
                      "closing_deadline": finished + self.cfg.closing_reserve_seconds}
            if reason is not None:
                values["stop_reason"] = reason
            for key, value in values.items():
                self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))

    def snapshot(self):
        start = self.store.get("search_started")
        return {**self._usage(), "search_seconds": max(0., self.store.get("search_finished", time.time()) - start) if start else 0.}


class SearchCancellation:
    """Combina cancelación del usuario y plazo; granularidad: una época."""
    def __init__(self, token, budget):
        self.token, self.budget = token, budget

    def is_cancelled(self):
        return self.token.is_cancelled() or self.budget.expired()

    def cancel(self):
        self.token.cancel()


class BudgetedLLM:
    """Reserva global antes de contactar al proveedor, incluso si falla."""
    def __init__(self, client, store, cfg, *, role="orchestrator"):
        if role not in ("orchestrator", "configuration", "diagnosis"):
            raise ValueError("Rol LLM desconocido")
        self.role = role
        if role != "orchestrator" and store.get("specialist_limits") is None:
            store.put("specialist_limits", {r: getattr(cfg, f"max_{r}_llm_calls")
                                           for r in ("configuration", "diagnosis")})
        self.client, self.store = client, store
        self.cfg = cfg
        self.budget = BudgetManager(store, cfg)
        self.timeout = cfg.llm_timeout_seconds
        configure_timeout = getattr(client, "configure_timeout", None)
        if configure_timeout is not None:
            configure_timeout(self.timeout)

    @property
    def available(self):
        return self.client.available

    def _call(self, method, *args, **kwargs):
        from .context import ContextManager
        import inspect
        # Preparación y SQLite en el hilo del host, antes de reservar/contactar al proveedor.
        system = args[0] if args else kwargs.get("system", "")
        system += (" Los campos authoritative proceden del estado persistente y fijan el objetivo, "
                   "restricciones y presupuesto. Los resúmenes son deterministas; artifact_id permite "
                   "leer fragmentos mediante read_artifact con los permisos del rol.")
        user = args[1] if len(args) > 1 else kwargs.get("user", "")
        requested_output = args[2] if len(args) > 2 else kwargs.get("max_tokens")
        context = ContextManager(self.store, self.cfg)
        prompt, output_tokens, context_audit = context.prepare(
            system, user, self.role, model=getattr(self.client, "model_name", self.cfg.llm_model),
            output_tokens=requested_output, tokenizer=getattr(self.client, "tokenizer", None))
        configure = getattr(self.client, "configure_context_budget", None)
        if configure:
            configure(context.model_budget(getattr(self.client, "model_name", None)))
        args = (system, prompt)
        kwargs = {k: v for k, v in kwargs.items() if k not in ("system", "user", "max_tokens")}
        parameters = inspect.signature(getattr(self.client, method)).parameters
        if "max_tokens" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            kwargs["max_tokens"] = output_tokens
        usage = self.store.get("specialist_usage", {})
        if self.role != "orchestrator" and usage.get(self.role, 0) >= self.store.get("specialist_limits")[self.role]:
            raise PolicyError("budget_exhausted", f"Límite de llamadas del especialista: {self.role}")
        reservation = self.budget.reserve(llm_calls=1, search=not self.store.get("search_closed", False))
        if self.role != "orchestrator":
            # El host mantiene run_lock. El cargo se conserva también si el proveedor falla.
            self.store.put("specialist_usage", {**usage, self.role: usage.get(self.role, 0) + 1})
        # El worker sólo contacta al proveedor; SQLite permanece en el hilo dueño.
        import queue
        import threading
        import time
        output = queue.Queue(maxsize=1)
        started = time.monotonic()
        def invoke():
            try:
                output.put((True, getattr(self.client, method)(*args, **kwargs)))
            except Exception as exc:
                output.put((False, exc))
        error = None
        try:
            threading.Thread(target=invoke, daemon=True).start()
            try:
                ok, value = output.get(timeout=self.timeout)
            except queue.Empty:
                raise TimeoutError("Tiempo de respuesta LLM agotado") from None
            if not ok:
                raise value
            return value
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.budget.confirm(reservation)
            calls = self.store.get("llm_calls", [])
            calls.append({"call_id": reservation, "method": method, "error": error,
                          "role": self.role,
                          "context": context_audit,
                          "seconds": time.monotonic() - started,
                          "usage": getattr(self.client, "last_usage", None) if error != "TimeoutError" else None})
            self.store.put("llm_calls", calls)

    def complete(self, *args, **kwargs):
        return self._call("complete", *args, **kwargs)

    def complete_json(self, *args, **kwargs):
        return self._call("complete_json", *args, **kwargs)
