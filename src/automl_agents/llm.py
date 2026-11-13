"""Cliente LLM opcional (Gemini, SDK google-genai). Sin clave API el sistema funciona en modo "reglas"."""
from __future__ import annotations

import json
import math
import os
import re

DEFAULT_LLM_TIMEOUT_SECONDS = 240.0


class LLM:
    def __init__(self, model: str, enabled: bool = True, *,
                 timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS):
        self.configure_timeout(timeout_seconds)
        self.model_name = model.removeprefix("models/")  # acepta "models/gemini-..." o "gemini-..."
        self.client = None
        self._types = None
        self.last_usage = None
        self.context_budget = None
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if enabled and key:
            try:
                from google import genai
                from google.genai import types

                self.client = genai.Client(api_key=key, http_options=types.HttpOptions(
                    timeout=math.ceil(self.timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1)))
                self._types = types
            except Exception:
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def configure_timeout(self, seconds: float):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("El timeout LLM debe ser positivo y finito")
        self.timeout_seconds = float(seconds)

    def configure_context_budget(self, budget):
        self.context_budget = budget

    def _generate(self, system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
        if not self.client:
            raise RuntimeError("LLM no disponible")
        self.last_usage = None
        if self.context_budget is not None:
            from .context import token_count
            tokens, _ = token_count(system + user, getattr(self, "tokenizer", None))
            max_tokens = min(max_tokens, self.context_budget.output_tokens)
            if (tokens > self.context_budget.input_tokens or
                    tokens + max_tokens > self.context_budget.window_tokens):
                raise ValueError("El prompt excede el presupuesto de contexto")
        t = self._types
        cfg = {"system_instruction": system, "max_output_tokens": max_tokens, "temperature": 0.2,
               "http_options": t.HttpOptions(timeout=math.ceil(self.timeout_seconds * 1000),
                                              retry_options=t.HttpRetryOptions(attempts=1))}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if "2.5-flash" in self.model_name:
            # Sin esto, los "tokens de pensamiento" consumen el presupuesto de salida
            # y la respuesta puede quedar vacía o cortada a mitad del JSON.
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=0)
        elif self.model_name.startswith("gemini-3"):
            # Gemini 3 comparte max_output_tokens entre razonamiento y respuesta.
            # El nivel predeterminado puede agotar la reserva antes de emitir JSON.
            # LOW es compatible con Flash y Pro; MINIMAL no lo es con todos.
            cfg["thinking_config"] = t.ThinkingConfig(thinking_level="LOW")
        resp = self.client.models.generate_content(
            model=self.model_name, contents=user, config=t.GenerateContentConfig(**cfg)
        )
        usage = getattr(resp, "usage_metadata", None)
        self.last_usage = usage.model_dump(mode="json") if usage is not None else None
        cands = getattr(resp, "candidates", None) or []
        reason = getattr(cands[0], "finish_reason", None) if cands else None
        if reason == "MAX_TOKENS":
            raise ValueError(
                f"Respuesta truncada de Gemini (MAX_TOKENS; límite={max_tokens}, "
                f"razonamiento={getattr(usage, 'thoughts_token_count', None)}, "
                f"respuesta={getattr(usage, 'candidates_token_count', None)}). "
                "Aumenta llm_context_budget.output_tokens (--llm-output-reserve) "
                "y comprueba que entrada y salida caben en window_tokens.")
        text = resp.text
        if not text:
            block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
            raise RuntimeError(f"Respuesta vacía de Gemini (finish_reason={reason}, block_reason={block})")
        return text

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        return self._generate(system, user, max_tokens)

    def complete_json(self, system: str, user: str, max_tokens: int = 2000) -> dict:
        raw = self._generate(system, user, max_tokens, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)  # respaldo si el modelo agrega texto alrededor
            if not m:
                raise ValueError("La respuesta del LLM no contiene JSON")
            return json.loads(m.group(0))
