"""Cliente LLM opcional (Gemini, SDK google-genai). Sin clave API el sistema funciona en modo "reglas"."""
from __future__ import annotations

import json
import os
import re


class LLM:
    def __init__(self, model: str, enabled: bool = True):
        self.model_name = model.removeprefix("models/")  # acepta "models/gemini-..." o "gemini-..."
        self.client = None
        self._types = None
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if enabled and key:
            try:
                from google import genai
                from google.genai import types

                self.client = genai.Client(api_key=key)
                self._types = types
            except Exception:
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def _generate(self, system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
        if not self.client:
            raise RuntimeError("LLM no disponible")
        t = self._types
        cfg = {"system_instruction": system, "max_output_tokens": max_tokens, "temperature": 0.2}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if "2.5-flash" in self.model_name:
            # Sin esto, los "tokens de pensamiento" consumen el presupuesto de salida
            # y la respuesta puede quedar vacía o cortada a mitad del JSON.
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=0)
        resp = self.client.models.generate_content(
            model=self.model_name, contents=user, config=t.GenerateContentConfig(**cfg)
        )
        text = resp.text
        if not text:
            cands = getattr(resp, "candidates", None) or []
            reason = getattr(cands[0], "finish_reason", None) if cands else None
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
