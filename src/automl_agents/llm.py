"""Cliente LLM opcional (Anthropic). Sin clave API el sistema funciona en modo "reglas"."""
from __future__ import annotations

import json
import os
import re


class LLM:
    def __init__(self, model: str, enabled: bool = True):
        self.model = model
        self.client = None
        key = os.getenv("ANTHROPIC_API_KEY")
        if enabled and key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=key)
            except Exception:
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        if not self.client:
            raise RuntimeError("LLM no disponible")
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

    def complete_json(self, system: str, user: str, max_tokens: int = 2000) -> dict:
        raw = self.complete(system + "\nResponde SOLO con JSON válido, sin texto adicional.", user, max_tokens)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("La respuesta del LLM no contiene JSON")
        return json.loads(m.group(0))
