"""Heurística de riesgo/datos personales sobre la descripción del caso de uso (secciones 2.2 y 2.3.3).

Es una señal temprana para el usuario, NO una determinación legal.
"""
from __future__ import annotations

import re
import unicodedata

_HIGH = [
    "biometr", "rostro", "cara", "facial", "huella", "candidato", "postulante", "selecci[oó]n de personal",
    "contrataci", "cr[eé]dito", "salud", "m[eé]dic", "diagn[oó]st", "paciente", "vigilancia",
    "menores", "ni[nñ]os", "seguridad de(l)? producto", "seguridad vial",
]
_PERSONAL = ["rostro", "persona", "trabajador", "operario", "cliente", "empleado", "empleados", "gente",
             "peatón", "face", "people", "worker", "customer"]


def _n(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))


def assess_use_case(text: str) -> dict:
    t = _n(text or "")
    high = [k for k in _HIGH if re.search(_n(k), t)]
    personal = [k for k in _PERSONAL if re.search(rf"\b{_n(k)}", t)]
    notes = []
    if personal:
        notes.append(
            "El caso de uso menciona personas; si las imágenes contienen rostros u otros datos personales, "
            "el dataset puede estar sujeto a la Ley N° 29733 (anonimice o recorte las imágenes antes de cargarlas)."
        )
    if high:
        notes.append(
            "El caso de uso podría clasificar como de alto riesgo bajo el reglamento de la Ley N° 31814 "
            "(DS N° 115-2025-PCM): mantenga supervisión humana y documente la decisión."
        )
    return {
        "risk_level": "posible alto riesgo (revisar)" if high else "riesgo bajo aparente",
        "personal_data_flag": bool(personal),
        "matched_terms": sorted(set(high + personal)),
        "notes": notes,
        "disclaimer": "Señal heurística automática; no constituye asesoría legal.",
    }
