"""Agente generador de configuraciones candidatas (R1.2) y reporte de criterios (R1.3).

Modo "llm": el LLM razona sobre el perfil del dataset y los documentos recuperados (RAG).
Modo "reglas": heurística determinista basada en las mismas guías (sin API).
En ambos casos la salida se VALIDA contra el catálogo y la rejilla: el LLM no puede
introducir arquitecturas ni valores inexistentes (mitiga alucinaciones, sección 3.6.2).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .catalog import (AUGMENT_GRID, BATCH_GRID, CATALOG, DROPOUT_GRID, LR_GRID, TIER_ALLOWED,
                      full_space_size, space_size)
from .config import RunConfig
from .dataset import DatasetProfile
from .kb import Doc
from .llm import LLM


@dataclass
class Candidate:
    id: str
    architecture: str
    rationale: str
    sources: list[str]
    lr_values: list[float]
    dropout_values: list[float]
    batch_sizes: list[int]
    augment_values: list[bool]
    img_size: int

    def to_dict(self):
        return asdict(self)


def _subset(vals, grid, fallback):
    ok = [v for v in (vals or []) if v in grid]
    return ok or list(fallback)


def default_space(profile: DatasetProfile, cfg: RunConfig) -> dict:
    bucket = profile.size_bucket(1 - cfg.val_fraction - cfg.test_fraction)
    lr = [3e-4, 1e-3, 3e-3] if cfg.pretrained else [1e-4, 3e-4, 1e-3]
    dropout = {"small": [0.2, 0.4], "medium": [0.0, 0.2, 0.4], "large": [0.0, 0.2]}[bucket]
    batch = [16, 32] if cfg.resource_tier == "low" else [32, 64]
    augment = [True] if bucket == "small" else [False, True]
    return dict(lr_values=lr, dropout_values=dropout, batch_sizes=batch, augment_values=augment)


def _allowed(cfg: RunConfig) -> dict:
    names = cfg.allowed_architectures or list(CATALOG)
    tiers = TIER_ALLOWED[cfg.resource_tier]
    ok, discarded = {}, []
    for n in CATALOG:
        s = CATALOG[n]
        if n not in names:
            discarded.append({"architecture": n, "reason": "Excluida por el usuario (lista de arquitecturas permitidas)."})
        elif s.tier not in tiers:
            discarded.append({"architecture": n, "reason": (
                f"Excluida por recursos: ~{s.params_m}M de parámetros (nivel '{s.tier}') supera lo "
                f"razonable para el nivel de recursos '{cfg.resource_tier}'.")})
        else:
            ok[n] = s
    return ok, discarded


def _score(spec, bucket: str) -> float:
    s = spec.prior
    if bucket == "small":
        s += 0.10 if spec.tier == "light" else -0.10
    elif bucket == "large":
        s += 0.05 if spec.tier == "medium" else 0.0
    return s


def _img_size(spec, cfg):
    return cfg.img_size_override or spec.input_size


def _src_for(arch: str, docs: list[Doc]) -> list[str]:
    ids = [d.id for d in docs if d.id == f"arch_{arch.lower()}"]
    ids += [d.id for d in docs if d.kind == "guide"][:3]
    return ids


def propose_rules(profile, docs, cfg) -> tuple[list[Candidate], list[dict]]:
    ok, discarded = _allowed(cfg)
    bucket = profile.size_bucket(1 - cfg.val_fraction - cfg.test_fraction)
    ranked = sorted(ok.values(), key=lambda s: -_score(s, bucket))
    chosen, rest = ranked[: cfg.max_candidates], ranked[cfg.max_candidates:]
    space = default_space(profile, cfg)
    cands = []
    for i, s in enumerate(chosen, 1):
        why = (f"~{s.params_m}M parámetros, nivel de recursos '{s.tier}'; dataset {bucket} "
               f"({profile.n_images} imágenes, {profile.n_classes} clases) → prioridad {_score(s, bucket):.2f}.")
        cands.append(Candidate(f"c{i}", s.name, why, _src_for(s.name, docs), img_size=_img_size(s, cfg), **space))
    for s in rest:
        discarded.append({"architecture": s.name, "reason": (
            f"Válida, pero fuera del top-{cfg.max_candidates} por prioridad "
            f"({_score(s, bucket):.2f}) para un dataset {bucket}.")})
    return cands, discarded


def propose_llm(profile, docs, cfg, llm: LLM) -> tuple[list[Candidate], list[dict]]:
    ok, discarded = _allowed(cfg)
    system = ("Eres un ingeniero de ML que acota el espacio de búsqueda de modelos de clasificación de imágenes. "
              "Usa EXCLUSIVAMENTE la evidencia recuperada y el perfil del dataset. Justifica cada decisión.")
    user = json.dumps({
        "perfil_dataset": profile.to_dict(),
        "caso_de_uso": cfg.use_case,
        "nivel_recursos": cfg.resource_tier,
        "preentrenado": cfg.pretrained,
        "evidencia_recuperada": [{"id": d.id, "titulo": d.title, "texto": d.text} for d in docs],
        "arquitecturas_permitidas": {n: {"params_m": s.params_m} for n, s in ok.items()},
        "rejilla": {"lr": LR_GRID, "dropout": DROPOUT_GRID, "batch": BATCH_GRID, "augment": AUGMENT_GRID},
        "instrucciones": (
            f"Elige entre 1 y {cfg.max_candidates} arquitecturas permitidas. Devuelve JSON: "
            '{"candidatas":[{"arquitectura":str,"justificacion":str,"fuentes":[ids],"lr":[..],"dropout":[..],'
            '"batch":[..],"augment":[..]}],"descartadas":[{"arquitectura":str,"razon":str}]}. '
            "Los valores deben salir de la rejilla; elige 2-3 valores por hiperparámetro."),
    }, ensure_ascii=False)
    data = llm.complete_json(system, user)
    default = default_space(profile, cfg)
    doc_ids = {d.id for d in docs}
    cands = []
    for item in data.get("candidatas", []):
        a = item.get("arquitectura")
        if a not in ok or any(c.architecture == a for c in cands):
            continue
        cands.append(Candidate(
            f"c{len(cands)+1}", a, str(item.get("justificacion", ""))[:500],
            [s for s in item.get("fuentes", []) if s in doc_ids],
            _subset(item.get("lr"), LR_GRID, default["lr_values"]),
            _subset(item.get("dropout"), DROPOUT_GRID, default["dropout_values"]),
            _subset(item.get("batch"), BATCH_GRID, default["batch_sizes"]),
            _subset(item.get("augment"), AUGMENT_GRID, default["augment_values"]),
            _img_size(ok[a], cfg)))
        if len(cands) == cfg.max_candidates:
            break
    if not cands:
        raise ValueError("El LLM no propuso candidatas válidas")
    picked = {c.architecture for c in cands}
    reasons = {d.get("arquitectura"): d.get("razon") for d in data.get("descartadas", [])}
    for n in ok:
        if n not in picked:
            discarded.append({"architecture": n, "reason": reasons.get(n) or "No seleccionada por el agente."})
    return cands, discarded


def propose(profile, docs, cfg, llm: LLM | None) -> dict:
    mode, note = "reglas", ""
    if llm is not None and llm.available:
        try:
            cands, discarded = propose_llm(profile, docs, cfg, llm)
            mode = "llm"
        except Exception as e:  # cae a reglas: la ejecución nunca depende del LLM
            note = f"Fallo del LLM ({type(e).__name__}); se usó el generador por reglas."
            cands, discarded = propose_rules(profile, docs, cfg)
    else:
        cands, discarded = propose_rules(profile, docs, cfg)
    total, reduced = full_space_size(), space_size(cands)
    return {
        "mode": mode, "note": note,
        "candidates": cands, "discarded": discarded,
        "space": {"total_configs": total, "candidate_configs": reduced,
                  "reduction_pct": round(100 * (1 - reduced / total), 1)},
        "criteria": {
            "dataset_bucket": profile.size_bucket(1 - cfg.val_fraction - cfg.test_fraction),
            "resource_tier": cfg.resource_tier, "pretrained": cfg.pretrained,
            "retrieved_docs": [],
        },
    }
