"""Mecanismo de iteración automática (R2.3): decide continuar, ajustar o descartar candidatos."""
from __future__ import annotations

import dataclasses

from .catalog import DROPOUT_GRID, LR_GRID
from .config import RunConfig


def _best(records, cid):
    done = [r["val_acc"] for r in records if r["candidate_id"] == cid and not r["pruned"]]
    if done:
        return max(done)
    partial = [r["val_acc"] for r in records if r["candidate_id"] == cid]
    return max(partial) if partial else 0.0


def _best_params(records, cid):
    rs = [r for r in records if r["candidate_id"] == cid and not r["pruned"]]
    return max(rs, key=lambda r: r["val_acc"])["params"] if rs else None


def _neighbors(values, grid, center):
    i = grid.index(center)
    keep = [v for v in values if abs(grid.index(v) - i) <= 1]
    return keep or list(values)


def decide(round_idx: int, cands, records, cfg: RunConfig):
    """Devuelve (candidatos para la siguiente ronda, decisiones registradas, ¿detener?)."""
    scores = {c.id: _best(records, c.id) for c in cands}
    leader_id = max(scores, key=scores.get)
    leader = scores[leader_id]
    last = round_idx >= cfg.rounds
    decisions, nxt = [], []
    for c in cands:
        s = scores[c.id]
        if leader >= cfg.target_accuracy:
            decisions.append({"round": round_idx, "candidate_id": c.id, "action": "finalizar",
                              "reason": f"El líder alcanzó la exactitud objetivo ({leader:.3f} ≥ {cfg.target_accuracy})."})
        elif last:
            decisions.append({"round": round_idx, "candidate_id": c.id, "action": "finalizar",
                              "reason": "Última ronda del presupuesto de búsqueda."})
        elif c.id != leader_id and s < leader - cfg.discard_margin:
            decisions.append({"round": round_idx, "candidate_id": c.id, "action": "descartar",
                              "reason": f"Mejor val_acc {s:.3f} < líder {leader:.3f} − margen {cfg.discard_margin}."})
        else:
            bp = _best_params(records, c.id)
            if bp:
                lr = _neighbors(c.lr_values, LR_GRID, bp["lr"])
                dr = _neighbors(c.dropout_values, DROPOUT_GRID, bp["dropout"])
                if (lr, dr) != (c.lr_values, c.dropout_values):
                    c = dataclasses.replace(c, lr_values=lr, dropout_values=dr)
                    decisions.append({"round": round_idx, "candidate_id": c.id, "action": "ajustar",
                                      "reason": f"Se acota el espacio alrededor de lr={bp['lr']} y dropout={bp['dropout']}."})
                    nxt.append(c)
                    continue
            decisions.append({"round": round_idx, "candidate_id": c.id, "action": "continuar",
                              "reason": f"Val_acc {s:.3f} dentro del margen del líder ({leader:.3f})."})
            nxt.append(c)
    stop = last or leader >= cfg.target_accuracy or not nxt
    return nxt, decisions, stop
