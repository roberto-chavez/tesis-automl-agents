"""Ahorro frente a línea base (R4.1, R4.3).

Las líneas base son ESTIMACIONES basadas en el costo medio por época medido en la propia
ejecución (se declara así en el reporte); no se entrenan realmente.
"""
from __future__ import annotations

from .catalog import full_space_size


def compute_benchmark(records: list[dict], cfg, space: dict) -> dict:
    n = len(records)
    pruned = sum(r["pruned"] for r in records)
    epochs_used = sum(r["epochs"] for r in records)
    seconds = sum(r["resources"]["seconds"] for r in records)
    per_epoch = seconds / epochs_used if epochs_used else 0.0
    epochs_full_trials = n * cfg.max_epochs
    base_full_epochs = full_space_size() * cfg.max_epochs
    base_cand_epochs = space["candidate_configs"] * cfg.max_epochs

    def pct(x, ref):
        return round(100 * (1 - x / ref), 1) if ref else 0.0

    return {
        "trials_total": n, "trials_pruned": pruned,
        "pruned_pct": round(100 * pruned / n, 1) if n else 0.0,
        "epochs_used": epochs_used, "seconds_measured": round(seconds, 1),
        "seconds_per_epoch_avg": round(per_epoch, 2),
        "epochs_without_pruning_same_trials": epochs_full_trials,
        "saving_from_early_stopping_pct": pct(epochs_used, epochs_full_trials),
        "baseline_exhaustive_full_space": {
            "configs": full_space_size(), "epochs": base_full_epochs,
            "est_hours": round(base_full_epochs * per_epoch / 3600, 2),
            "reduction_epochs_pct": pct(epochs_used, base_full_epochs)},
        "baseline_exhaustive_candidate_space": {
            "configs": space["candidate_configs"], "epochs": base_cand_epochs,
            "est_hours": round(base_cand_epochs * per_epoch / 3600, 2),
            "reduction_epochs_pct": pct(epochs_used, base_cand_epochs)},
        "note": ("Líneas base estimadas: configuraciones x épocas máximas x costo medio por época medido. "
                 "Como el RAG excluye modelos pesados, el costo real de la línea base completa sería mayor."),
    }
