"""Agente de evaluación (R2.2): métricas estandarizadas sobre el conjunto de prueba."""
from __future__ import annotations

import time
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_validation(model, dataset, y_true, class_names: list[str]) -> dict:
    """Evalúa el checkpoint restaurado usando exclusivamente validación."""
    y_pred = model.predict(dataset, verbose=0).argmax(1)
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": list(class_names),
    }


def evaluate_model(model_path: Path, ds_test, y_true, class_names: list[str]) -> dict:
    try:
        return _evaluate_model(model_path, ds_test, y_true, class_names)
    finally:
        keras.backend.clear_session()


def _evaluate_model(model_path, ds_test, y_true, class_names):
    model = keras.saving.load_model(model_path, compile=False)
    t0 = time.perf_counter()
    probs = model.predict(ds_test, verbose=0)
    dt = time.perf_counter() - t0
    y_pred = probs.argmax(1)
    y_true = np.asarray(y_true)
    if probs.shape != (len(y_true), len(class_names)) or not len(y_true) or not np.isfinite(probs).all():
        raise ValueError("Predicciones incompatibles con las clases o el conjunto de prueba")
    labels = list(range(len(class_names)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                output_dict=True, zero_division=0)
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "classes": list(class_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {c: {"precision": round(rep[c]["precision"], 3), "recall": round(rep[c]["recall"], 3),
                          "f1": round(rep[c]["f1-score"], 3), "support": int(rep[c]["support"])}
                      for c in class_names},
        "latency_ms_per_image": round(1000 * dt / len(y_true), 2),
        "model_size_mb": round(Path(model_path).stat().st_size / 1e6, 2),
        "n_test": int(len(y_true)),
    }
    return out


def evaluate_frozen_selection(store, engine):
    """Evalúa sólo la selección durable y recupera una publicación tras una caída.

    Los resultados finales viven fuera de los registros/contextos de búsqueda.
    El host debe mantener run_lock durante esta operación.
    """
    import json
    from . import trainer
    from .events import write_json
    from .schemas import Candidate
    from .state_store import file_sha256

    best = store.get("selection")
    if not store.get("search_closed") or best is None:
        raise RuntimeError("La evaluación requiere una selección congelada")
    if file_sha256(best["checkpoint"]) != best["checkpoint_sha256"]:
        raise ValueError("El checkpoint congelado cambió después de la selección")
    candidate = best.get("candidate") or store.get("candidate_history", {}).get(
        f"{best['candidate_id']}:v{best['candidate_version']}")
    if not candidate:
        raise ValueError("Falta la versión persistida de la candidata seleccionada")
    cand = Candidate.from_dict(candidate)
    if (cand.id, cand.version, cand.architecture) != (
            best["candidate_id"], best["candidate_version"], best["architecture"]):
        raise ValueError("La versión persistida no corresponde a la selección")
    winner = {key: best[key] for key in ("candidate_id", "candidate_version", "study_id", "trial_id",
              "attempt_id", "architecture", "params", "val_acc", "checkpoint", "checkpoint_sha256",
              "selection_id", "metric_policy")}
    winner["img_size"] = cand.img_size
    key = "final_evaluation:" + best["selection_id"]
    path = store.run_dir / "final_evaluation.json"
    saved = store.action(key)
    if saved is None and path.exists():
        saved = json.loads(path.read_text("utf-8"))
    if saved is not None:
        if saved["winner"] != winner or saved["metrics"]["classes"] != engine.profile.classes:
            raise ValueError("La evaluación publicada no corresponde a la selección congelada")
    else:
        store.put("test_opened", True)  # Durable ANTES de acceder a test.
        ds = trainer.batched(engine.base_ds("test", cand), 64, 0, False, 0)
        metrics = evaluate_model(Path(best["checkpoint"]), ds, engine.splits["test"][1], engine.profile.classes)
        saved = {"winner": winner, "metrics": metrics}
        write_json(path, saved)
    store.put("test_opened", True)
    store.confirm_action(key, saved)
    store.put("final_evaluation", saved)
    return saved
