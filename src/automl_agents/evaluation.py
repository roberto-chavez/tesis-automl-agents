"""Agente de evaluación (R2.2): métricas estandarizadas sobre el conjunto de prueba."""
from __future__ import annotations

import time
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_model(model_path: Path, ds_test, y_true, class_names: list[str]) -> dict:
    model = keras.saving.load_model(model_path)
    t0 = time.perf_counter()
    probs = model.predict(ds_test, verbose=0)
    dt = time.perf_counter() - t0
    y_pred = probs.argmax(1)
    y_true = np.asarray(y_true)
    labels = list(range(len(class_names)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                output_dict=True, zero_division=0)
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {c: {"precision": round(rep[c]["precision"], 3), "recall": round(rep[c]["recall"], 3),
                          "f1": round(rep[c]["f1-score"], 3), "support": int(rep[c]["support"])}
                      for c in class_names},
        "latency_ms_per_image": round(1000 * dt / len(y_true), 2),
        "model_size_mb": round(Path(model_path).stat().st_size / 1e6, 2),
        "n_test": int(len(y_true)),
    }
    keras.backend.clear_session()
    return out
