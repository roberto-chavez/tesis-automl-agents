"""Entrenamiento de un modelo candidato con TensorFlow/Keras (R2.1)."""
from __future__ import annotations

import gc
import importlib
import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from .catalog import ArchSpec  # noqa: E402
from .preprocessing import decode_and_preprocess  # noqa: E402

CACHE_LIMIT_BYTES = 1.5e9


def configure_tf() -> dict:
    """Crecimiento dinámico de memoria de GPU (importante en GPU de 4 GB como la GTX 1650)."""
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    return {"gpus": [g.name for g in gpus], "tf": tf.__version__, "keras": keras.__version__}


def preprocess_fn(spec: ArchSpec):
    if spec.module == "small_cnn":
        return lambda x: x / 255.0
    return importlib.import_module(f"keras.applications.{spec.module}").preprocess_input


def load_base_dataset(paths, labels, img_size: int, spec: ArchSpec, *, cache=True):
    """Dataset sin lotes: decodifica, redimensiona y aplica el preprocesamiento del backbone."""
    def _load(p, y):
        return decode_and_preprocess(tf.io.read_file(p), img_size, spec.module), y

    ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), np.asarray(labels, dtype="int32")))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache and len(paths) * img_size * img_size * 3 * 4 < CACHE_LIMIT_BYTES:
        ds = ds.cache()
    return ds


def batched(ds, batch: int, n: int, shuffle: bool, seed: int):
    if shuffle:
        ds = ds.shuffle(min(n, 4096), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(spec: ArchSpec, n_classes: int, img_size: int, dropout: float, augment: bool,
                lr: float, pretrained: bool):
    if spec.name == "SmallCNN":
        # Train all layers: ImageNet features are unnecessary for tiny images.
        # Standard Keras layers keep the exported model independently loadable.
        inp = keras.Input((img_size, img_size, 3))
        x = inp
        if augment:
            x = keras.Sequential([
                keras.layers.RandomRotation(0.04, fill_mode="constant"),
                keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant"),
                keras.layers.RandomZoom(0.1, fill_mode="constant"),
            ], name="spatial_augmentation")(x)
        for filters in (32, 64):
            x = keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
            x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)
        out = keras.layers.Dense(n_classes, activation="softmax")(x)
        model = keras.Model(inp, out, name="SmallCNN")
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model
    base_cls = getattr(keras.applications, spec.name)
    base = base_cls(include_top=False, weights="imagenet" if pretrained else None,
                    input_shape=(img_size, img_size, 3), pooling="avg")
    base.trainable = not pretrained          # transferencia: backbone congelado + cabeza nueva
    inp = keras.Input((img_size, img_size, 3))
    x = inp
    if augment:
        x = keras.Sequential([keras.layers.RandomRotation(0.05),
                              keras.layers.RandomZoom(0.1)])(x)
    x = base(x, training=False) if pretrained else base(x)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


class PruneCallback(keras.callbacks.Callback):
    """Informa val_accuracy a Optuna y detiene el entrenamiento si el pruner (Hyperband) lo indica."""

    def __init__(self, trial):
        super().__init__()
        self.trial, self.pruned = trial, False

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(float((logs or {}).get("val_accuracy", 0.0)), step=epoch)
        if self.trial.should_prune():
            self.pruned = True
            self.model.stop_training = True


class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, progress, cancellation=None, on_epoch=None):
        super().__init__()
        self.progress, self.cancellation, self.on_epoch = progress, cancellation, on_epoch

    def on_epoch_end(self, epoch, logs=None):
        if any(not math.isfinite(float(value)) for key, value in (logs or {}).items()
               if key in ("loss", "accuracy", "val_loss", "val_accuracy")):
            raise FloatingPointError("El entrenamiento produjo métricas no finitas")
        for key in ("loss", "accuracy", "val_loss", "val_accuracy"):
            if key in (logs or {}):
                self.progress.setdefault("curves", {}).setdefault(key, []).append(float(logs[key]))
        self.progress["epochs"] = epoch + 1
        if self.cancellation and self.cancellation.is_cancelled():
            self.progress["cancelled"] = True
            self.model.stop_training = True
        if self.on_epoch:
            self.on_epoch(self.progress)


def train_trial(spec, params, n_classes, img_size, ds_train, ds_val, max_epochs, pretrained,
                class_weight, trial=None, seed=0, cancellation=None, progress=None, on_epoch=None):
    progress = progress if progress is not None else {}
    keras.utils.set_random_seed(seed)
    model = build_model(spec, n_classes, img_size, params["dropout"], params["augment"], params["lr"],
                        pretrained)
    # Small validation sets have coarse, noisy accuracy steps; allow recovery
    # between plateaus while retaining the same best-checkpoint criterion.
    patience = 5 if spec.name == "SmallCNN" else 2
    early = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience,
                                          restore_best_weights=True)
    cbs = [early]
    prune = PruneCallback(trial) if trial is not None else None
    if prune:
        cbs.append(prune)
    cbs.append(ProgressCallback(progress, cancellation, on_epoch))
    hist = model.fit(ds_train, validation_data=ds_val, epochs=max_epochs, callbacks=cbs,
                     class_weight=class_weight, verbose=0)
    val_accs = hist.history.get("val_accuracy", [0.0])
    cancelled = bool(progress.get("cancelled"))
    reason = ("cancelled" if cancelled else "pruned" if prune and prune.pruned else
              "early_stopping" if early.stopped_epoch > 0 else "max_epochs")
    return model, {
        "val_acc": float(max(val_accs)),
        "epochs": len(val_accs),
        "pruned": bool(prune and prune.pruned),
        "cancelled": cancelled,
        "curves": progress.get("curves", {}),
        "stop_reason": reason,
    }


def release(model=None):
    del model
    keras.backend.clear_session()
    gc.collect()
