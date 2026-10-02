"""Entrenamiento de un modelo candidato con TensorFlow/Keras (R2.1)."""
from __future__ import annotations

import gc
import importlib
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from .catalog import ArchSpec  # noqa: E402

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
    return importlib.import_module(f"keras.applications.{spec.module}").preprocess_input


def load_base_dataset(paths, labels, img_size: int, spec: ArchSpec):
    """Dataset sin lotes: decodifica, redimensiona y aplica el preprocesamiento del backbone."""
    pre = preprocess_fn(spec)

    def _load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.cast(tf.image.resize(img, (img_size, img_size)), tf.float32)
        return pre(img), y

    ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), np.asarray(labels, dtype="int32")))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if len(paths) * img_size * img_size * 3 * 4 < CACHE_LIMIT_BYTES:
        ds = ds.cache()
    return ds


def batched(ds, batch: int, n: int, shuffle: bool, seed: int):
    if shuffle:
        ds = ds.shuffle(min(n, 4096), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(spec: ArchSpec, n_classes: int, img_size: int, dropout: float, augment: bool,
                lr: float, pretrained: bool):
    base_cls = getattr(keras.applications, spec.name)
    base = base_cls(include_top=False, weights="imagenet" if pretrained else None,
                    input_shape=(img_size, img_size, 3), pooling="avg")
    base.trainable = not pretrained          # transferencia: backbone congelado + cabeza nueva
    inp = keras.Input((img_size, img_size, 3))
    x = inp
    if augment:
        x = keras.Sequential([keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.05),
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


def train_trial(spec, params, n_classes, img_size, ds_train, ds_val, max_epochs, pretrained,
                class_weight, trial=None, seed=0):
    keras.utils.set_random_seed(seed)
    model = build_model(spec, n_classes, img_size, params["dropout"], params["augment"], params["lr"],
                        pretrained)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)]
    prune = PruneCallback(trial) if trial is not None else None
    if prune:
        cbs.append(prune)
    hist = model.fit(ds_train, validation_data=ds_val, epochs=max_epochs, callbacks=cbs,
                     class_weight=class_weight, verbose=0)
    val_accs = hist.history.get("val_accuracy", [0.0])
    return model, {
        "val_acc": float(max(val_accs)),
        "epochs": len(val_accs),
        "pruned": bool(prune and prune.pruned),
    }


def release(model=None):
    del model
    keras.backend.clear_session()
    gc.collect()
