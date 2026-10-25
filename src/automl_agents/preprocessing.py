"""Transformación compartida por entrenamiento y la API exportada.

Este archivo se copia sin cambios al artefacto; no depende del paquete AutoML.
"""
import importlib

import tensorflow as tf


PREPROCESSING = {
    "version": 1, "decoder": "tf.io.decode_image", "channels": 3,
    "expand_animations": False, "resize": "bilinear", "antialias": False,
    "dtype": "float32", "range_before_backbone": [0, 255],
}


def decode_and_preprocess(contents, img_size, preprocess_module):
    img = tf.io.decode_image(contents, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.cast(tf.image.resize(img, (img_size, img_size), method="bilinear",
                                  antialias=False), tf.float32)
    pre = importlib.import_module("keras.applications." + preprocess_module).preprocess_input
    return pre(img)
