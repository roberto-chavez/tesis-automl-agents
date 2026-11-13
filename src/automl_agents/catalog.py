"""Catálogo de arquitecturas (Keras Applications y CNN local) y rejilla discretizada.

El espacio se discretiza a propósito: así el "tamaño del espacio total" (R1.2) y el
reparto de trabajo de Optuna (R2.3) se miden sobre exactamente los mismos valores.
"""
from __future__ import annotations

from dataclasses import dataclass

# Rejilla completa de hiperparámetros considerada por el sistema.
LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
DROPOUT_GRID = (0.0, 0.2, 0.4)
BATCH_GRID = (16, 32, 64)
AUGMENT_GRID = (False, True)
GRID_SIZE = len(LR_GRID) * len(DROPOUT_GRID) * len(BATCH_GRID) * len(AUGMENT_GRID)  # 90


@dataclass(frozen=True)
class ArchSpec:
    name: str          # nombre en Keras Applications o SmallCNN local
    module: str        # submódulo de keras.applications o normalización small_cnn
    params_m: float    # parámetros aproximados (millones), backbone sin cabeza
    input_size: int    # resolución nativa
    tier: str          # light | medium | heavy (peso computacional relativo)
    prior: float       # prioridad a priori (0-1) usada por el generador basado en reglas


CATALOG: dict[str, ArchSpec] = {
    s.name: s
    for s in [
        ArchSpec("SmallCNN", "small_cnn", 0.42, 28, "light", 0.30),
        ArchSpec("MobileNetV2", "mobilenet_v2", 3.5, 224, "light", 0.80),
        ArchSpec("MobileNetV3Small", "mobilenet_v3", 2.5, 224, "light", 0.70),
        ArchSpec("EfficientNetB0", "efficientnet", 5.3, 224, "light", 0.85),
        ArchSpec("EfficientNetV2B0", "efficientnet_v2", 7.2, 224, "light", 0.90),
        ArchSpec("NASNetMobile", "nasnet", 5.3, 224, "light", 0.50),
        ArchSpec("DenseNet121", "densenet", 8.1, 224, "light", 0.70),
        ArchSpec("ResNet50V2", "resnet_v2", 25.6, 224, "medium", 0.75),
        ArchSpec("Xception", "xception", 22.9, 299, "medium", 0.70),
        ArchSpec("InceptionV3", "inception_v3", 23.9, 299, "medium", 0.65),
        ArchSpec("ConvNeXtTiny", "convnext", 28.6, 224, "medium", 0.80),
        ArchSpec("VGG16", "vgg16", 138.4, 224, "heavy", 0.20),
    ]
}

TIER_ALLOWED = {
    "low": {"light"},                      # CPU / GPU de ~4 GB (p. ej. GTX 1650)
    "medium": {"light", "medium"},         # GPU de 8-12 GB
    "high": {"light", "medium", "heavy"},  # GPU de gama alta o nube
}


def full_space_size() -> int:
    """Nº de configuraciones (arquitectura x hiperparámetros) del espacio completo."""
    return len(CATALOG) * GRID_SIZE


def space_size(candidates) -> int:
    """Nº de configuraciones cubiertas por un conjunto de candidatos con espacios acotados."""
    total = 0
    for c in candidates:
        total += (
            len(c.lr_values) * len(c.dropout_values) * len(c.batch_sizes) * len(c.augment_values)
        )
    return total
