"""Perfilado y partición del dataset etiquetado (carpetas por clase)."""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


class DatasetError(ValueError):
    """Error de formato de dataset con mensaje entendible para un usuario no técnico."""


@dataclass
class DatasetProfile:
    root: str
    classes: list[str]
    counts: dict[str, int]
    n_images: int
    imbalance_ratio: float
    median_size: tuple[int, int]
    min_side: int
    warnings: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def size_bucket(self, train_fraction: float = 0.7) -> str:
        n_train = int(self.n_images * train_fraction)
        return "small" if n_train < 1000 else ("large" if n_train > 10000 else "medium")

    def to_dict(self) -> dict:
        return {
            "n_images": self.n_images, "n_classes": self.n_classes, "classes": self.classes,
            "counts": self.counts, "imbalance_ratio": round(self.imbalance_ratio, 2),
            "median_size": list(self.median_size), "min_side": self.min_side,
            "warnings": self.warnings,
        }


def find_dataset_root(path: str | Path) -> Path:
    """Si el ZIP se extrajo con una única carpeta contenedora, desciende hasta las clases."""
    p = Path(path)
    while True:
        subs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))]
        files = [f for f in p.iterdir() if f.suffix.lower() in IMG_EXT]
        if len(subs) == 1 and not files and not any(f.suffix.lower() in IMG_EXT for f in subs[0].iterdir() if f.is_file()):
            p = subs[0]
            continue
        return p


def list_files(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in sorted(x for x in root.iterdir() if x.is_dir() and not x.name.startswith((".", "__"))):
        files = sorted(str(f) for f in d.rglob("*") if f.suffix.lower() in IMG_EXT)
        if files:
            out[d.name] = files
    return out


def profile_dataset(path: str | Path, sample: int = 200, seed: int = 0) -> tuple[DatasetProfile, dict[str, list[str]]]:
    root = find_dataset_root(path)
    files = list_files(root)
    if len(files) < 2:
        raise DatasetError(
            "El dataset debe tener una carpeta por clase con imágenes dentro (al menos 2 clases). "
            "Ejemplo: mi_dataset/defectuoso/*.jpg y mi_dataset/correcto/*.jpg."
        )
    counts = {c: len(v) for c, v in files.items()}
    tiny = [c for c, n in counts.items() if n < 6]
    if tiny:
        raise DatasetError(
            f"Las clases {tiny} tienen menos de 6 imágenes; se necesitan al menos 6 por clase "
            "para separar entrenamiento, validación y prueba."
        )
    rng = random.Random(seed)
    allf = [f for v in files.values() for f in v]
    sizes = []
    for f in rng.sample(allf, min(sample, len(allf))):
        try:
            with Image.open(f) as im:
                sizes.append(im.size)
        except Exception:
            continue
    if not sizes:
        raise DatasetError("No se pudo leer ninguna imagen; verifique que los archivos sean JPG/PNG válidos.")
    mw = int(statistics.median(s[0] for s in sizes))
    mh = int(statistics.median(s[1] for s in sizes))
    ratio = max(counts.values()) / min(counts.values())
    warnings = []
    if ratio > 1.5:
        warnings.append(f"Clases desbalanceadas (razón {ratio:.1f}:1): se usarán pesos de clase y F1 macro.")
    if len(allf) < 200:
        warnings.append("Dataset muy pequeño (<200 imágenes): los resultados serán poco estables.")
    if min(min(s) for s in sizes) < 32:
        warnings.append("Hay imágenes menores de 32 px por lado; podrían perder información al reescalarse.")
    prof = DatasetProfile(str(root), list(files), counts, len(allf), ratio, (mw, mh),
                          min(min(s) for s in sizes), warnings)
    return prof, files


def split_files(files: dict[str, list[str]], classes: list[str], val_frac: float, test_frac: float,
                seed: int) -> dict[str, tuple[list[str], list[int]]]:
    """Partición estratificada train/val/test → {split: (rutas, etiquetas)}."""
    from sklearn.model_selection import train_test_split

    paths, labels = [], []
    for i, c in enumerate(classes):
        paths += files[c]
        labels += [i] * len(files[c])
    p_tr, p_rest, y_tr, y_rest = train_test_split(
        paths, labels, test_size=val_frac + test_frac, stratify=labels, random_state=seed)
    rel = test_frac / (val_frac + test_frac)
    p_va, p_te, y_va, y_te = train_test_split(
        p_rest, y_rest, test_size=rel, stratify=y_rest, random_state=seed)
    return {"train": (p_tr, y_tr), "val": (p_va, y_va), "test": (p_te, y_te)}
