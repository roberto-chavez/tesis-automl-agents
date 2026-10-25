"""Integridad de imágenes, perfiles de entrenamiento y particiones persistentes."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image
from pydantic import ValidationError

from .schemas import DatasetManifest, DomainError, ImageEntry, SplitManifest

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


class DatasetError(ValueError):
    """Error legible y serializable mediante el contrato común de errores."""

    def __init__(self, message: str, code: str = "invalid_dataset", recoverable: bool = False):
        super().__init__(message)
        self.error = DomainError(code=code, message=message, recoverable=recoverable)

    @property
    def code(self):
        return self.error.code

    def to_dict(self):
        return self.error.to_dict()


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
    scope: str = "all"

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def size_bucket(self, train_fraction: float = 0.7) -> str:
        n_train = self.n_images if self.scope == "train" else int(self.n_images * train_fraction)
        return "small" if n_train < 1000 else ("large" if n_train > 10000 else "medium")

    def to_dict(self) -> dict:
        return {
            "n_images": self.n_images, "n_classes": self.n_classes, "classes": self.classes,
            "counts": self.counts, "imbalance_ratio": round(self.imbalance_ratio, 2),
            "median_size": list(self.median_size), "min_side": self.min_side,
            "warnings": self.warnings, "scope": self.scope,
        }


def find_dataset_root(path: str | Path) -> Path:
    p = Path(path).resolve()
    if not p.is_dir():
        raise DatasetError(f"No existe el directorio del dataset: {p}", "dataset_not_found")
    while True:
        subs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))]
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]
        if len(subs) == 1 and not files and not any(
                f.suffix.lower() in IMG_EXT for f in subs[0].iterdir() if f.is_file()):
            p = subs[0]
            continue
        return p


def list_files(root: Path) -> dict[str, list[str]]:
    return {d.name: sorted(str(f) for f in d.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXT)
            for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith((".", "__"))}


def _inspect(root: Path, files: dict[str, list[str]]) -> DatasetManifest:
    if len(files) < 2:
        raise DatasetError("El dataset debe tener una carpeta por clase con imágenes dentro (al menos 2 clases).")
    entries, labels_by_hash = [], {}
    for label, (cls, paths) in enumerate(files.items()):
        if not paths:
            raise DatasetError(f"La clase '{cls}' no contiene imágenes.", "empty_class")
        for filename in paths:
            p = Path(filename)
            try:
                relative = p.relative_to(root).as_posix()
                p.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise DatasetError(f"Ruta fuera del dataset: {p}", "invalid_image_path") from exc
            try:
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                with Image.open(p) as im:
                    if im.format not in {"JPEG", "PNG", "BMP", "GIF"}:
                        raise ValueError(f"Formato {im.format} no compatible con el entrenamiento")
                    im.verify()
                with Image.open(p) as im:
                    im.convert("RGB").load()
                    width, height = im.size
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                raise DatasetError(f"Imagen corrupta o ilegible: {relative}. {exc}", "corrupt_image") from exc
            previous = labels_by_hash.setdefault(digest, cls)
            if previous != cls:
                raise DatasetError(f"Imagen duplicada con etiquetas en conflicto: {relative} ({previous}, {cls}).",
                                   "duplicate_label_conflict")
            entries.append(ImageEntry(path=relative, label=label, sha256=digest,
                                      size_bytes=p.stat().st_size, width=width, height=height))
    return DatasetManifest(classes=list(files), images=entries)


def _profile(root: Path, manifest: DatasetManifest, paths: set[str] | None = None) -> DatasetProfile:
    images = [e for e in manifest.images if paths is None or e.path in paths]
    counts = {c: sum(e.label == i for e in images) for i, c in enumerate(manifest.classes)}
    ratio = max(counts.values()) / min(counts.values())
    warnings = []
    if ratio > 1.5:
        warnings.append(f"Clases desbalanceadas (razón {ratio:.1f}:1): se usarán pesos de clase y F1 macro.")
    if len(images) < 200:
        warnings.append("Dataset muy pequeño (<200 imágenes): los resultados serán poco estables.")
    min_side = min(min(e.width, e.height) for e in images)
    if min_side < 32:
        warnings.append("Hay imágenes menores de 32 px por lado; podrían perder información al reescalarse.")
    duplicates = len(images) - len({e.sha256 for e in images})
    if duplicates:
        warnings.append(f"Hay {duplicates} copias exactas; cada grupo se conserva en una sola partición.")
    return DatasetProfile(str(root), manifest.classes, counts, len(images), ratio,
                          (int(statistics.median(e.width for e in images)),
                           int(statistics.median(e.height for e in images))),
                          min_side, warnings, "train" if paths is not None else "all")


def profile_dataset(path: str | Path, sample: int = 200, seed: int = 0) -> tuple[DatasetProfile, dict[str, list[str]]]:
    # sample y seed se conservan por compatibilidad; la integridad nunca se muestrea.
    root = find_dataset_root(path)
    files = list_files(root)
    manifest = _inspect(root, files)
    return _profile(root, manifest), files


def _partition(manifest: DatasetManifest, val_frac: float, test_frac: float, seed: int) -> dict[str, list[str]]:
    if not (math.isfinite(val_frac) and math.isfinite(test_frac) and
            0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise DatasetError("Las fracciones de validación y prueba deben ser positivas y sumar menos de 1.",
                           "invalid_split_fractions")
    rng = random.Random(seed)
    result = {name: [] for name in ("train", "val", "test")}
    for label, cls in enumerate(manifest.classes):
        groups = {}
        for entry in manifest.images:
            if entry.label == label:
                groups.setdefault(entry.sha256, []).append(entry.path)
        values = list(groups.values())
        rng.shuffle(values)
        n = len(values)
        nv, nt = math.floor(n * val_frac), math.floor(n * test_frac)
        if min(nv, nt, n - nv - nt) < 1:
            raise DatasetError(f"La clase '{cls}' tiene {n} imágenes únicas; no permite train/val/test "
                               f"con val={val_frac:g} y test={test_frac:g}. Aumente las imágenes o ajuste las fracciones.",
                               "insufficient_class_images")
        for name, selected in (("val", values[:nv]), ("test", values[nv:nv + nt]),
                               ("train", values[nv + nt:])):
            result[name].extend(p for group in selected for p in group)
    return result


def split_files(files: dict[str, list[str]], classes: list[str], val_frac: float, test_frac: float,
                seed: int) -> dict[str, tuple[list[str], list[int]]]:
    # Compatibilidad para consumidores sin persistencia. También valida y agrupa duplicados.
    if list(files) != classes:
        raise DatasetError("Las clases no coinciden con las etiquetas del dataset.")
    if len(classes) < 2 or any(not paths for paths in files.values()):
        raise DatasetError("Se necesitan al menos 2 clases con imágenes para crear las particiones.")
    import os
    files = {c: [str(Path(p).resolve()) for p in paths] for c, paths in files.items()}
    root = Path(os.path.commonpath([str(Path(p).resolve()) for paths in files.values() for p in paths]))
    manifest = _inspect(root, files)
    partitions = _partition(manifest, val_frac, test_frac, seed)
    return _materialize(root, manifest, partitions)


def _fingerprint(manifest: DatasetManifest) -> str:
    return hashlib.sha256(json.dumps(manifest.to_dict(), sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_manifest(path: Path, value) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(value.to_dict(), f, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_splits(manifest: DatasetManifest, splits: SplitManifest) -> dict[str, list[str]]:
    entries = {e.path: e for e in manifest.images}
    partitions = {name: getattr(splits, name) for name in ("train", "val", "test")}
    seen, hashes = set(), {}
    for name, paths in partitions.items():
        labels = set()
        for p in paths:
            if p not in entries or p in seen:
                raise ValueError("Rutas desconocidas o repetidas en particiones")
            seen.add(p)
            entry = entries[p]
            labels.add(entry.label)
            if hashes.setdefault(entry.sha256, name) != name:
                raise ValueError("Duplicados exactos entre particiones")
        if labels != set(range(len(manifest.classes))):
            raise ValueError("Cada partición debe contener todas las clases")
    if seen != set(entries):
        raise ValueError("Las particiones no cubren el dataset")
    return partitions


def _materialize(root, manifest, partitions):
    labels = {e.path: e.label for e in manifest.images}
    return {name: ([str(root / p) for p in paths], [labels[p] for p in paths])
            for name, paths in partitions.items()}


def prepare_dataset(path: str | Path, run_dir: str | Path, val_frac: float, test_frac: float, seed: int):
    """Crea manifiestos o lee los existentes verificando integridad y compatibilidad.

    Las fracciones se aplican por clase a grupos de archivos idénticos, redondeando
    hacia abajo val/test; entrenamiento recibe el resto. Nunca se separa un grupo.
    Un par incompleto se rechaza para no cambiar una partición tras una interrupción.
    """
    root = find_dataset_root(path)
    files = list_files(root)
    manifest = _inspect(root, files)
    directory = Path(run_dir) / "data"
    dataset_path, split_path = directory / "dataset_manifest.json", directory / "split_manifest.json"
    if dataset_path.exists() or split_path.exists():
        try:
            saved = DatasetManifest.model_validate_json(dataset_path.read_text("utf-8"))
            splits = SplitManifest.model_validate_json(split_path.read_text("utf-8"))
            if saved != manifest:
                raise DatasetError("El dataset o sus clases cambiaron respecto al manifiesto de la ejecución.",
                                   "dataset_changed")
            if (splits.seed, splits.val_fraction, splits.test_fraction) != (seed, val_frac, test_frac):
                raise DatasetError("La semilla o las fracciones cambiaron respecto a las particiones guardadas.",
                                   "split_config_changed")
            if splits.dataset_sha256 != _fingerprint(manifest):
                raise ValueError("La huella del manifiesto no coincide")
            partitions = _validate_splits(manifest, splits)
        except (OSError, ValidationError, ValueError) as exc:
            if isinstance(exc, DatasetError):
                raise
            raise DatasetError(f"Manifiestos incompletos o inválidos: {exc}", "invalid_manifest") from exc
    else:
        partitions = _partition(manifest, val_frac, test_frac, seed)
        splits = SplitManifest(dataset_sha256=_fingerprint(manifest), seed=seed,
                               val_fraction=val_frac, test_fraction=test_frac, **partitions)
        directory.mkdir(parents=True, exist_ok=True)
        _write_manifest(dataset_path, manifest)
        _write_manifest(split_path, splits)
    profile = _profile(root, manifest, set(partitions["train"]))
    return profile, files, _materialize(root, manifest, partitions)
