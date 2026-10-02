"""Dataset sintético para pruebas de humo (sin descargas)."""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw


def make_demo_dataset(root: str | Path, per_class: int = 30, size: int = 64, seed: int = 0,
                      imbalance: bool = False) -> Path:
    rng = random.Random(seed)
    root = Path(root)
    for ci, name in enumerate(["circulo", "cuadrado", "rayas"]):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        n = per_class if not (imbalance and ci == 0) else per_class * 2
        for i in range(n):
            bg = tuple(rng.randint(180, 255) for _ in range(3))
            im = Image.new("RGB", (size, size), bg)
            dr = ImageDraw.Draw(im)
            col = tuple(rng.randint(0, 120) for _ in range(3))
            o = rng.randint(4, size // 4)
            if name == "circulo":
                dr.ellipse([o, o, size - o, size - o], fill=col)
            elif name == "cuadrado":
                dr.rectangle([o, o, size - o, size - o], fill=col)
            else:
                for y in range(0, size, 8):
                    dr.rectangle([0, y, size, y + 3], fill=col)
            im.save(d / f"{i:03d}.png")
    return root
