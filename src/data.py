"""Manifest-based data access.

A manifest is a CSV with header: path,label[,generator,source]
- path: image path relative to DATA_ROOT (env var) or absolute
- label: 1 = AI-generated, 0 = real
- generator/source: optional metadata for per-group breakdowns
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from PIL import Image, ImageOps


@dataclass
class Sample:
    path: str
    label: int
    generator: str = ""
    source: str = ""


def data_root() -> str:
    return os.environ.get("DATA_ROOT", ".")


def load_manifest(csv_path: str) -> list[Sample]:
    samples = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            p = row["path"]
            if not os.path.isabs(p):
                p = os.path.join(data_root(), p)
            samples.append(Sample(
                path=p,
                label=int(row["label"]),
                generator=row.get("generator", "") or "",
                source=row.get("source", "") or "",
            ))
    return samples


def load_image(path: str) -> Image.Image:
    """RGB, EXIF-rotation-corrected. Raises on corrupt files (fail loud)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")
