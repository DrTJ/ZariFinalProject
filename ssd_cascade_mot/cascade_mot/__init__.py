"""Cascade SSD detection + Deep SORT tracking for real-time multi-object tracking.

Detection strategies:
    full     - heavy SSD300-VGG16 on every frame (baseline, "non-cascade")
    lite     - light SSDLite320-MobileNetV3 on every frame
    cascade  - SSDLite on every frame, SSD300 only when SSDLite is unsure,
               the object count changes, or a refresh keyframe is due

GitHub rejects files larger than 100MB, so the SSD300 weights are stored as
pieces under vendor/parts and joined here before import.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(_ROOT / "models"))


def _ssd300_weights(root: Path) -> Path:
    return root / "models" / "hub" / "checkpoints" / "ssd300_vgg16_coco-b556d3b4.pth"


def _assemble_vendor_parts() -> None:
    root = _ROOT
    jobs = (
        ("ssd300_vgg16_coco-b556d3b4.pth.part*", _ssd300_weights(root)),
    )
    parts_dir = root / "vendor" / "parts"
    for pattern, dest in jobs:
        parts = sorted(parts_dir.glob(pattern))
        if not parts:
            continue
        total = sum(part.stat().st_size for part in parts)
        if dest.is_file() and dest.stat().st_size == total:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with tmp.open("wb") as out:
            for part in parts:
                with part.open("rb") as inp:
                    shutil.copyfileobj(inp, out)
        tmp.replace(dest)


_assemble_vendor_parts()

__version__ = "1.0.0"
