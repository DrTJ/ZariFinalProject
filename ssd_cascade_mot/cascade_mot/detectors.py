"""Thin wrappers around torchvision's SSD models.

Every detector takes a BGR uint8 frame (as read by OpenCV) and returns a
`Detections` object holding numpy arrays, so the rest of the code never has to
care about torch tensors or devices.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# Weights live in <project>/models/hub/checkpoints and are read from disk.
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models"))

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
)

MODELS = {
    # name: (builder, weights enum)
    "ssdlite": (ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights.DEFAULT),
    "ssd300": (ssd300_vgg16, SSD300_VGG16_Weights.DEFAULT),
}

# Both models are trained on COCO and share torchvision's 91-slot label map.
COCO_CATEGORIES: list[str] = list(SSD300_VGG16_Weights.DEFAULT.meta["categories"])


@dataclass
class Detections:
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # xyxy, pixels
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))

    def __len__(self) -> int:
        return len(self.scores)

    def filter(self, mask: np.ndarray) -> "Detections":
        return Detections(self.boxes[mask], self.scores[mask], self.labels[mask])

    def above(self, thr: float) -> "Detections":
        return self.filter(self.scores >= thr)


def resolve_classes(names: list[str] | None) -> set[int] | None:
    """Turn class names such as ['person', 'car'] into COCO label ids."""
    if not names:
        return None
    ids = set()
    for n in names:
        if n not in COCO_CATEGORIES:
            raise ValueError(f"Unknown COCO class '{n}'. Valid names include: "
                             f"{', '.join(c for c in COCO_CATEGORIES if c != 'N/A')}")
        ids.add(COCO_CATEGORIES.index(n))
    return ids


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class SSDDetector:
    """A torchvision SSD model with timing and class filtering built in."""

    def __init__(self, name: str, device: torch.device, pretrained: bool = True,
                 min_score: float = 0.2, classes: set[int] | None = None,
                 half: bool = False):
        if name not in MODELS:
            raise ValueError(f"Unknown model '{name}', choose from {list(MODELS)}")
        builder, weights = MODELS[name]
        if pretrained:
            model = builder(weights=weights)
        else:  # random weights: only useful for smoke-testing the pipeline offline
            model = builder(weights=None, weights_backbone=None, num_classes=91)
        self.name = name
        self.device = device
        self.half = half and device.type == "cuda"
        self.model = model.eval().to(device)
        if self.half:
            self.model = self.model.half()
        self.min_score = min_score
        self.classes = classes
        self.calls = 0
        self.total_time = 0.0

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray) -> Detections:
        t0 = time.perf_counter()
        img = TF.to_tensor(frame_bgr[:, :, ::-1].copy()).to(self.device)
        if self.half:
            img = img.half()
        out = self.model([img])[0]
        self._sync()
        self.total_time += time.perf_counter() - t0
        self.calls += 1

        scores = out["scores"].float().cpu().numpy()
        keep = scores >= self.min_score
        dets = Detections(out["boxes"].float().cpu().numpy()[keep].astype(np.float32),
                          scores[keep].astype(np.float32),
                          out["labels"].cpu().numpy()[keep].astype(np.int64))
        if self.classes is not None:
            dets = dets.filter(np.isin(dets.labels, list(self.classes)))
        return dets

    def warmup(self, frame_bgr: np.ndarray, n: int = 3):
        for _ in range(n):
            self(frame_bgr)
        self.reset_stats()

    def reset_stats(self):
        self.calls = 0
        self.total_time = 0.0

    @property
    def mean_ms(self) -> float:
        return 1000 * self.total_time / max(self.calls, 1)
