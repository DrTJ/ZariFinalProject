"""Detection strategies: Full-only, Lite-only and the Lite -> Full cascade.

All strategies expose the same interface:

    dets, info = strategy(frame)

where `info` is a `StepInfo` saying whether the heavy model was used and why.
Detectors are plain callables (frame -> Detections), so the logic can be unit
tested with fake detectors and no neural network at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .detectors import Detections

Detector = Callable[[np.ndarray], Detections]


@dataclass
class StepInfo:
    used_full: bool
    reason: str  # "always", "never", "first", "uncertain", "count_change", "keyframe", "lite"


@dataclass
class CascadeConfig:
    conf: float = 0.5           # final score threshold for a detection to be kept
    low: float = 0.3            # lower edge of the "uncertain" band
    high: float = 0.6           # Lite scores >= high count as confident
    min_uncertain: int = 1      # how many uncertain Lite boxes trigger Full
    count_tolerance: int = 0    # allowed change in confident-object count before Full
    refresh_interval: int = 30  # force Full every N frames (0 = never)

    def validate(self):
        if not 0 <= self.low <= self.high <= 1:
            raise ValueError("Need 0 <= low <= high <= 1")
        if self.min_uncertain < 1:
            raise ValueError("min_uncertain must be >= 1")


class FullOnly:
    """Baseline: heavy model on every frame (the non-cascade option)."""
    name = "full"

    def __init__(self, full: Detector, conf: float = 0.5):
        self.full, self.conf = full, conf

    def reset(self):
        pass

    def __call__(self, frame):
        return self.full(frame).above(self.conf), StepInfo(True, "always")


class LiteOnly:
    """Fast lower bound: light model on every frame."""
    name = "lite"

    def __init__(self, lite: Detector, conf: float = 0.5):
        self.lite, self.conf = lite, conf

    def reset(self):
        pass

    def __call__(self, frame):
        return self.lite(frame).above(self.conf), StepInfo(False, "never")


class Cascade:
    """Run Lite on every frame; escalate to Full only when it is worth it."""
    name = "cascade"

    def __init__(self, lite: Detector, full: Detector, cfg: CascadeConfig | None = None):
        self.lite, self.full = lite, full
        self.cfg = cfg or CascadeConfig()
        self.cfg.validate()
        self.reset()

    def reset(self):
        self.prev_count: int | None = None
        self.since_full = 0

    def decide(self, lite_dets: Detections) -> str | None:
        """Return the reason to call Full, or None to trust Lite."""
        c = self.cfg
        s = lite_dets.scores
        count = int((s >= c.high).sum())
        uncertain = int(((s >= c.low) & (s < c.high)).sum())
        prev, self.prev_count = self.prev_count, count

        if prev is None:
            return "first"
        if uncertain >= c.min_uncertain:
            return "uncertain"
        if abs(count - prev) > c.count_tolerance:
            return "count_change"
        if c.refresh_interval and self.since_full >= c.refresh_interval:
            return "keyframe"
        return None

    def __call__(self, frame):
        lite_dets = self.lite(frame)
        reason = self.decide(lite_dets)
        if reason is None:
            self.since_full += 1
            return lite_dets.above(self.cfg.conf), StepInfo(False, "lite")
        self.since_full = 0
        return self.full(frame).above(self.cfg.conf), StepInfo(True, reason)
