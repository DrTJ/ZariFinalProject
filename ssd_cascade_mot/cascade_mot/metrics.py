"""Accuracy metrics.

1. Agreement with the Full-only baseline (no labels needed): how many of the
   heavy model's detections each strategy reproduces (recall) and how many of
   its own detections the heavy model confirms (precision).
2. Optional MOT metrics (MOTA, IDF1, ID switches...) against a MOTChallenge
   style ground-truth file, using `motmetrics`.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .detectors import Detections


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    # Zero-area boxes: count identical ones as a perfect match.
    same = np.all(a[:, None, :] == b[None, :, :], axis=2)
    return np.where(same, 1.0, iou).astype(np.float32)


def match_count(ref: Detections, test: Detections, thr: float = 0.5) -> int:
    """Greedy one-to-one matching of same-class boxes with IoU >= thr."""
    if len(ref) == 0 or len(test) == 0:
        return 0
    iou = iou_matrix(ref.boxes, test.boxes)
    iou[ref.labels[:, None] != test.labels[None, :]] = 0
    matched, used_r, used_t = 0, set(), set()
    for flat in np.argsort(-iou, axis=None):
        r, t = np.unravel_index(flat, iou.shape)
        if iou[r, t] < thr:
            break
        if r in used_r or t in used_t:
            continue
        used_r.add(r); used_t.add(t); matched += 1
    return matched


def agreement(ref_frames: list[Detections], test_frames: list[Detections], thr: float = 0.5) -> dict:
    hit = n_ref = n_test = 0
    for r, t in zip(ref_frames, test_frames):
        hit += match_count(r, t, thr)
        n_ref += len(r)
        n_test += len(t)
    recall = hit / max(n_ref, 1)
    precision = hit / max(n_test, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)
    return {"recall_vs_full": recall, "precision_vs_full": precision, "f1_vs_full": f1}


# ---------------------------------------------------------------- MOT metrics
def load_mot_gt(path: str, pedestrians_only: bool = True) -> dict[int, list[tuple[int, list[float]]]]:
    """Read MOTChallenge gt.txt: frame,id,x,y,w,h,conf[,class,visibility]. Frames are 1-based."""
    gt: dict[int, list] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            if len(p) >= 7 and float(p[6]) == 0:      # "ignore" entries
                continue
            if pedestrians_only and len(p) >= 8 and int(float(p[7])) != 1:
                continue
            gt[frame].append((tid, [x, y, x + w, y + h]))
    return gt


class MOTEvaluator:
    def __init__(self, gt_path: str, iou_thr: float = 0.5):
        import motmetrics as mm  # optional dependency
        self.mm = mm
        self.gt = load_mot_gt(gt_path)
        self.iou_thr = iou_thr
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, frame_idx0: int, tracks) -> None:
        gts = self.gt.get(frame_idx0 + 1, [])
        g_ids = [g[0] for g in gts]
        g_boxes = np.array([g[1] for g in gts], np.float32).reshape(-1, 4)
        t_ids = [t.track_id for t in tracks]
        t_boxes = np.array([t.ltrb for t in tracks], np.float32).reshape(-1, 4)
        dist = 1 - iou_matrix(g_boxes, t_boxes)
        dist[dist > 1 - self.iou_thr] = np.nan
        self.acc.update(g_ids, t_ids, dist)

    def summary(self) -> dict:
        mh = self.mm.metrics.create()
        names = ["mota", "motp", "idf1", "num_switches", "num_fragmentations",
                 "num_false_positives", "num_misses", "mostly_tracked"]
        df = mh.compute(self.acc, metrics=names, name="seq")
        row = df.iloc[0].to_dict()
        row["motp"] = 1 - row["motp"]  # report as mean IoU of matches (higher is better)
        return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in row.items()}
