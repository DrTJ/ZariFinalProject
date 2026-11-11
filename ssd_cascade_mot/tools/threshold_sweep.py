"""Offline threshold sweep for the light model in the cascade.

Measures, on the same frames the benchmark uses, how the light model
(SSDLite320) compares with the heavy model (SSD300) and what each cascade
threshold setting would cost. It only reads the video and prints tables.
It does not change the pipeline or write result files.

The heavy model's boxes at score >= 0.5 are used as the reference, the same
way `recall_vs_full` is computed in cascade_mot/run.py. They are not human
labels.

Run from the project folder:

    python tools/threshold_sweep.py
    python tools/threshold_sweep.py --video data/vtest.avi --max-frames 300 --classes person
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cascade_mot.cascade import Cascade, CascadeConfig
from cascade_mot.detectors import SSDDetector, pick_device, resolve_classes
from cascade_mot.metrics import agreement
from cascade_mot.run import read_frames

FULL_CONF = 0.5      # the heavy model keeps boxes at this score, as in the pipeline
CACHE_MIN_SCORE = 0.05


def run_timed(det, frames):
    """Detections plus wall-clock ms per frame, including the copy back to the CPU."""
    det.warmup(frames[0])
    t0 = time.perf_counter()
    dets = [det(f) for f in frames]
    return dets, 1000 * (time.perf_counter() - t0) / len(frames)


def cache_detections(name, frames, device, classes):
    det = SSDDetector(name, device=device, pretrained=True, min_score=CACHE_MIN_SCORE, classes=classes)
    return run_timed(det, frames)


def time_on(name, frames, device, classes, n=30):
    det = SSDDetector(name, device=device, pretrained=True, min_score=CACHE_MIN_SCORE, classes=classes)
    return run_timed(det, frames[:n])[1]


def cutoff_timing(name, frames, device, classes, baseline, cutoffs, n=60, keep_from=0.3):
    """Time the model with a higher internal score cut-off.

    torchvision runs non-maximum suppression on every box above the model's own
    `score_thresh` (0.001 for SSDLite, 0.01 for SSD300). Raising it drops
    low-score boxes before that step. Boxes at `keep_from` and above are
    compared with the default model's output to confirm they do not change.
    """
    det = SSDDetector(name, device=device, pretrained=True, min_score=CACHE_MIN_SCORE, classes=classes)
    default = det.model.score_thresh
    rows = []
    for cut in (default,) + tuple(cutoffs):
        det.model.score_thresh = cut
        dets, ms = run_timed(det, frames[:n])
        same = 0
        for got, want in zip(dets, baseline[:n]):
            a, b = np.sort(got.above(keep_from).scores), np.sort(want.above(keep_from).scores)
            same += int(len(a) == len(b) and np.allclose(a, b, atol=1e-4))
        rows.append((cut, ms, same))
    return rows


def simulate(lite_dets, full_dets, low, high, min_uncertain, count_tolerance, lite_conf, refresh=30):
    """Replay the cascade decision on cached detections.

    `lite_conf` is the score a light-model box needs to be kept on frames where
    the heavy model is skipped. The pipeline today uses one `--conf` for both
    models, so a `lite_conf` other than 0.5 needs a small code change.
    """
    cfg = CascadeConfig(conf=FULL_CONF, low=low, high=high, min_uncertain=min_uncertain,
                        count_tolerance=count_tolerance, refresh_interval=refresh)
    casc = Cascade(lambda f: None, lambda f: None, cfg)
    floor = min(low, lite_conf)
    out, heavy = [], 0
    for lite, full in zip(lite_dets, full_dets):
        reason = casc.decide(lite.above(floor))
        if reason is None:
            casc.since_full += 1
            out.append(lite.above(lite_conf))
        else:
            casc.since_full = 0
            heavy += 1
            out.append(full.above(FULL_CONF))
    return out, heavy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="data/vtest.avi")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--classes", nargs="*", default=["person"])
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    frames, _ = read_frames(args.video, args.max_frames, None)
    device = pick_device(args.device)
    classes = resolve_classes(args.classes)
    n = len(frames)
    print(f"{n} frames of {frames[0].shape[1]}x{frames[0].shape[0]} | device {device} | classes {args.classes}")

    lite_dets, lite_ms = cache_detections("ssdlite", frames, device, classes)
    full_dets, full_ms = cache_detections("ssd300", frames, device, classes)
    ref = [d.above(FULL_CONF) for d in full_dets]
    n_ref = sum(len(d) for d in ref)

    # ---------------------------------------------------------------- timing
    print("\n== 1. Time per frame (ms), detection only ==")
    print(f"{'device':>8} {'light':>8} {'heavy':>8} {'heavy/light':>12}")
    print(f"{str(device):>8} {lite_ms:8.1f} {full_ms:8.1f} {full_ms / lite_ms:11.2f}x")
    if device.type != "cpu":
        cpu = torch.device("cpu")
        lite_cpu = time_on("ssdlite", frames, cpu, classes)
        full_cpu = time_on("ssd300", frames, cpu, classes)
        print(f"{'cpu':>8} {lite_cpu:8.1f} {full_cpu:8.1f} {full_cpu / lite_cpu:11.2f}x")

    # ------------------------------------------------------- score histogram
    print("\n== 2. Where the light model's scores fall ==")
    lite_scores = np.concatenate([d.scores for d in lite_dets]) if lite_dets else np.zeros(0)
    full_scores = np.concatenate([d.scores for d in full_dets])
    edges = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
    print(f"{'score':>11} {'light boxes':>12} {'heavy boxes':>12}")
    for a, b in zip(edges[:-1], edges[1:]):
        print(f"{a:5.2f}-{min(b, 1.0):4.2f} {int(((lite_scores >= a) & (lite_scores < b)).sum()):12d} "
              f"{int(((full_scores >= a) & (full_scores < b)).sum()):12d}")
    print(f"reference boxes (heavy, score >= {FULL_CONF}): {n_ref}, {n_ref / n:.2f} per frame")

    # ------------------------------------------------- light model on its own
    print("\n== 3. Light model alone at each score threshold, against the heavy reference ==")
    print(f"{'threshold':>9} {'boxes/frame':>12} {'recall':>8} {'precision':>10} {'frames with 0 boxes':>20}")
    for t in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1):
        test = [d.above(t) for d in lite_dets]
        a = agreement(ref, test)
        print(f"{t:9.2f} {sum(len(d) for d in test) / n:12.2f} {a['recall_vs_full']:8.1%} "
              f"{a['precision_vs_full']:10.1%} {sum(1 for d in test if len(d) == 0):20d}")

    # ------------------------------------------------------ cascade settings
    print("\n== 4. Cascade settings: heavy-model share and agreement with the heavy reference ==")
    rows = []
    highs = (0.6, 0.5, 0.45, 0.4, 0.35, 0.3)
    lows = (0.3, 0.25, 0.2)
    for high, low, mu, tol in itertools.product(highs, lows + ("none",), (1, 2, 3, 5), (0, 1)):
        lo = high if low == "none" else low
        if lo > high:
            continue
        for lite_conf in sorted({FULL_CONF, high, 0.3}):
            if lite_conf > FULL_CONF:
                continue
            out, heavy = simulate(lite_dets, full_dets, lo, high, mu, tol, lite_conf)
            a = agreement(ref, out)
            rows.append({"low": lo, "high": high, "min_unc": mu, "tol": tol, "lite_conf": lite_conf,
                         "heavy": heavy / n, "recall": a["recall_vs_full"], "precision": a["precision_vs_full"],
                         "det_ms": lite_ms + heavy / n * full_ms})

    def show(title, items):
        print(f"\n{title}")
        print(f"{'low':>5} {'high':>5} {'min_unc':>8} {'tol':>4} {'lite_conf':>10} {'heavy':>7} "
              f"{'recall':>8} {'precision':>10} {'detect ms':>10}")
        for r in items:
            print(f"{r['low']:5.2f} {r['high']:5.2f} {r['min_unc']:8d} {r['tol']:4d} {r['lite_conf']:10.2f} "
                  f"{r['heavy']:7.1%} {r['recall']:8.1%} {r['precision']:10.1%} {r['det_ms']:10.1f}")

    current = [r for r in rows if (r["low"], r["high"], r["min_unc"], r["tol"], r["lite_conf"]) == (0.3, 0.6, 1, 0, 0.5)]
    show("Current defaults (low 0.3, high 0.6, min_uncertain 1, count_tolerance 0):", current)
    for floor in (0.95, 0.90, 0.80, 0.70):
        ok = sorted((r for r in rows if r["recall"] >= floor), key=lambda r: (r["heavy"], -r["recall"]))
        show(f"Fewest heavy calls with recall >= {floor:.0%}:", ok[:6])
    show("Fewest heavy calls overall:", sorted(rows, key=lambda r: (r["heavy"], -r["recall"]))[:6])
    print(f"\nHeavy model alone costs {full_ms:.1f} ms per frame on {device}. "
          f"A cascade row is faster than that only if its 'detect ms' is lower.")

    # ------------------------------------------- internal cut-off and speed
    print("\n== 5. Speed against each model's internal score cut-off (first 60 frames) ==")
    print("'same' counts frames whose boxes at score >= 0.3 match the default model's.")
    devices = [device] + ([torch.device("cpu")] if device.type != "cpu" else [])
    for dev in devices:
        for name, base in (("ssdlite", lite_dets), ("ssd300", full_dets)):
            if dev != device:  # the cached boxes came from the main device
                base = run_timed(SSDDetector(name, device=dev, pretrained=True, min_score=CACHE_MIN_SCORE,
                                             classes=classes), frames[:60])[0]
            print(f"\n{name} on {dev}")
            print(f"{'cut-off':>8} {'ms/frame':>9} {'same':>8}")
            for cut, ms, same in cutoff_timing(name, frames, dev, classes, base, (0.1, 0.2, 0.3)):
                print(f"{cut:8.3f} {ms:9.1f} {same:5d}/60")


if __name__ == "__main__":
    main()
