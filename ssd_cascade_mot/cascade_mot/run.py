"""Benchmark Full-only vs Lite-only vs Cascade, with optional Deep SORT tracking.

Example:
    python -m cascade_mot.run --video data/vtest.avi --classes person --track --save-video

Outputs (in --out, default results/):
    summary.csv / summary.json   one row per mode: FPS, latency, Full-call rate, accuracy
    frames_<mode>.csv            per-frame log: Full used?, reason, detections, timings
    <mode>.mp4                   annotated video (with --save-video)
    comparison.png               bar chart of the main numbers
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .cascade import Cascade, CascadeConfig, FullOnly, LiteOnly
from .detectors import COCO_CATEGORIES, SSDDetector, pick_device, resolve_classes
from .metrics import MOTEvaluator, agreement


# ------------------------------------------------------------------ video I/O
def read_frames(path: str, max_frames: int, resize_width: int | None) -> tuple[list[np.ndarray], float]:
    """Decode once and keep frames in memory so every mode sees identical input
    and decoding time is excluded from the benchmark."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if resize_width and f.shape[1] > resize_width:
            h = int(f.shape[0] * resize_width / f.shape[1])
            f = cv2.resize(f, (resize_width, h), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"No frames read from {path}")
    return frames, fps


# BGR. Alternate pure red and pure yellow so neighboring people stay distinct
# and the label stays readable on the pavement.
_VIVID_BGR = ((0, 0, 255), (0, 255, 255))


def color_for(i: int) -> tuple[int, int, int]:
    return _VIVID_BGR[int(i) % 2]


def draw(frame, mode, dets, tracks, info, fps_now):
    img = frame.copy()
    if tracks is not None:
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.ltrb)
            c = color_for(t.track_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
            label = f"ID {t.track_id}"
            org = (x1, max(y1 - 8, 18))
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
    else:
        for (x1, y1, x2, y2), s, l in zip(dets.boxes.astype(int), dets.scores, dets.labels):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, f"{COCO_CATEGORIES[l]} {s:.2f}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    tag = "FULL" if info.used_full else "LITE"
    banner = f"{mode} | {tag} ({info.reason}) | {fps_now:.1f} FPS"
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, banner, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255) if info.used_full else (0, 255, 255), 1)
    return img


# ------------------------------------------------------------------ one mode
def run_mode(strategy, frames, args, out_dir: Path, tracker=None, video_fps=25.0):
    strategy.reset()
    if tracker is not None:
        tracker.reset()
    mot = MOTEvaluator(args.gt) if (args.gt and tracker is not None) else None

    writer = None
    if args.save_video:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_dir / f"{strategy.name}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))

    all_dets, rows, reasons = [], [], Counter()
    proc_time = 0.0
    if tracker is not None:
        tracker.paths.clear()
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        dets, info = strategy(frame)
        t1 = time.perf_counter()
        tracks = tracker.update(frame, dets) if tracker is not None else None
        t2 = time.perf_counter()
        proc_time += t2 - t0

        all_dets.append(dets)
        reasons[info.reason] += 1
        rows.append({"frame": i, "used_full": int(info.used_full), "reason": info.reason,
                     "n_dets": len(dets), "n_tracks": len(tracks) if tracks is not None else "",
                     "detect_ms": round(1000 * (t1 - t0), 2), "track_ms": round(1000 * (t2 - t1), 2)})
        if mot is not None:
            mot.update(i, tracks)
        if writer is not None:
            writer.write(draw(frame, strategy.name, dets, tracks, info, (i + 1) / proc_time))
        if args.verbose and i % 25 == 0:
            print(f"  [{strategy.name}] frame {i}/{len(frames)}  {info.reason}")

    if writer is not None:
        writer.release()
    if tracker is not None and tracker.paths:
        with open(out_dir / f"tracks_{strategy.name}.csv", "w", newline="") as fh:
            tw = csv.DictWriter(fh, fieldnames=list(tracker.paths[0]))
            tw.writeheader(); tw.writerows(tracker.paths)
    with open(out_dir / f"frames_{strategy.name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    n = len(frames)
    full_calls = sum(r["used_full"] for r in rows)
    res = {
        "mode": strategy.name,
        "frames": n,
        "fps": n / proc_time,
        "ms_per_frame": 1000 * proc_time / n,
        "detect_ms": float(np.mean([r["detect_ms"] for r in rows])),
        "full_calls": full_calls,
        "full_call_rate": full_calls / n,
        "reasons": dict(reasons),
    }
    if tracker is not None:
        res["track_ms"] = tracker.mean_ms
        res.update(tracker.summary())
    if mot is not None:
        res.update(mot.summary())
    return res, all_dets


# ------------------------------------------------------------------ reporting
def print_table(results):
    cols = [("mode", "{}"), ("fps", "{:.1f}"), ("ms_per_frame", "{:.1f}"), ("full_call_rate", "{:.0%}"),
            ("recall_vs_full", "{:.1%}"), ("precision_vs_full", "{:.1%}"), ("speedup_vs_full", "{:.2f}x"),
            ("unique_ids", "{}"), ("mean_track_len", "{:.1f}"), ("mota", "{:.3f}"), ("idf1", "{:.3f}"),
            ("num_switches", "{}")]
    cols = [(k, f) for k, f in cols if any(k in r for r in results)]
    cells = [[k for k, _ in cols]] + [[(f.format(r[k]) if k in r else "-") for k, f in cols] for r in results]
    widths = [max(len(row[j]) for row in cells) for j in range(len(cols))]
    for i, row in enumerate(cells):
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def save_chart(results, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [r["mode"] for r in results]
    colors = ["#4C78A8", "#F58518", "#54A24B"][: len(names)]
    panels = [("fps", "Frames per second (higher = faster)", 1),
              ("recall_vs_full", "Recall vs Full-only (%)", 100),
              ("full_call_rate", "Frames using heavy SSD300 (%)", 100)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (k, title, scale) in zip(axes, panels):
        vals = [r.get(k, 0) * scale for r in results]
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.bar_label(bars, fmt="%.1f")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SSD cascade vs non-cascade", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ main
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="input video file")
    p.add_argument("--modes", nargs="+", default=["full", "lite", "cascade"],
                   choices=["full", "lite", "cascade"])
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--resize-width", type=int, default=None, help="downscale wide videos first")
    p.add_argument("--classes", nargs="*", default=None, help="COCO classes to keep, e.g. person car")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--half", action="store_true", help="FP16 inference (CUDA only)")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--no-pretrained", action="store_true",
                   help="random weights: smoke-test the pipeline without downloading models")
    # cascade
    g = p.add_argument_group("cascade")
    g.add_argument("--conf", type=float, default=0.5)
    g.add_argument("--low", type=float, default=0.3)
    g.add_argument("--high", type=float, default=0.6)
    g.add_argument("--min-uncertain", type=int, default=1)
    g.add_argument("--count-tolerance", type=int, default=0)
    g.add_argument("--refresh-interval", type=int, default=30,
                   help="max consecutive Lite-only frames before forcing Full (0 = off)")
    # tracking
    g = p.add_argument_group("tracking")
    g.add_argument("--track", action="store_true", help="add Deep SORT tracking")
    g.add_argument("--max-age", type=int, default=120,
                   help="frames an ID is remembered while its person is hidden or not detected")
    g.add_argument("--n-init", type=int, default=3)
    g.add_argument("--gt", default=None, help="MOTChallenge gt.txt for MOTA/IDF1 (needs --track)")
    # output
    p.add_argument("--out", default="results")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    classes = resolve_classes(args.classes)

    print(f"Device: {device} | loading frames from {args.video}")
    frames, vfps = read_frames(args.video, args.max_frames, args.resize_width)
    print(f"{len(frames)} frames of {frames[0].shape[1]}x{frames[0].shape[0]}")

    need_full = any(m in ("full", "cascade") for m in args.modes)
    need_lite = any(m in ("lite", "cascade") for m in args.modes)
    common = dict(device=device, pretrained=not args.no_pretrained, min_score=min(args.low, args.conf),
                  classes=classes, half=args.half)
    full = SSDDetector("ssd300", **common) if need_full else None
    lite = SSDDetector("ssdlite", **common) if need_lite else None
    for d in (full, lite):
        if d is not None:
            d.warmup(frames[0])

    cfg = CascadeConfig(conf=args.conf, low=args.low, high=args.high, min_uncertain=args.min_uncertain,
                        count_tolerance=args.count_tolerance, refresh_interval=args.refresh_interval)
    strategies = {"full": lambda: FullOnly(full, args.conf),
                  "lite": lambda: LiteOnly(lite, args.conf),
                  "cascade": lambda: Cascade(lite, full, cfg)}

    tracker = None
    if args.track:
        from .tracking import Tracker
        tracker = Tracker(max_age=args.max_age, n_init=args.n_init,
                          embedder_gpu=device.type == "cuda", half=args.half)

    # Full-only is the accuracy reference, so run it first whenever possible.
    order = sorted(args.modes, key=lambda m: m != "full")
    results, dets_by_mode = [], {}
    for m in order:
        print(f"Running {m} ...")
        res, dets = run_mode(strategies[m](), frames, args, out_dir, tracker, vfps)
        results.append(res); dets_by_mode[m] = dets

    if "full" in dets_by_mode:
        base = next(r for r in results if r["mode"] == "full")
        for r in results:
            r.update(agreement(dets_by_mode["full"], dets_by_mode[r["mode"]]))
            r["speedup_vs_full"] = r["fps"] / base["fps"]

    print()
    print_table(results)
    for r in results:
        if r["mode"] == "cascade":
            print(f"\nCascade triggers: {r['reasons']}")

    flat_keys = sorted({k for r in results for k in r if k != "reasons"},
                       key=lambda k: list(results[0]).index(k) if k in results[0] else 99)
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flat_keys, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    meta = {"args": vars(args), "device": str(device), "frame_size": list(frames[0].shape[1::-1])}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    save_chart(results, out_dir / "comparison.png")
    print(f"\nSaved results to {out_dir.resolve()}")
    return results


if __name__ == "__main__":
    main()
