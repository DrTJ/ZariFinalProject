# SSD Cascade + Deep SORT: real-time multi-object tracking

This project is a working version of the idea in the thesis *"Using SSD and Deep SORT to improve the speed and accuracy of real-time multi-object detection and tracking."* The thesis runs SSD-Lite and then SSD-Full on every frame. Here the two models form a **cascade** instead:

| Mode | What runs on each frame |
|---|---|
| `full` (non-cascade baseline) | SSD300-VGG16, the heavy and accurate model |
| `lite` | SSDLite320-MobileNetV3, the light and fast model |
| `cascade` | SSDLite on every frame. SSD300 runs only when it's needed (see below). |

SSD300 is called when one of these happens:

1. **first**: the first frame.
2. **uncertain**: SSDLite has at least `--min-uncertain` boxes scoring between `--low` and `--high`.
3. **count_change**: the number of confident objects changes by more than `--count-tolerance`, which usually means something entered or left the scene.
4. **keyframe**: `--refresh-interval` Lite-only frames have gone by in a row, which keeps drift in check.

On every other frame the cascade trusts SSDLite, so most frames cost only the light model.

Deep SORT tracking (`--track`) uses a Kalman filter, a MobileNetV2 appearance embedding and Hungarian matching. It can be added on top of any mode.

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch from https://pytorch.org first. The pretrained COCO weights download automatically on the first run (about 14 MB for SSDLite and 136 MB for SSD300).

## Run

A sample pedestrian video from OpenCV is already in `data/vtest.avi`.

```bash
# Compare all three modes (detection only)
python -m cascade_mot.run --video data/vtest.avi --classes person

# Full system: detection + Deep SORT, with annotated videos
python -m cascade_mot.run --video data/vtest.avi --classes person --track --save-video

# With ground truth (MOTChallenge gt.txt) for MOTA / IDF1 / ID switches
python -m cascade_mot.run --video MOT17-04.mp4 --classes person --track --gt MOT17-04/gt/gt.txt
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--modes full cascade` | run only some modes |
| `--max-frames 300` | how many frames to use |
| `--resize-width 960` | downscale 1080p/4K input first, which is much faster |
| `--device cuda --half` | GPU with FP16 |
| `--classes person car bus` | which COCO classes to keep (default: all) |
| `--low 0.3 --high 0.6` | width of the uncertain band. A narrower band means fewer Full calls. |
| `--count-tolerance 1` | ignore ±1 flicker in the object count |
| `--refresh-interval 30` | force Full after N Lite-only frames (0 = off) |
| `--no-pretrained` | random weights, only for smoke-testing without internet |

## Output (`results/`)

| File | Contents |
|---|---|
| `summary.csv`, `summary.json` | one row per mode (see the metrics below) |
| `comparison.png` | bar chart of FPS, recall vs Full, and Full-call rate |
| `frames_<mode>.csv` | per frame: was Full used, why, detection count, timings |
| `<mode>.mp4` | annotated video with `--save-video`. The banner shows FULL/LITE and the trigger reason. |

Metrics in the summary:

- `fps`, `ms_per_frame`: end-to-end speed (detection + tracking). Video decoding is excluded, and every mode gets identical frames.
- `full_call_rate`: the share of frames where SSD300 ran.
- `recall_vs_full`, `precision_vs_full`: how closely a mode reproduces the Full-only detections (IoU ≥ 0.5, same class). No labels needed.
- `speedup_vs_full`: FPS divided by the Full-only FPS.
- With `--track`: `unique_ids`, `mean_track_len`, `short_tracks(<10f)`. Fewer IDs and longer tracks usually mean fewer ID switches.
- With `--gt`: `mota`, `idf1`, `num_switches`, `motp` (mean IoU) and more.

## Tuning the cascade

- **Full is called too often** (high `full_call_rate`, small speedup): raise `--min-uncertain` to 2–3, set `--count-tolerance 1`, or narrow the band (`--low 0.35 --high 0.55`). Check `Cascade triggers:` in the output to see which reason dominates.
- **Recall vs Full is too low**: widen the band, or lower `--refresh-interval`.
- In crowded scenes some box is almost always uncertain, so `--min-uncertain` matters most there.

## Tests

```bash
pytest -q
```

The tests check the cascade decision rules and the metrics with scripted fake detectors, so they need no model download.

## Project layout

```
cascade_mot/
  detectors.py   torchvision SSD300 / SSDLite wrappers (timing, class filter, FP16)
  cascade.py     FullOnly, LiteOnly, Cascade strategies + CascadeConfig
  tracking.py    Deep SORT wrapper + track statistics
  metrics.py     agreement vs Full, MOTChallenge evaluation
  run.py         CLI benchmark, videos, CSV/JSON, chart
tests/           unit tests
data/vtest.avi   sample pedestrian video (OpenCV samples)
```

## Notes for the thesis

- Both models here are generic COCO-pretrained models. To match the thesis's SSD-Lite/SSD-Full idea, fine-tune SSDLite on the narrower dataset and SSD300 on the full one, then load them in `detectors.py`.
- Further speed-ups not included here: export to ONNX/TensorRT with INT8, and run detection and tracking in separate threads joined by a queue (Chapter 3.4 of the thesis).
