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

On every other frame the cascade trusts SSDLite and skips SSD300. How many frames that is depends on the video and the thresholds. On the sample video with the default thresholds it is 7% of frames (see "Results on the sample video" below).

Deep SORT tracking (`--track`) uses a Kalman filter, a MobileNetV2 appearance embedding and Hungarian matching. It can be added on top of any mode. The matching step is the project's own (`cascade_mot/tracking.py`): every match has to pass an appearance check (embedding plus a clothing-colour histogram), and a hidden person can only come back within walking distance of where they were last seen. This keeps IDs on the right person when people pass behind the streetlight pole or each other.

## Install

Install Python 3.11, then create the environment from this folder:

```bat
setup_windows.bat
```

```powershell
.\setup_windows.ps1
```

That creates `.venv-windows` from `requirements.txt`. Model weights are read from `models/`. On an NVIDIA GPU, install the CUDA build of PyTorch into `.venv-windows` before the rest of `requirements.txt`. Otherwise the run uses CPU.

## Run

A sample pedestrian video from OpenCV is already in `data/vtest.avi`.

```bat
run.bat --video data\vtest.avi --classes person
run.bat --video data\vtest.avi --classes person --track --save-video
```

```powershell
.\run.ps1 --video data/vtest.avi --classes person
.\run.ps1 --video data/vtest.avi --classes person --track --save-video
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
| `--max-age 120` | frames an ID is remembered while its person is hidden or not detected |
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

## Results on the sample video

Measured on 11 November 2026 on an Apple M2 Max (PyTorch on MPS): first 300 frames of `data/vtest.avi`, class `person`, default thresholds, with `--track --save-video`. Times are milliseconds per frame.

| Mode | FPS | Detection | Tracking | Full-call rate | Recall vs Full | Precision vs Full | Unique IDs | Mean track length |
|---|---|---|---|---|---|---|---|---|
| `full` | 1.76 | 65.9 | 502.4 | 100% | 100% | 100% | 11 | 160.8 |
| `lite` | 7.61 | 65.5 | 65.8 | 0% | 2.2% | 39.2% | 3 | 39.3 |
| `cascade` | 1.69 | 121.6 | 471.4 | 93.3% | 94.1% | 100% | 11 | 159.0 |

What these numbers say:

- The cascade is not faster than `full` here (0.96x). SSDLite's scores on this video are low, so SSD300 is still called on 93% of frames, and on those frames both models run.
- SSDLite is not cheaper than SSD300 on this machine. Detection takes about the same time for both, so no threshold setting can make the cascade faster than `full` on this hardware.
- The high FPS of `lite` comes from the tracker, not the detector. SSDLite finds a person in only 78 of the 300 frames, so the tracker has almost nothing to do.
- Tracking takes most of the time in `full` and `cascade`.

The step-by-step log of how this was measured, with the threshold sweep tables, is in `docs/cascade-light-model-tuning.md` (in Persian).

## Tuning the cascade

- **Try settings offline first**: `.venv-windows\Scripts\python.exe tools\threshold_sweep.py` (PowerShell: `.\.venv-windows\Scripts\python.exe tools\threshold_sweep.py`) runs both models once on the benchmark frames, times them, and prints the Full-call rate and recall each threshold setting would give. It only reads the video.
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
  tracking.py    Deep SORT with appearance-first matching + track statistics
  metrics.py     agreement vs Full, MOTChallenge evaluation
  run.py         CLI benchmark, videos, CSV/JSON, chart
tools/
  threshold_sweep.py   offline threshold sweep: timing, light-model recall, cascade settings
tests/           unit tests
docs/            work logs (Persian): track-ID fix report, light-model tuning log
data/vtest.avi   sample pedestrian video (OpenCV samples)
```

## Notes for the thesis

- Both models here are generic COCO-pretrained models. To match the thesis's SSD-Lite/SSD-Full idea, fine-tune SSDLite on the narrower dataset and SSD300 on the full one, then load them in `detectors.py`.
- Further speed-ups not included here: export to ONNX/TensorRT with INT8, and run detection and tracking in separate threads joined by a queue (Chapter 3.4 of the thesis).
