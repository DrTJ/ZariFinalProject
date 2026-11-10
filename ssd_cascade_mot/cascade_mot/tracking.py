"""Deep SORT tracking on top of any detection strategy.

Uses the `deep-sort-realtime` package: a Kalman filter predicts each track's
motion, and a MobileNetV2 appearance embedding plus the Hungarian algorithm
associates new detections with existing tracks (the same building blocks
described in the thesis).

Occlusion handling is tuned for a pedestrian walking behind a thin occluder
(the streetlight pole in data/vtest.avi). The stock defaults handed that
walker's ID to a person standing near the pole, then gave the walker a new ID
when they came out the other side. Two things caused it:

* The Mahalanobis gate uses box height and aspect. The pole clips both, so the
  real detection is rejected and the track is free to attach to someone else.
* IoU association runs for every track missed for exactly one frame, with a
  loose 0.7 cost (IoU only needs to be 0.3). The person circling by the pole
  is close enough to win that fallback.

A confirmed track with a steady velocity now stays on its predicted path.
If the walker is hidden, the track coasts instead of adopting a neighbor.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from deep_sort_realtime.deep_sort import iou_matching, linear_assignment
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deep_sort.tracker import Tracker as DeepSortCore
from deep_sort_realtime.deepsort_tracker import DeepSort

from .detectors import Detections


# Stock deep-sort-realtime values this project used before the occlusion fix.
DEFAULTS_BEFORE = dict(max_age=30, n_init=3, max_iou_distance=0.7,
                       max_cosine_distance=0.2, gating_only_position=False)

# Values used after the fix. max_iou_distance is 1 - IoU, so 0.4 means the
# box must still overlap by at least 0.6. Cosine is slightly looser so a
# walker whose crop was partly the pole can still match their own gallery.
MAX_AGE = 50
N_INIT = 3
MAX_IOU_DISTANCE = 0.4
MAX_COSINE_DISTANCE = 0.3
NN_BUDGET = 100

# Motion gate. Distances are fractions of the track's box height.
MOTION_SPEED = 1.5          # px/frame; slower tracks (a person circling) are not corridor-gated
NEAR_RADIUS_H = 0.75        # seen last frame: detection must sit near the prediction
COAST_RADIUS_STEP_H = 0.12  # extra radius per missed frame, while the pole hides the walker
COAST_RADIUS_CAP_H = 2.0
LATERAL_H = 0.45            # max sideways offset from the velocity heading
LATERAL_STEP_H = 0.06
BACK_H = 0.5                # reject a match that steps backwards against the heading
OCCLUDED_HEIGHT_RATIO = 0.65  # shorter than this vs the predicted box => pole clipped the crop


@dataclass
class TrackOut:
    track_id: int
    ltrb: tuple[float, float, float, float]
    label: int | None


class OcclusionTrack(Track):
    """Same as Deep SORT's track, but do not learn appearance from a clipped box.

    A crop that is mostly the pole (or the sign on it) pulls the gallery toward
    whoever happens to be standing there. Keeping the pre-occlusion feature
    lets the walker match themselves on the way out.
    """

    def update(self, kf, detection):
        prev_h = float(self.mean[3])
        xyah = detection.to_xyah()
        det_h = float(xyah[3])
        super().update(kf, detection)
        # Where the person was last actually seen. The Kalman mean runs ahead
        # during a detection gap and would otherwise miss them on the way back.
        self._last_center = (float(xyah[0]), float(xyah[1]))
        clipped = prev_h > 1.0 and det_h < OCCLUDED_HEIGHT_RATIO * prev_h
        if clipped and len(self.features) > 1:
            self.features.pop()


class OcclusionAwareTracker(DeepSortCore):
    """Deep SORT association with a path gate for steady walkers."""

    def __init__(self, metric, max_iou_distance: float, max_age: int, n_init: int):
        super().__init__(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            override_track_class=OcclusionTrack,
            gating_only_position=True,
        )

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost = self.metric.distance(features, targets)
            cost = linear_assignment.gate_cost_matrix(
                self.kf, cost, tracks, dets, track_indices, detection_indices,
                only_position=True,
            )
            return self._motion_gate(cost, tracks, dets, track_indices, detection_indices)

        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_a, unmatched_dets = linear_assignment.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age,
            self.tracks, detections, confirmed,
        )

        # IoU fallback is what stuck the walker's ID on the person by the pole:
        # appearance had rejected the clipped crop, then any nearby box with
        # IoU >= 0.3 was accepted. A track that already has a steady heading
        # coasts through that gap instead.
        iou_candidates = list(unconfirmed)
        for k in unmatched_a:
            if self.tracks[k].time_since_update != 1:
                continue
            if _steady(self.tracks[k]):
                continue
            iou_candidates.append(k)
        unmatched_a = [k for k in unmatched_a if k not in iou_candidates]

        def gated_iou(tracks, dets, track_indices, detection_indices):
            cost = iou_matching.iou_cost(tracks, dets, track_indices, detection_indices)
            return self._motion_gate(cost, tracks, dets, track_indices, detection_indices)

        matches_b, unmatched_b, unmatched_dets = linear_assignment.min_cost_matching(
            gated_iou, self.max_iou_distance, self.tracks, detections,
            iou_candidates, unmatched_dets,
        )
        matches = matches_a + matches_b
        unmatched = list(set(unmatched_a + unmatched_b))

        # A walker lost for more than one frame (the lite model dropped every
        # box, or the pole hid them) is still near the last measured center,
        # while the Kalman mean has kept moving. Match that detection back
        # before a new ID is born on the same person.
        coasting = [
            k for k in unmatched
            if self.tracks[k].is_confirmed() and self.tracks[k].time_since_update > 1
        ]
        unmatched = [k for k in unmatched if k not in coasting]
        matches_c, unmatched_c, unmatched_dets = linear_assignment.min_cost_matching(
            self._reacquire_cost, 5.0, self.tracks, detections, coasting, unmatched_dets,
        )
        matches = matches + matches_c
        unmatched = list(set(unmatched + unmatched_c))
        return matches, unmatched, unmatched_dets

    def _motion_gate(self, cost, tracks, dets, track_indices, detection_indices):
        cost = np.array(cost, dtype=float, copy=True)
        for row, ti in enumerate(track_indices):
            track = tracks[ti]
            if not _steady(track):
                continue
            px, py = float(track.mean[0]), float(track.mean[1])
            height = max(float(track.mean[3]), 1.0)
            vx, vy = float(track.mean[4]), float(track.mean[5])
            speed = float(np.hypot(vx, vy))
            missed = max(int(track.time_since_update) - 1, 0)
            # Just-seen tracks stay on a tight gate around the prediction.
            # Longer gaps use the last real center, so a person the filter
            # ran ahead of can still be recovered.
            if track.time_since_update > 1 and getattr(track, "_last_center", None):
                px, py = track._last_center
                lat_lim = 0.75 * height
                back_lim = 0.45 * height
                ahead_lim = speed * track.time_since_update + 0.4 * height
            else:
                radius = min(COAST_RADIUS_CAP_H, NEAR_RADIUS_H + COAST_RADIUS_STEP_H * missed) * height
                lat_lim = (LATERAL_H + LATERAL_STEP_H * missed) * height
                back_lim = BACK_H * height
                ahead_lim = radius
            ux, uy = vx / speed, vy / speed
            for col, di in enumerate(detection_indices):
                det = dets[di].to_xyah()
                dx = float(det[0]) - px
                dy = float(det[1]) - py
                lateral = abs(dx * -uy + dy * ux)
                along = dx * ux + dy * uy
                if lateral > lat_lim or along < -back_lim or along > ahead_lim:
                    cost[row, col] = linear_assignment.INFTY_COST
        return cost

    def _reacquire_cost(self, tracks, dets, track_indices, detection_indices):
        """Cost of giving a coasting track its own lost detection back.

        Anything outside the corridor is left at an infinite cost. Inside it,
        a box on the walking line beats a box that has stepped toward the pole.
        """
        cost = np.full((len(track_indices), len(detection_indices)), linear_assignment.INFTY_COST)
        for row, ti in enumerate(track_indices):
            track = tracks[ti]
            last = getattr(track, "_last_center", None)
            if last is None:
                continue
            lx, ly = last
            height = max(float(track.mean[3]), 1.0)
            vx, vy = float(track.mean[4]), float(track.mean[5])
            speed = float(np.hypot(vx, vy))
            if speed < 1e-3:
                continue
            ux, uy = vx / speed, vy / speed
            missed = int(track.time_since_update)
            for col, di in enumerate(detection_indices):
                det = dets[di].to_xyah()
                rx = float(det[0]) - lx
                ry = float(det[1]) - ly
                along = rx * ux + ry * uy
                lateral = abs(rx * -uy + ry * ux)
                if along < -0.45 * height or along > speed * missed + 0.4 * height:
                    continue
                if lateral > 0.75 * height:
                    continue
                expected = speed * missed
                cost[row, col] = (lateral / height) + 0.15 * abs(along - expected) / height
        return cost


def _steady(track) -> bool:
    """True when the Kalman heading is reliable enough to refuse a sideways jump."""
    if not track.is_confirmed() or track.hits < 5:
        return False
    vx, vy = float(track.mean[4]), float(track.mean[5])
    return float(np.hypot(vx, vy)) >= MOTION_SPEED


class Tracker:
    def __init__(self, max_age: int = MAX_AGE, n_init: int = N_INIT,
                 max_iou_distance: float = MAX_IOU_DISTANCE,
                 max_cosine_distance: float = MAX_COSINE_DISTANCE,
                 nn_budget: int | None = NN_BUDGET,
                 embedder_gpu: bool = False, half: bool = False, min_box: float = 4.0):
        self.min_box = min_box
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.embedder_gpu = embedder_gpu
        self.half = half
        self.reset()

    def reset(self):
        self.ds = DeepSort(
            max_age=self.max_age, n_init=self.n_init,
            max_iou_distance=self.max_iou_distance,
            max_cosine_distance=self.max_cosine_distance,
            nn_budget=self.nn_budget, gating_only_position=True,
            embedder="mobilenet", half=self.half, bgr=True,
            embedder_gpu=self.embedder_gpu,
        )
        # DeepSort builds its own core tracker. Swap in the occlusion-aware one
        # before any frames are seen; the metric (cosine threshold) stays.
        self.ds.tracker = OcclusionAwareTracker(
            self.ds.tracker.metric, self.max_iou_distance, self.max_age, self.n_init,
        )
        self.total_time = 0.0
        self.frames = 0
        self.track_lengths: dict[int, int] = defaultdict(int)
        self.paths: list[dict] = []

    def update(self, frame_bgr: np.ndarray, dets: Detections) -> list[TrackOut]:
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        raw = []
        for (x1, y1, x2, y2), s, l in zip(dets.boxes, dets.scores, dets.labels):
            # Clip to the frame and skip degenerate boxes: they would give the
            # appearance embedder an empty crop and crash it.
            x1, x2 = max(0.0, float(x1)), min(float(w), float(x2))
            y1, y2 = max(0.0, float(y1)), min(float(h), float(y2))
            if x2 - x1 < self.min_box or y2 - y1 < self.min_box:
                continue
            raw.append(([x1, y1, x2 - x1, y2 - y1], float(s), int(l)))
        tracks = self.ds.update_tracks(raw, frame=frame_bgr)
        out = []
        frame_idx = self.frames
        for t in tracks:
            if not t.is_confirmed():
                continue
            ltrb = tuple(float(v) for v in t.to_ltrb())
            coasting = t.time_since_update > 0
            self.paths.append({
                "frame": frame_idx,
                "id": int(t.track_id),
                "cx": round((ltrb[0] + ltrb[2]) / 2, 1),
                "cy": round((ltrb[1] + ltrb[3]) / 2, 1),
                "w": round(ltrb[2] - ltrb[0], 1),
                "h": round(ltrb[3] - ltrb[1], 1),
                "coasting": int(coasting),
            })
            # Keep the label up for a short gap so a dropped frame does not
            # look like the ID vanished. A long coast still stays off screen;
            # by then the box would have drifted off the person.
            if coasting and t.time_since_update > 8:
                continue
            tid = int(t.track_id)
            self.track_lengths[tid] += 1
            out.append(TrackOut(tid, ltrb, t.get_det_class()))
        self.total_time += time.perf_counter() - t0
        self.frames += 1
        return out

    @property
    def mean_ms(self) -> float:
        return 1000 * self.total_time / max(self.frames, 1)

    def summary(self) -> dict:
        lengths = list(self.track_lengths.values())
        return {
            "unique_ids": len(lengths),
            "mean_track_len": float(np.mean(lengths)) if lengths else 0.0,
            # Many short tracks usually mean fragmentation / ID switches.
            "short_tracks(<10f)": int(sum(1 for n in lengths if n < 10)),
        }
