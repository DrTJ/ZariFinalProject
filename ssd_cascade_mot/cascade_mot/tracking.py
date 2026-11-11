"""Deep SORT tracking on top of any detection strategy.

Uses the `deep-sort-realtime` package for the Kalman filter, the track
life-cycle and the MobileNetV2 appearance embedding. The step that decides
which detection belongs to which track is replaced, because the stock one
swaps IDs around the streetlight pole and its sign in data/vtest.avi.

What goes wrong there, and what this module does about it:

* Two people at the sign are closer than a body width, so position cannot
  tell them apart. Every match therefore has to pass an appearance check.
  The stock threshold (cosine 0.2-0.3) does not do that: two different people
  in this video are on average only 0.21 apart, the same person 0.10.
* The embedding alone is weak for 80 px pedestrians in dark clothes. A small
  clothing-colour histogram (torso + legs) is added to it. With both, the
  right person is the closest one in 97% of re-appearances (86% without).
* A box that covers two people, or a person half hidden by the sign, would
  teach a track the wrong appearance. Samples are only stored from boxes
  that stand clear of every other box and of every hidden track.
* A hidden person can only come back where they could have walked to. The
  search radius grows with walking speed and is capped.
* A person who was missed gets a provisional track. Before that track becomes
  a new ID, it is compared over several frames with the IDs that are
  currently hidden nearby, and handed back to the matching one.
* Someone walking in at the border of the picture is a new person, unless a
  hidden track was last seen at that border or was walking towards it. A
  track that walked out of the picture is dropped after a few frames.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from deep_sort_realtime.deep_sort.track import Track, TrackState
from deep_sort_realtime.deep_sort.tracker import Tracker as DeepSortCore
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.optimize import linear_sum_assignment

from .detectors import Detections


# Stock deep-sort-realtime values this project started from.
DEFAULTS_BEFORE = dict(max_age=30, n_init=3, max_iou_distance=0.7,
                       max_cosine_distance=0.2, gating_only_position=False)

MAX_AGE = 120               # frames a hidden track is remembered (12 s at 10 fps)
N_INIT = 3                  # detections needed before a track gets an ID
NN_BUDGET = 100             # appearance samples kept per track
TENTATIVE_GRACE = 2         # missed frames a not-yet-confirmed track survives
PROBATION_HITS = 10         # detections needed instead of N_INIT while a hidden ID is nearby
EDGE_GRACE = 5              # frames before a track that walked out of the picture is dropped
ENTRY_H = 0.35              # a box this close to the picture border is someone walking in
HEADING_H = 0.04            # per-frame speed towards a spot that counts as "was walking there"

# Appearance. Distance = (1 - cosine of embeddings) + COLOR_WEIGHT * (1 -
# overlap of colour histograms). 0 means identical.
COLOR_WEIGHT = 0.5
COLOR_BINS = (4, 6, 6)      # L, a, b bins per body part
APP_NEAR = 0.32             # limit while the track was seen in the previous frame
APP_LOST = 0.26             # limit when the track comes back after a gap
APP_HANDBACK = 0.27         # limit (median over several frames) for handing a new track back
APP_LEARN = 0.25            # a box further than this from the gallery is not stored as a sample
APP_MARGIN = 0.03           # a hidden track must beat every other candidate by this much

# Geometry. Distances are fractions of the person's box height.
NEAR_H = 0.5                # seen last frame: next box must be this close
WALK_H = 0.11               # fastest walk in this scene, per frame
SLACK_H = 0.5               # box jitter allowance on top of the walk
REACH_CAP_H = 2.5           # a hidden track is never re-attached further away than this
POS_WEIGHT = 0.2            # how much position counts next to appearance
MIN_HEIGHT_RATIO = 0.45     # box much shorter/taller than the track => not this person
MAX_HEIGHT_RATIO = 1.8

# Appearance samples are only taken from clean boxes.
CLEAN_IOU = 0.05            # overlap with another box or hidden track that spoils a sample
DUPLICATE_IOU = 0.4         # a new box overlapping an owned box this much is the same person
DUPLICATE_COVER = 0.5       # ... and so is a new box with this share of it inside owned boxes
HIDDEN_AGE = 15             # a hidden track guards its place for this many frames
OCCLUDED_HEIGHT_RATIO = 0.65  # box this much shorter than before => clipped by the pole

_INF = 1e5


@dataclass
class TrackOut:
    track_id: int
    ltrb: tuple[float, float, float, float]
    label: int | None


def color_descriptor(frame_bgr: np.ndarray, ltwh) -> np.ndarray:
    """Clothing colours of the torso and the legs as a unit-length vector.

    Only the middle half of the box is read, so road and grass next to the
    person stay out of it.
    """
    x, y, w, h = (float(v) for v in ltwh)
    fh, fw = frame_bgr.shape[:2]
    xa = int(np.clip(round(x + 0.25 * w), 0, fw - 1))
    xb = int(np.clip(round(x + 0.75 * w), xa + 1, fw))
    parts = []
    for top, bottom in ((0.15, 0.50), (0.55, 0.92)):
        ya = int(np.clip(round(y + top * h), 0, fh - 1))
        yb = int(np.clip(round(y + bottom * h), ya + 1, fh))
        lab = cv2.cvtColor(np.ascontiguousarray(frame_bgr[ya:yb, xa:xb]), cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0, 1, 2], None, list(COLOR_BINS),
                            [0, 256, 96, 160, 96, 160]).ravel()
        parts.append(np.sqrt(hist / max(float(hist.sum()), 1.0)))
    vec = np.concatenate(parts)
    return vec / (float(np.linalg.norm(vec)) + 1e-9)


def appearance_feature(embed, color) -> np.ndarray:
    """Embedding and colours in one vector, so one dot product gives the distance."""
    embed = np.asarray(embed, dtype=np.float32)
    embed = embed / (float(np.linalg.norm(embed)) + 1e-9)
    return np.concatenate([embed, np.sqrt(COLOR_WEIGHT) * np.asarray(color, dtype=np.float32)])


def _ltwh_overlap(a, b) -> float:
    """Shared area of two boxes in pixels."""
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return float(iw * ih) if iw > 0 and ih > 0 else 0.0


def _ltwh_iou(a, b) -> float:
    inter = _ltwh_overlap(a, b)
    return float(inter / (a[2] * a[3] + b[2] * b[3] - inter)) if inter > 0 else 0.0


class OcclusionTrack(Track):
    """Deep SORT track that keeps its own clean appearance gallery."""

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None, **kwargs):
        super().__init__(mean, covariance, track_id, n_init, max_age, feature=feature, **kwargs)
        self.gallery = [np.asarray(feature, dtype=np.float32)]
        self.recent = [self.gallery[0]]     # last few samples, clean or not
        self.entered = False                # first seen at the border of the picture
        self.last_center = (float(mean[0]), float(mean[1]))
        self.first_center = self.last_center
        self.last_height = float(mean[3])
        ltwh = kwargs.get("original_ltwh")
        if ltwh is None:
            width = float(mean[2]) * float(mean[3])
            ltwh = (mean[0] - width / 2, mean[1] - mean[3] / 2, width, mean[3])
        self.last_ltwh = tuple(float(v) for v in ltwh)

    def appearance_distance(self, features: np.ndarray) -> np.ndarray:
        """Distance from each feature row to the closest stored sample."""
        sims = np.asarray(features) @ np.asarray(self.gallery).T
        return (1.0 + COLOR_WEIGHT) - sims.max(axis=1)

    def update(self, kf, detection, clean: bool = True):
        xyah = detection.to_xyah()
        clipped = float(xyah[3]) < OCCLUDED_HEIGHT_RATIO * self.last_height
        feature = np.asarray(detection.feature, dtype=np.float32)
        # A crop that suddenly looks different is usually two people in one
        # box, or the sign in front of the person. Do not learn from it.
        # A provisional track is exempt: its first view is often half a
        # person stepping into the picture, and it has nothing else yet.
        if self.is_confirmed():
            clean = clean and float(self.appearance_distance(feature[None])[0]) <= APP_LEARN
        super().update(kf, detection)
        self.features = [feature]           # the library only needs the latest one
        self.recent = (self.recent + [feature])[-N_INIT:]
        self.last_center = (float(xyah[0]), float(xyah[1]))
        self.last_ltwh = tuple(float(v) for v in detection.get_ltwh())
        if clean and not clipped:
            self.gallery = (self.gallery + [feature])[-NN_BUDGET:]
        if not clipped:
            self.last_height = float(xyah[3])

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            if self.time_since_update > TENTATIVE_GRACE:
                self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def reach(self) -> float:
        """How far (in box heights) this person can be from where they were last seen."""
        return min(REACH_CAP_H, SLACK_H + WALK_H * self.time_since_update)


class OcclusionAwareTracker(DeepSortCore):
    """Deep SORT bookkeeping with appearance-first association."""

    def __init__(self, metric, max_age: int, n_init: int):
        super().__init__(metric, max_age=max_age, n_init=n_init,
                         override_track_class=OcclusionTrack)
        self.frame_size: tuple[int, int] | None = None   # (width, height), set by Tracker

    def update(self, detections, today=None):
        matches, unmatched_tracks, unmatched_dets = self._match(detections)
        matches, unmatched_tracks = self._hand_back(matches, unmatched_tracks, detections)

        # A second box on a person who already has an ID (the detector does
        # this for two people standing together) must not grow into a new ID.
        owned = [detections[di].get_ltwh() for ti, di in matches if self.tracks[ti].is_confirmed()]

        def duplicate(di):
            box = detections[di].get_ltwh()
            if any(_ltwh_iou(box, o) > DUPLICATE_IOU for o in owned):
                return True
            # A box drawn across two neighbours overlaps neither of them
            # much, but most of it lies inside their boxes.
            return sum(_ltwh_overlap(box, o) for o in owned) > DUPLICATE_COVER * box[2] * box[3]

        dropped = [(ti, di) for ti, di in matches
                   if not self.tracks[ti].is_confirmed() and duplicate(di)]
        matches = [m for m in matches if m not in dropped]
        unmatched_tracks = unmatched_tracks + [ti for ti, _ in dropped]
        unmatched_dets = [di for di in unmatched_dets if not duplicate(di)]

        matched_tracks = {ti for ti, _ in matches}
        for ti, di in matches:
            clean = self._is_clean(ti, di, detections, matched_tracks)
            self.tracks[ti].update(self.kf, detections[di], clean=clean)
        for ti in unmatched_tracks:
            track = self.tracks[ti]
            track.mark_missed()
            if track.time_since_update > EDGE_GRACE and self._left_the_picture(track):
                track.state = TrackState.Deleted
        for di in unmatched_dets:
            self._initiate_track(detections[di])
            box = detections[di].get_ltwh()
            self.tracks[-1].entered = self._border_gap(box) < ENTRY_H * box[3]
            self.tracks[-1].first_center = self.tracks[-1].last_center

        self.del_tracks_ids = [t.track_id for t in self.tracks if t.is_deleted()]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _match(self, detections):
        n_tracks, n_dets = len(self.tracks), len(detections)
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        xyah = np.array([d.to_xyah() for d in detections], dtype=float)
        centers, heights = xyah[:, :2], xyah[:, 3]
        feats = np.array([d.feature for d in detections], dtype=np.float32)
        # Boxes at the border of the picture are people walking in. A hidden
        # track may only claim one if it was last seen at the border itself
        # or was walking towards that spot.
        entering = np.array([self._border_gap(d.get_ltwh()) < ENTRY_H * h
                             for d, h in zip(detections, heights)])

        cost = np.full((n_tracks, n_dets), _INF)
        app = np.full((n_tracks, n_dets), _INF)
        in_reach = np.zeros((n_tracks, n_dets), dtype=bool)
        for ti, track in enumerate(self.tracks):
            h = max(track.last_height, 1.0)
            d_last = np.linalg.norm(centers - np.array(track.last_center), axis=1) / h
            d_pred = np.linalg.norm(centers - track.mean[:2], axis=1) / h
            pos = np.minimum(d_last, d_pred)
            ratio = heights / h
            size_ok = (ratio > MIN_HEIGHT_RATIO) & (ratio < MAX_HEIGHT_RATIO)
            app[ti] = track.appearance_distance(feats)
            if track.time_since_update <= 1:
                in_reach[ti] = (pos <= NEAR_H) & size_ok
                ok = in_reach[ti] & (app[ti] <= APP_NEAR)
            else:
                in_reach[ti] = (d_last <= track.reach()) & size_ok
                in_reach[ti] &= ~entering | self._could_be_at_border(track, centers)
                # A box that fails here can still be handed back over
                # several frames.
                ok = in_reach[ti] & (app[ti] <= APP_LOST)
            cost[ti, ok] = app[ti, ok] + POS_WEIGHT * np.minimum(pos[ok], NEAR_H)

        # A track that comes back after a gap has only its looks to go on. If
        # another hidden track that could also be standing there looks about
        # as similar, do not guess; _hand_back decides over several frames.
        confirmed = [ti for ti, t in enumerate(self.tracks) if t.is_confirmed()]
        hidden = [ti for ti in confirmed if self.tracks[ti].time_since_update > 1]
        for ti in hidden:
            for di in np.flatnonzero(cost[ti] < _INF):
                rivals = [app[tj, di] for tj in hidden if tj != ti and in_reach[tj, di]]
                if rivals and min(rivals) < app[ti, di] + APP_MARGIN:
                    cost[ti, di] = _INF

        # People who already have an ID choose first. Provisional tracks only
        # get the boxes nobody claimed, so they cannot take over a person.
        provisional = [ti for ti in range(n_tracks) if not self.tracks[ti].is_confirmed()]
        matches = _assign(cost, confirmed, list(range(n_dets)))
        taken = {di for _, di in matches}
        matches += _assign(cost, provisional, [di for di in range(n_dets) if di not in taken])
        matched_t = {ti for ti, _ in matches}
        matched_d = {di for _, di in matches}
        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_t]
        unmatched_dets = [i for i in range(n_dets) if i not in matched_d]
        return matches, unmatched_tracks, unmatched_dets

    def _hand_back(self, matches, unmatched_tracks, detections):
        """Give a provisional track back to a hidden ID instead of creating a new one.

        Runs once the provisional track has N_INIT views of the person, which
        is steadier evidence than the single box the hidden track was offered
        when the person came back. While a hidden ID is within walking
        distance and looks have not settled it, the provisional track stays
        on probation: it needs PROBATION_HITS detections to become a new ID.
        """
        hidden = [ti for ti in unmatched_tracks
                  if self.tracks[ti].is_confirmed() and self.tracks[ti].time_since_update > 1]
        out = []
        for ti, di in matches:
            new = self.tracks[ti]
            if not new.is_tentative() or new.hits + 1 < self.n_init:
                out.append((ti, di))
                continue
            det = detections[di]
            xyah = det.to_xyah()
            nearby = []
            for tj in hidden:
                old = self.tracks[tj]
                h = max(old.last_height, 1.0)
                dist = float(np.hypot(xyah[0] - old.last_center[0], xyah[1] - old.last_center[1])) / h
                if dist > old.reach() or not MIN_HEIGHT_RATIO < float(xyah[3]) / h < MAX_HEIGHT_RATIO:
                    continue
                # The new track walked in from the border; this hidden one
                # was well inside the picture and not heading there.
                if new.entered and not self._could_be_at_border(old, np.array([new.first_center]))[0]:
                    continue
                nearby.append(tj)
            if not nearby:
                new._n_init = self.n_init
                out.append((ti, di))
                continue
            samples = np.array(new.recent[-(self.n_init - 1):] + [det.feature], dtype=np.float32)
            scores = {tj: float(np.median(self.tracks[tj].appearance_distance(samples)))
                      for tj in nearby}
            best = min(nearby, key=lambda tj: scores[tj])
            # Nobody appears out of thin air in the middle of the picture. If
            # probation is over and the person did not walk in at the border,
            # they are one of the hidden IDs, so a looser match is enough.
            last_chance = new.hits + 1 >= PROBATION_HITS and not new.entered
            limit = APP_NEAR if last_chance else APP_HANDBACK
            # Two hidden people who both look like this one: wait.
            settled = scores[best] <= limit and all(
                scores[tj] >= scores[best] + APP_MARGIN for tj in nearby if tj != best)
            if not settled:
                new._n_init = PROBATION_HITS
                out.append((ti, di))
                continue
            old = self.tracks[best]
            old.mean, old.covariance = new.mean.copy(), new.covariance.copy()
            new.state = TrackState.Deleted
            hidden.remove(best)
            unmatched_tracks = [k for k in unmatched_tracks if k != best]
            out.append((best, di))
        return out, unmatched_tracks

    def _border_gap(self, ltwh) -> float:
        """Pixels between a box and the nearest border of the picture."""
        if self.frame_size is None:
            return float("inf")
        width, height = self.frame_size
        x, y, w, h = ltwh
        return float(min(x, y, width - (x + w), height - (y + h)))

    def _could_be_at_border(self, track, centers) -> np.ndarray:
        """Per position: may this hidden track be the person seen there at the border?

        Yes if it was last seen at the border itself, or was walking towards
        that position when it disappeared. A person standing in the middle of
        the picture does not turn up at the edge a second later.
        """
        h = max(track.last_height, 1.0)
        if self._border_gap(track.last_ltwh) <= h:
            return np.ones(len(centers), dtype=bool)
        step = np.asarray(centers, dtype=float) - np.array(track.last_center)
        dist = np.linalg.norm(step, axis=1) + 1e-9
        towards = (step @ np.asarray(track.mean[4:6], dtype=float)) / dist
        return towards >= HEADING_H * h

    def _left_the_picture(self, track) -> bool:
        """True when the track was last seen at the border, walking outwards."""
        if self.frame_size is None:
            return False
        width, height = self.frame_size
        x, y = track.last_center
        vx, vy = float(track.mean[4]), float(track.mean[5])
        margin = 0.6 * track.last_height
        return ((x < margin and vx < 0) or (x > width - margin and vx > 0)
                or (y < margin and vy < 0) or (y > height - margin and vy > 0))

    def _is_clean(self, ti, di, detections, matched_tracks) -> bool:
        """True when the box shows one person and nothing else."""
        box = detections[di].get_ltwh()
        for dj, other in enumerate(detections):
            if dj != di and _ltwh_iou(box, other.get_ltwh()) > CLEAN_IOU:
                return False
        for tj, other in enumerate(self.tracks):
            if tj == ti or tj in matched_tracks or not other.is_confirmed():
                continue
            if other.time_since_update > HIDDEN_AGE:
                continue
            if _ltwh_iou(box, other.to_ltwh()) > CLEAN_IOU:
                return False
        return True


def _assign(cost, track_idx, det_idx):
    """Cheapest one-to-one pairing of the given tracks and boxes."""
    if not track_idx or not det_idx:
        return []
    sub = cost[np.ix_(track_idx, det_idx)]
    rows, cols = linear_sum_assignment(sub)
    return [(track_idx[r], det_idx[c]) for r, c in zip(rows, cols) if sub[r, c] < _INF]


class Tracker:
    def __init__(self, max_age: int = MAX_AGE, n_init: int = N_INIT,
                 nn_budget: int | None = NN_BUDGET,
                 embedder_gpu: bool = False, half: bool = False, min_box: float = 4.0):
        self.min_box = min_box
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.embedder_gpu = embedder_gpu
        self.half = half
        self.reset()

    def reset(self):
        self.ds = DeepSort(
            max_age=self.max_age, n_init=self.n_init, nn_budget=self.nn_budget,
            embedder="mobilenet", half=self.half, bgr=True,
            embedder_gpu=self.embedder_gpu,
        )
        # DeepSort builds its own core tracker. Swap in the occlusion-aware one
        # before any frames are seen.
        self.ds.tracker = OcclusionAwareTracker(self.ds.tracker.metric, self.max_age, self.n_init)
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
        embeds = self.ds.generate_embeds(frame_bgr, raw) if raw else []
        feats = [appearance_feature(e, color_descriptor(frame_bgr, r[0])) for e, r in zip(embeds, raw)]
        self.ds.tracker.frame_size = (w, h)
        tracks = self.ds.update_tracks(raw, embeds=feats)
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
