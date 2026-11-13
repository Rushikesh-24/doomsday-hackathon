"""Lightweight multi-object tracker for sparsely sampled frames.

Detections are associated per class by greedy IoU matching, with a centre-distance fallback
because sampled frames (e.g. 2 fps) can move boxes far enough that IoU drops to ~0.
A track becomes *confirmed* after `min_hits` observations (or one very confident hit), and
is retired after `max_age` consecutive processed frames without a match. This is what
prevents the same staircase from being reported as a new object in every frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vision.postprocessing.nms import box_iou
from vision.schemas import Detection


@dataclass
class Track:
    track_id: int
    class_name: str
    category: str
    bbox: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 0
    misses: int = 0
    observations: list[tuple[int, float, Detection]] = field(default_factory=list)  # (frame, t, det)

    @property
    def max_conf(self) -> float:
        return max(o[2].confidence for o in self.observations)

    def predicted(self) -> np.ndarray:
        return self.bbox + self.velocity


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, center_threshold: float = 0.6, max_age: int = 3,
                 min_hits: int = 2, strong_conf: float | None = None):
        self.iou_threshold = iou_threshold
        self.center_threshold = center_threshold  # centre distance / sqrt(box area)
        self.max_age = max_age
        self.min_hits = min_hits
        self.strong_conf = strong_conf
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next = 1

    def _center_cost(self, a: np.ndarray, b: np.ndarray) -> float:
        ca, cb = (a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2
        scale = math.sqrt(max((a[2] - a[0]) * (a[3] - a[1]), 1.0))
        size_ratio = ((b[2] - b[0]) * (b[3] - b[1])) / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
        if not 0.33 < size_ratio < 3.0:
            return float("inf")
        return float(np.linalg.norm(ca - cb) / scale)

    def update(self, detections: list[Detection], frame_index: int, t: float) -> list[Detection]:
        """Assign `track_id` to each detection (in place) and return them."""
        unmatched = set(range(len(detections)))
        matched_tracks = set()
        for cls in {d.class_name for d in detections} | {tr.class_name for tr in self.active}:
            tr_idx = [i for i, tr in enumerate(self.active) if tr.class_name == cls]
            de_idx = [j for j in unmatched if detections[j].class_name == cls]
            if not tr_idx or not de_idx:
                continue
            tb = np.stack([self.active[i].predicted() for i in tr_idx])
            db = np.stack([np.array(detections[j].bbox.xyxy()) for j in de_idx])
            ious = box_iou(tb, db)
            pairs = []
            for a, i in enumerate(tr_idx):
                for b, j in enumerate(de_idx):
                    if ious[a, b] >= self.iou_threshold:
                        pairs.append((1.0 + ious[a, b], i, j))  # IoU matches rank first
                    else:
                        c = self._center_cost(tb[a], db[b])
                        if c <= self.center_threshold:
                            pairs.append((1.0 - c / (self.center_threshold + 1e-9) * 0.99, i, j))
            for _, i, j in sorted(pairs, reverse=True):
                if i in matched_tracks or j not in unmatched:
                    continue
                tr, d = self.active[i], detections[j]
                nb = np.array(d.bbox.xyxy())
                tr.velocity = 0.5 * tr.velocity + 0.5 * (nb - tr.bbox)
                tr.bbox = nb
                tr.hits += 1
                tr.misses = 0
                tr.observations.append((frame_index, t, d))
                d.track_id = tr.track_id
                matched_tracks.add(i)
                unmatched.discard(j)
        # age unmatched tracks
        keep = []
        for i, tr in enumerate(self.active):
            if i not in matched_tracks:
                tr.misses += 1
                tr.velocity *= 0.5
            (self.finished if tr.misses > self.max_age else keep).append(tr)
        self.active = keep
        # new tracks
        for j in sorted(unmatched):
            d = detections[j]
            tr = Track(self._next, d.class_name, d.category, np.array(d.bbox.xyxy()), hits=1,
                       observations=[(frame_index, t, d)])
            d.track_id = tr.track_id
            self._next += 1
            self.active.append(tr)
        return detections

    def is_confirmed(self, tr: Track) -> bool:
        return tr.hits >= self.min_hits or (self.strong_conf is not None and tr.max_conf >= self.strong_conf)

    def all_tracks(self) -> list[Track]:
        return self.finished + self.active

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.all_tracks() if self.is_confirmed(t)]
