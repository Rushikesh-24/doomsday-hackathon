"""Facility tracking across keyframes, reusing the existing detection engine (no new detector).

Two 2D trackers are available and were compared on real keyframes (see docs/SPATIAL.md):
  * iou_centre  - existing vision.tracking.IoUTracker (greedy IoU + centre-distance fallback), designed for
                  sparse frames where boxes move far between samples
  * bytetrack   - Ultralytics BYTETracker (Kalman + two-stage association), designed for dense video
2D tracks are later merged in 3D (registration stage) so a facility leaving and re-entering view, or seen
from another angle, keeps one identity.
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from vision.schemas import ImageResult, SourceRef
from vision.tracking.tracker import IoUTracker

TRACKED_CLASSES = {"staircase", "ramp", "elevator", "door", "entrance_exit", "handrail", "tactile_paving",
                   "accessibility_symbol", "signboard", "obstacle", "corridor", "walkway"}


class _ByteAdapter:
    def __init__(self, fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker
        args = Namespace(track_high_thresh=0.25, track_low_thresh=0.05, new_track_thresh=0.25,
                         track_buffer=max(3, int(round(2 * fps))), match_thresh=0.8, fuse_score=True)
        self.tr = BYTETracker(args)

    def update(self, dets, frame_idx, t):
        if not dets:
            self.tr.update(_Res(np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
            return dets
        classes = sorted({d.class_name for d in dets})
        xyxy = np.array([d.bbox.xyxy() for d in dets], dtype=np.float32)
        conf = np.array([d.confidence for d in dets], dtype=np.float32)
        cls = np.array([classes.index(d.class_name) for d in dets], dtype=np.float32)
        out = self.tr.update(_Res(xyxy, conf, cls))
        # out rows: x1,y1,x2,y2,track_id,score,cls,det_index
        for row in out:
            j = int(row[-1])
            if 0 <= j < len(dets):
                dets[j].track_id = int(row[4])
        return dets


class _Res:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls
        self.xywh = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                              xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1) if len(xyxy) else np.zeros((0, 4))

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        return _Res(self.xyxy[i], self.conf[i], self.cls[i])


def detect_frames(frames_dir: Path, frames: list[dict], out_dir: Path, pipeline=None, model: str | None = None,
                  progress=None) -> list[ImageResult]:
    """Run the existing vision pipeline once per keyframe; cache per-frame JSON (resumable)."""
    from vision.inference.pipeline import get_pipeline
    pipe = pipeline or get_pipeline()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, f in enumerate(frames):
        cache = out_dir / (Path(f["name"]).stem + ".json")
        if cache.exists():
            results.append(ImageResult.model_validate_json(cache.read_text()))
            continue
        bgr = cv2.imread(str(frames_dir / f["name"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = pipe.analyze_array(rgb, f["name"], model=model,
                               source=SourceRef(name=f["name"], kind="video_frame", frame_index=f["frame_index"],
                                                timestamp_s=f["t"]),
                               save_dir=None, save_annotated=False, save_json=False)
        cache.write_text(r.model_dump_json())
        results.append(r)
        if progress:
            progress((i + 1) / len(frames), f"detected {i + 1}/{len(frames)} frames")
    return results


def track(results: list[ImageResult], frames: list[dict], method: str = "iou_centre", fps: float = 5.0,
          min_hits: int = 2, classes: set[str] | None = None) -> dict:
    classes = classes or TRACKED_CLASSES
    if method == "bytetrack":
        tr = _ByteAdapter(fps)
    else:
        tr = IoUTracker(iou_threshold=0.25, center_threshold=0.8, max_age=4, min_hits=min_hits)
    tracks: dict[int, dict] = {}
    for r, f in zip(results, frames):
        dets = [d.model_copy(deep=True) for d in r.detections if d.class_name in classes]
        for d in dets:
            d.track_id = None
        tr.update(dets, f["frame_index"], f["t"])
        for d in dets:
            if d.track_id is None:
                continue
            T = tracks.setdefault(d.track_id, {"track_id": d.track_id, "class_name": d.class_name, "observations": []})
            T["observations"].append({"frame": f["name"], "frame_index": f["frame_index"], "t": f["t"],
                                      "detection_id": d.id, "bbox": d.bbox.xyxy(), "confidence": d.confidence,
                                      "prompt": d.prompt})
    out = []
    for T in tracks.values():
        obs = T["observations"]
        confs = [o["confidence"] for o in obs]
        T.update(first_t=obs[0]["t"], last_t=obs[-1]["t"], n_observations=len(obs),
                 max_confidence=round(max(confs), 4), mean_confidence=round(float(np.mean(confs)), 4),
                 # repeated independent sightings raise confidence that the facility exists (capped, not a probability)
                 track_confidence=round(float(np.mean(confs)) * min(1.0, 0.5 + 0.1 * len(obs)), 4),
                 confirmed=len(obs) >= min_hits)
        out.append(T)
    out.sort(key=lambda x: (x["first_t"], x["track_id"]))
    n_det = sum(len([d for d in r.detections if d.class_name in classes]) for r in results)
    return {"method": method, "frames": len(results), "detections": n_det, "tracks": out,
            "confirmed_tracks": sum(t["confirmed"] for t in out),
            "per_class_confirmed": _count([t for t in out if t["confirmed"]])}


def _count(ts):
    c = {}
    for t in ts:
        c[t["class_name"]] = c.get(t["class_name"], 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))
