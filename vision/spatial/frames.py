"""Keyframe selection & extraction for reconstruction.

Walk the analysed frames in order; open a new keyframe window once the camera has moved enough
(accumulated feature displacement >= target parallax) or overlap since the last keyframe drops; inside
the window keep the sharpest well-exposed frame. Near-duplicates (no motion) are skipped, a frame is
forced after max_gap_s even if imperfect (flagged), and the number of keyframes is capped per preset.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# SfM on hand-held walkthroughs needs dense, overlapping frames (a 1.8 fps selection registered only 18% of
# frames on the real stairway video; ~6 fps registered ~3x more) - budgets are sized accordingly.
PRESET_KEYFRAMES = {"fast": 160, "balanced": 260, "high": 400}


def select_keyframes(frames: list[dict], target_motion_px: float = 10.0, min_overlap: float = 0.7,
                     max_gap_s: float = 0.5, max_keyframes: int = 160) -> list[dict]:
    if not frames:
        return []

    def run(target, min_overlap):
        picks, window, acc_motion, acc_overlap = [], [], 0.0, 1.0
        last_t = -1e9
        for f in frames:
            if f.get("scene_cut") and window:
                picks.append(_best(window, forced=False)); window, acc_motion, acc_overlap = [], 0.0, 1.0
            acc_motion += f.get("motion_px") or 0.0
            acc_overlap *= f.get("overlap") if f.get("overlap") is not None else 1.0
            window.append(f)
            due = acc_motion >= target or acc_overlap < min_overlap
            forced = f["t"] - last_t >= max_gap_s and acc_motion > 2.0
            if due or forced or not picks:
                if not picks and not due:   # first keyframe: first decent frame
                    good = [w for w in window if not w.get("blurry") and not w.get("exposure_issue")]
                    if not good and len(window) < 5:
                        continue
                k = _best(window, forced=forced and not due)
                picks.append(k)
                last_t = k["t"]
                window, acc_motion, acc_overlap = [], 0.0, 1.0
        return picks
    t, mo = target_motion_px, min_overlap
    picks = run(t, mo)
    while len(picks) > max_keyframes and t < 400:
        t *= 1.25
        mo *= 0.85
        picks = run(t, mo)
    # de-duplicate indices
    seen, out = set(), []
    for p in picks:
        if p["index"] not in seen:
            seen.add(p["index"])
            out.append(p)
    if len(out) > max_keyframes:   # hard cap: forced-gap picks can exceed it; keep an even temporal spread
        keep = np.unique(np.round(np.linspace(0, len(out) - 1, max_keyframes)).astype(int))
        out = [out[i] for i in keep]
    return out


def _best(window: list[dict], forced: bool) -> dict:
    good = [w for w in window if not w.get("blurry") and not w.get("exposure_issue") and not w.get("near_duplicate")]
    pool = good or window
    b = dict(max(pool, key=lambda w: w["sharpness"]))
    b["forced_imperfect"] = not good
    b["selection"] = "forced_gap" if forced else "parallax"
    return b


def extract_frames(video: Path, picks: list[dict], out_dir: Path, max_side: int = 1600,
                   crop: list[int] | None = None) -> list[dict]:
    if out_dir.exists():                       # re-extraction must not leave stale frames behind
        for old in out_dir.glob("*.jpg"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {p["index"]: p for p in picks}
    cap = cv2.VideoCapture(str(video))
    idx = -1
    rows = []
    last = max(want) if want else -1
    while idx < last:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx not in want:
            continue
        ok, bgr = cap.retrieve()
        if not ok:
            continue
        if crop:   # remove letterbox bars: they carry no geometry and would be learnt as black
            bgr = bgr[crop[1]:crop[3], crop[0]:crop[2]]
        h, w = bgr.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1:
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        name = f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        p = want[idx]
        rows.append({"name": name, "frame_index": idx, "t": p["t"], "sharpness": round(p["sharpness"], 1),
                     "blurry": p.get("blurry", False), "exposure_issue": p.get("exposure_issue"),
                     "forced_imperfect": p.get("forced_imperfect", False), "selection": p.get("selection"),
                     "width": bgr.shape[1], "height": bgr.shape[0]})
    cap.release()
    (out_dir.parent / "frames.json").write_text(json.dumps(rows, indent=2))
    return rows
