"""Video ingestion & capture-quality analysis for 3D reconstruction.

Per analysed frame (dense sampling, low resolution): sharpness (variance of Laplacian, normalised by
the video's own distribution and an absolute floor), exposure (mean, clipped fractions), global camera
motion between consecutive analysed frames (sparse LK optical flow + RANSAC similarity), overlap (fraction
of tracked features that survive), scene cuts (HSV histogram distance), near-duplicates (low motion +
tiny pixel change) and revisits (perceptual-hash similarity to non-adjacent frames).
Thresholds are engineering defaults (not learned) and are reported in the output.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".m4v"}


class VideoError(ValueError):
    pass


def probe(path: Path) -> dict:
    """Container/stream metadata. ffprobe when available (rotation, codec, creation time, GPS tag)."""
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoError(f"{path.name}: empty or missing file")
    if path.suffix.lower() not in VIDEO_EXTS:
        raise VideoError(f"{path.name}: unsupported extension (MP4/MOV only)")
    head = path.read_bytes()[:16] if path.stat().st_size < 1e9 else open(path, "rb").read(16)
    if b"ftyp" not in head and b"moov" not in head and b"wide" not in head:
        raise VideoError(f"{path.name}: not an ISO-BMFF (MP4/MOV) container")
    meta = {"file": path.name, "size_bytes": path.stat().st_size}
    ff = shutil.which("ffprobe")
    if ff:
        r = subprocess.run([ff, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            j = json.loads(r.stdout)
            vs = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), None)
            if vs is None:
                raise VideoError(f"{path.name}: no video stream")
            num, den = (vs.get("avg_frame_rate") or "0/1").split("/")
            fps = float(num) / float(den) if float(den) else 0.0
            rot = 0
            for sd in vs.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = int(sd["rotation"])
            rot = int(vs.get("tags", {}).get("rotate", rot))
            tags = j.get("format", {}).get("tags", {})
            meta.update(codec=vs.get("codec_name"), width=vs.get("width"), height=vs.get("height"), fps=round(fps, 3),
                        duration_s=float(j.get("format", {}).get("duration") or vs.get("duration") or 0),
                        frame_count=int(vs.get("nb_frames") or 0), rotation_deg=rot,
                        creation_time=tags.get("creation_time"),
                        location=tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709"),
                        device=tags.get("com.apple.quicktime.model") or tags.get("model"),
                        bit_rate=int(j.get("format", {}).get("bit_rate") or 0))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"{path.name}: cannot be decoded")
    ok, fr = cap.read()
    if not ok:
        raise VideoError(f"{path.name}: no decodable frames")
    meta.setdefault("fps", cap.get(cv2.CAP_PROP_FPS))
    meta.setdefault("frame_count", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if not meta.get("frame_count"):
        meta["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta["decoded_width"], meta["decoded_height"] = fr.shape[1], fr.shape[0]   # OpenCV applies rotation
    if not meta.get("duration_s") and meta.get("fps"):
        meta["duration_s"] = meta["frame_count"] / meta["fps"]
    cap.release()
    if not meta.get("fps") or meta["fps"] <= 0:
        raise VideoError(f"{path.name}: invalid frame rate")
    return meta


@dataclass
class FrameStat:
    index: int
    t: float
    sharpness: float
    brightness: float
    dark_frac: float
    bright_frac: float
    motion_px: float | None = None        # median feature displacement since previous analysed frame (px @ 480p)
    rotation_deg: float | None = None     # in-plane rotation since previous frame
    overlap: float | None = None          # fraction of tracked features surviving
    hist_dist: float | None = None
    scene_cut: bool = False
    near_duplicate: bool = False
    blurry: bool = False
    exposure_issue: str | None = None


@dataclass
class QualityReport:
    video: dict
    analysis_fps: float
    frames_analysed: int
    stats: dict
    issues: list[dict]
    recommendations: list[str]
    suitability_score: float
    suitability: str
    thresholds: dict
    frames: list[dict] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)


THRESH = dict(blur_rel=0.35, blur_abs=25.0, dark_mean=45, bright_mean=215, clip_frac=0.25, fast_motion_px=45.0,
              shake_rot_deg=4.0, min_overlap=0.35, scene_cut_hist=0.55, dup_motion_px=0.6, revisit_hash_bits=6,
              min_duration_s=4.0, min_parallax_frames=8)


def detect_letterbox(path: Path, samples: int = 12) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the active picture if constant black bars exist in every sampled frame."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    acc = None
    for i in np.linspace(0, max(n - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        acc = g if acc is None else np.maximum(acc, g)      # max over time: bars stay black
    cap.release()
    if acc is None:
        return None
    rows = np.nonzero(acc.mean(1) > 12)[0]
    cols = np.nonzero(acc.mean(0) > 12)[0]
    if not len(rows) or not len(cols):
        return None
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    H, W = acc.shape
    if (y1 - y0) > 0.97 * H and (x1 - x0) > 0.97 * W:
        return None
    return (x0, y0, x1, y1)


def _dhash(gray: np.ndarray) -> int:
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def analyze_quality(path: Path, analysis_fps: float = 6.0, progress=None, max_frames: int = 3000) -> QualityReport:
    meta = probe(path)
    crop = detect_letterbox(path)
    meta["letterbox_active_area"] = list(crop) if crop else None
    cap = cv2.VideoCapture(str(path))
    fps = float(meta["fps"])
    step = max(1, int(round(fps / analysis_fps)))
    prev_gray, prev_pts, prev_hist = None, None, None
    frames: list[FrameStat] = []
    hashes: list[tuple[int, int]] = []
    idx = -1
    n_total = meta.get("frame_count") or 0
    while len(frames) < max_frames:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        ok, bgr = cap.retrieve()
        if not ok:
            break
        if crop:
            bgr = bgr[crop[1]:crop[3], crop[0]:crop[2]]
        h, w = bgr.shape[:2]
        s = 480 / max(h, w)
        small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fs = FrameStat(index=idx, t=round(idx / fps, 3), sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                       brightness=float(gray.mean()), dark_frac=float((gray < 16).mean()),
                       bright_frac=float((gray > 245).mean()))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_gray is not None:
            fs.hist_dist = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            if prev_pts is not None and len(prev_pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=4)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, winSize=(21, 21), maxLevel=4)
                fb = np.linalg.norm((back - prev_pts).reshape(-1, 2), axis=1)
                good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)     # forward-backward consistent
                fs.overlap = float(good.mean())
                if good.sum() >= 8:
                    p0, p1 = prev_pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
                    fs.motion_px = float(np.median(np.linalg.norm(p1 - p0, axis=1)))
                    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3)
                    if M is not None and inl is not None and inl.mean() > 0.5:
                        fs.rotation_deg = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
            else:
                fs.overlap = 0.0
        prev_pts = cv2.goodFeaturesToTrack(gray, 400, 0.01, 8)
        prev_gray, prev_hist = gray, hist
        hashes.append((len(frames), _dhash(gray)))
        frames.append(fs)
        if progress and n_total and len(frames) % 20 == 0:
            progress(min(0.99, idx / n_total), f"analysed {len(frames)} frames")
    cap.release()
    if len(frames) < 3:
        raise VideoError(f"{path.name}: only {len(frames)} frames could be analysed")
    T = THRESH
    sharp = np.array([f.sharpness for f in frames])
    med_sharp = float(np.median(sharp))
    for f in frames:
        f.blurry = f.sharpness < max(T["blur_abs"], T["blur_rel"] * med_sharp)
        if f.brightness < T["dark_mean"] or f.dark_frac > T["clip_frac"]:
            f.exposure_issue = "underexposed"
        elif f.brightness > T["bright_mean"] or f.bright_frac > T["clip_frac"]:
            f.exposure_issue = "overexposed"
        f.scene_cut = bool(f.hist_dist is not None and f.hist_dist > T["scene_cut_hist"]
                           and (f.overlap is None or f.overlap < 0.2))   # real cut: appearance AND tracking break
        f.near_duplicate = bool(f.motion_px is not None and f.motion_px < T["dup_motion_px"])
    # revisits: similar perceptual hash to a frame more than 3 s earlier
    revisits = []
    min_gap = int(3 * analysis_fps)
    for i, (fi, hi) in enumerate(hashes):
        for fj, hj in hashes[: max(0, i - min_gap)]:
            if bin(hi ^ hj).count("1") <= T["revisit_hash_bits"] and not frames[fi].exposure_issue:
                revisits.append((frames[fj].t, frames[fi].t))
                break
    motion = np.array([f.motion_px for f in frames if f.motion_px is not None] or [0.0])
    rot = np.array([abs(f.rotation_deg) for f in frames if f.rotation_deg is not None] or [0.0])
    ovl = np.array([f.overlap for f in frames if f.overlap is not None] or [0.0])
    # shake = high-frequency part of the rotation signal (frame-to-frame jitter around a smooth path)
    rsig = np.array([f.rotation_deg or 0.0 for f in frames if not f.scene_cut])
    shake = float(np.std(rsig - np.convolve(rsig, np.ones(5) / 5, mode="same"))) if len(rsig) > 6 else 0.0
    n = len(frames)
    frac = lambda cond: round(sum(1 for f in frames if cond(f)) / n, 3)
    stats = {
        "blurry_fraction": frac(lambda f: f.blurry),
        "underexposed_fraction": frac(lambda f: f.exposure_issue == "underexposed"),
        "overexposed_fraction": frac(lambda f: f.exposure_issue == "overexposed"),
        "near_duplicate_fraction": frac(lambda f: f.near_duplicate),
        "scene_cuts": sum(f.scene_cut for f in frames),
        "fast_motion_fraction": round(float((motion > T["fast_motion_px"]).mean()), 3),
        "low_overlap_fraction": round(float((ovl < T["min_overlap"]).mean()), 3),
        "median_motion_px_per_analysed_frame": round(float(np.median(motion)), 2),
        "p90_motion_px": round(float(np.percentile(motion, 90)), 2),
        "median_overlap": round(float(np.median(ovl)), 3),
        "shake_rotation_jitter_deg": round(shake, 3),
        "p90_abs_rotation_deg": round(float(np.percentile(rot, 90)), 2),
        "median_sharpness": round(med_sharp, 1),
        "revisits": len(revisits),
        "total_translation_proxy_px": round(float(motion.sum()), 1),
    }
    issues, recs = [], []

    def issue(code, sev, msg, rec):
        issues.append({"code": code, "severity": sev, "message": msg})
        if rec not in recs:
            recs.append(rec)
    dur = float(meta.get("duration_s") or n / analysis_fps)
    if dur < T["min_duration_s"]:
        issue("TOO_SHORT", "error", f"video is {dur:.1f}s long", "Record at least 15-30 s while walking slowly.")
    if stats["blurry_fraction"] > 0.3:
        issue("BLURRY_FRAMES", "warning", f"{stats['blurry_fraction']:.0%} of frames are blurry",
              "Move slower, keep the phone steady, and record in good light (motion blur kills feature matching).")
    if stats["underexposed_fraction"] > 0.25:
        issue("POOR_LIGHTING", "warning", f"{stats['underexposed_fraction']:.0%} of frames underexposed",
              "Turn on lights or record in daylight; avoid pointing into dark areas.")
    if stats["overexposed_fraction"] > 0.25:
        issue("OVEREXPOSED", "warning", f"{stats['overexposed_fraction']:.0%} of frames overexposed",
              "Avoid pointing the camera at windows/sun; lock exposure.")
    if stats["fast_motion_fraction"] > 0.2:
        issue("FAST_MOTION", "warning", f"{stats['fast_motion_fraction']:.0%} of frame steps move > {T['fast_motion_px']} px",
              "Walk slower and turn gradually (no quick pans).")
    if stats["shake_rotation_jitter_deg"] > T["shake_rot_deg"] / 4:
        issue("CAMERA_SHAKE", "warning", f"rotation jitter {stats['shake_rotation_jitter_deg']:.2f} deg between frames",
              "Hold the phone with both hands or use stabilisation; avoid bouncing steps.")
    if stats["low_overlap_fraction"] > 0.2:
        issue("LOW_OVERLAP", "warning", f"{stats['low_overlap_fraction']:.0%} of frame steps keep < {T['min_overlap']:.0%} features",
              "Keep consecutive views overlapping by >60%: slower movement, no sudden turns.")
    if stats["near_duplicate_fraction"] > 0.5:
        issue("LITTLE_MOVEMENT", "warning", f"{stats['near_duplicate_fraction']:.0%} of frames barely move",
              "Move the camera through space (walk), not only rotate on the spot - 3D needs parallax.")
    if stats["scene_cuts"] > 0:
        issue("SCENE_CUTS", "warning", f"{stats['scene_cuts']} abrupt scene changes",
              "Record one continuous take; edited videos break reconstruction into pieces.")
    if stats["revisits"] > 0:
        issue("REPEATED_VIEWS", "info", f"{stats['revisits']} frames revisit earlier views",
              "Revisits are useful for loop closure; avoid long repeated passes that add no new coverage.")
    # suitability score (0-100): weighted penalties, transparent
    pen = (35 * stats["blurry_fraction"] + 45 * stats["low_overlap_fraction"] + 20 * stats["fast_motion_fraction"]
           + 20 * max(stats["underexposed_fraction"], stats["overexposed_fraction"])
           + 15 * min(1.0, stats["near_duplicate_fraction"] / 0.8) + 4 * min(stats["scene_cuts"], 5)
           + 10 * min(1.0, stats["shake_rotation_jitter_deg"] / T["shake_rot_deg"]) + (25 if dur < T["min_duration_s"] else 0))
    score = round(max(0.0, 100 - pen), 1)
    label = "good" if score >= 70 else "fair" if score >= 45 else "poor"
    if not recs:
        recs.append("Capture looks suitable; keep the same technique.")
    # continuous segments (split at scene cuts / lost tracking)
    segs, start = [], 0
    for i, f in enumerate(frames):
        if i > 0 and (f.scene_cut or (f.overlap is not None and f.overlap < 0.1)):
            segs.append({"start_t": frames[start].t, "end_t": frames[i - 1].t, "frames": i - start})
            start = i
    segs.append({"start_t": frames[start].t, "end_t": frames[-1].t, "frames": n - start})
    return QualityReport(video=meta, analysis_fps=analysis_fps, frames_analysed=n, stats=stats, issues=issues,
                         recommendations=recs, suitability_score=score, suitability=label, thresholds=T,
                         frames=[asdict(f) for f in frames], segments=segs)
