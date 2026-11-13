"""Video analysis: smart frame sampling, per-frame detection, tracking, summary,
annotated frames and an annotated (H.264 when ffmpeg is available) video."""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from vision.inference.pipeline import VisionPipeline, get_pipeline, now_iso
from vision.schemas import SourceRef, TrackSummary, VideoInfo, VideoResult
from vision.tracking.tracker import IoUTracker
from vision.visualization.draw import annotate

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".m4v", ".mov"}


class InvalidVideoError(ValueError):
    pass


def probe_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    if not path.is_file() or path.stat().st_size == 0:
        raise InvalidVideoError(f"{path.name}: empty or missing file")
    with open(path, "rb") as f:
        head = f.read(12)
    if path.suffix.lower() not in VIDEO_EXTS or b"ftyp" not in head:
        raise InvalidVideoError(f"{path.name}: not an MP4/MOV container (expected 'ftyp' box)")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InvalidVideoError(f"{path.name}: cannot be decoded")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise InvalidVideoError(f"{path.name}: no decodable frames")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if fps <= 0 or fps > 240:
        fps = 30.0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or None
    return cap, VideoInfo(name=path.name, width=frame.shape[1], height=frame.shape[0], fps=round(fps, 3),
                          frame_count=n, duration_s=round(n / fps, 3) if n else 0.0, codec=codec)


def _thumb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(bgr, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _transcode_h264(src: Path, dst: Path) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(dst)],
                       capture_output=True, timeout=600)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def analyze_video(path: str | Path, *, model: str | None = None, conf: float | None = None,
                  classes: list[str] | None = None, sample_fps: float = 2.0, static_threshold: float = 0.01,
                  max_static_gap_s: float = 3.0, max_frames: int = 600, min_hits: int = 2,
                  save_frames: bool = True, write_video: bool = True, output_root: Path | None = None,
                  pipeline: VisionPipeline | None = None, request_id: str | None = None) -> VideoResult:
    """Sampling strategy:
      1. temporal subsampling at `sample_fps` (default 2 fps) instead of every frame;
      2. among those, skip frames where fewer than `static_threshold` (fraction) of the 96x54 grey
         thumbnail pixels changed by more than 12 levels vs the last *processed* frame (camera
         standing still), but never go longer than `max_static_gap_s` without processing a frame.
    """
    pipe = pipeline or get_pipeline()
    s = pipe.settings
    path = Path(path)
    if path.stat().st_size > s.max_video_mb * 1e6:
        raise InvalidVideoError(f"{path.name}: larger than {s.max_video_mb} MB")
    cap, info = probe_video(path)
    if info.duration_s > s.max_video_seconds:
        cap.release()
        raise InvalidVideoError(f"{path.name}: {info.duration_s:.0f}s exceeds limit {s.max_video_seconds:.0f}s")
    rid = request_id or uuid.uuid4().hex[:12]
    out_dir = (output_root or s.output_dir / "videos") / rid
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    key = model or pipe.registry.default_key
    det = pipe.registry.get(key)  # load before timing
    tracker = IoUTracker(min_hits=min_hits, max_age=max(2, int(round(sample_fps * 1.5))))

    step = max(1, int(round(info.fps / max(sample_fps, 0.01))))
    writer = None
    raw_video = out_dir / "annotated_raw.mp4"
    if write_video:
        scale = min(1.0, 1280 / max(info.width, info.height))
        vw, vh = int(info.width * scale) // 2 * 2, int(info.height * scale) // 2 * 2
        writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (vw, vh))
        if not writer.isOpened():
            writer = None
    t0 = time.perf_counter()
    frame_results = []
    last_thumb, last_proc_t = None, -1e9
    current = []          # detections drawn on in-between frames
    skipped_static = 0
    idx = -1
    infer_ms = 0.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        ts = idx / info.fps
        if idx % step == 0 and len(frame_results) < max_frames:
            th = _thumb(bgr)
            static = last_thumb is not None and float((np.abs(th - last_thumb) > 12).mean()) < static_threshold \
                and (ts - last_proc_t) < max_static_gap_s
            if static:
                skipped_static += 1
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                r = pipe.analyze_array(rgb, f"{path.stem}_f{idx:06d}.jpg", model=key, conf=conf, classes=classes,
                                       source=SourceRef(name=path.name, kind="video_frame", frame_index=idx,
                                                        timestamp_s=round(ts, 3)),
                                       save_dir=None, save_annotated=False, save_json=False,
                                       request_id=f"{rid}-f{idx:06d}")
                infer_ms += r.timing.inference_ms
                tracker.update(r.detections, idx, ts)
                frame_results.append(r)
                current = r.detections
                last_thumb, last_proc_t = th, ts
                if save_frames:
                    ann = annotate(rgb, current, pipe.taxonomy, header=f"frame {idx}  t={ts:.1f}s  {key}")
                    p = frames_dir / f"frame_{idx:06d}.jpg"
                    cv2.imwrite(str(p), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    r.annotated_image = str(p)
        if writer is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ann = annotate(rgb, current, pipe.taxonomy, header=f"t={ts:.1f}s  tracks={len(tracker.confirmed_tracks())}")
            if ann.shape[1] != vw or ann.shape[0] != vh:
                ann = cv2.resize(ann, (vw, vh), interpolation=cv2.INTER_AREA)
            writer.write(ann)
    cap.release()
    warnings = []
    if info.frame_count and idx + 1 < info.frame_count * 0.9:
        warnings.append(f"decoded {idx + 1} of {info.frame_count} reported frames")
    annotated_video = None
    if writer is not None:
        writer.release()
        h264 = out_dir / "annotated.mp4"
        if _transcode_h264(raw_video, h264):
            raw_video.unlink(missing_ok=True)
            annotated_video = str(h264)
        else:
            annotated_video = str(raw_video)
            warnings.append("ffmpeg not available; annotated video is MPEG-4 Part 2 (may not play in browsers)")

    summaries = []
    for tr in sorted(tracker.confirmed_tracks(), key=lambda t: t.observations[0][0]):
        best = max(tr.observations, key=lambda o: o[2].confidence)
        confs = [o[2].confidence for o in tr.observations]
        summaries.append(TrackSummary(
            track_id=tr.track_id, class_name=tr.class_name, category=tr.category,
            first_seen_frame=tr.observations[0][0], last_seen_frame=tr.observations[-1][0],
            first_seen_s=round(tr.observations[0][1], 3), last_seen_s=round(tr.observations[-1][1], 3),
            observations=len(tr.observations), max_confidence=round(max(confs), 4),
            mean_confidence=round(float(np.mean(confs)), 4), best_frame_index=best[0], best_bbox=best[2].bbox,
            attributes=best[2].attributes))
    counts: dict[str, int] = {}
    for t in summaries:
        counts[t.class_name] = counts.get(t.class_name, 0) + 1
    unconfirmed = len(tracker.all_tracks()) - len(summaries)
    if unconfirmed:
        warnings.append(f"{unconfirmed} single-observation tracks not counted as unique facilities (min_hits={min_hits})")
    res = VideoResult(
        request_id=rid, video=info, model=det.info(),
        sampling={"strategy": "fixed-rate + static-scene skip", "sample_fps": sample_fps, "frame_step": step,
                  "static_threshold": static_threshold, "max_static_gap_s": max_static_gap_s,
                  "max_frames": max_frames, "tracker": {"type": "IoU+centre greedy", "min_hits": min_hits,
                                                        "max_age": tracker.max_age}},
        frames_processed=len(frame_results), frames_skipped_static=skipped_static,
        unique_facilities=summaries, unique_counts_by_class=counts, frame_results=frame_results,
        timing={"total_s": round(time.perf_counter() - t0, 2), "inference_s": round(infer_ms / 1000, 2),
                "per_processed_frame_ms": round(infer_ms / max(len(frame_results), 1), 1)},
        processed_at=now_iso(), annotated_video=annotated_video,
        annotated_frames_dir=str(frames_dir) if save_frames else None, warnings=warnings)
    jp = out_dir / "result.json"
    res.json_path = str(jp)
    jp.write_text(res.model_dump_json(indent=2))
    log.info("video %s: %d frames decoded, %d processed, %d static-skipped, %d unique facilities in %.1fs",
             path.name, idx + 1, len(frame_results), skipped_static, len(summaries), res.timing["total_s"])
    return res
