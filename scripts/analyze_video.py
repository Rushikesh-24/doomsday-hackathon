#!/usr/bin/env python
"""Analyze an MP4 video.

    python scripts/analyze_video.py samples/videos/access_city_logrono.mp4 --sample-fps 2
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.video.processor import analyze_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ("vision.inference.pipeline",):
        logging.getLogger(n).setLevel(logging.WARNING)
    r = analyze_video(a.video, model=a.model, conf=a.conf, sample_fps=a.sample_fps, max_frames=a.max_frames,
                      min_hits=a.min_hits, write_video=not a.no_video)
    print(f"\n{r.video.name}: {r.video.duration_s:.1f}s @ {r.video.fps}fps, {r.video.width}x{r.video.height}")
    print(f"processed {r.frames_processed} frames, skipped {r.frames_skipped_static} static; "
          f"{r.timing['per_processed_frame_ms']:.0f} ms/frame inference, {r.timing['total_s']}s total")
    print("unique facilities:", r.unique_counts_by_class)
    for t in r.unique_facilities:
        print(f"  #{t.track_id:<3} {t.class_name:<20} {t.first_seen_s:6.1f}s-{t.last_seen_s:6.1f}s  "
              f"obs={t.observations:<3} max_conf={t.max_confidence:.2f}")
    for w in r.warnings:
        print("  warning:", w)
    print("annotated video:", r.annotated_video)
    print("frames:", r.annotated_frames_dir)
    print("json:", r.json_path)


if __name__ == "__main__":
    main()
