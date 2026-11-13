#!/usr/bin/env python
"""Benchmark registered models on the same images.

    python scripts/benchmark_models.py --images samples/images \
        --models yolo_world_l grounding_dino_tiny owlv2_base campus_ensemble [--gt gt.json]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.evaluation.benchmark import run_benchmark  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="samples/images")
    ap.add_argument("--models", nargs="+", default=["yolo_world_s", "yolo_world_l", "grounding_dino_tiny",
                                                    "owlv2_base", "campus_ensemble"])
    ap.add_argument("--out", default="outputs/benchmark")
    ap.add_argument("--gt", default=None, help="ground-truth JSON (see vision/evaluation/metrics.py)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    rep = run_benchmark(imgs, a.models, Path(a.out), Path(a.gt) if a.gt else None)
    print((Path(a.out) / "benchmark.md").read_text())


if __name__ == "__main__":
    main()
