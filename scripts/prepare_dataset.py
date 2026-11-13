#!/usr/bin/env python
"""Merge public datasets into one fine-tuning dataset in the canonical taxonomy.

Download sources yourself (respecting their licences), e.g. a Roboflow Universe project
exported as "YOLOv8" or a COCO-JSON dataset, then:

    python scripts/prepare_dataset.py --out datasets/campus_v1 \
        --source yolo:path/to/roboflow_stairs:rf_stairs:"CC BY 4.0" \
        --source coco:path/to/coco_ds:my_coco:"CC BY 4.0":_annotations.coco.json:images
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.config import get_settings  # noqa: E402
from vision.taxonomy import load_taxonomy  # noqa: E402
from vision.training.dataset import SourceSpec, build_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="format:path:name[:license[:coco_json[:images_dir]]]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--class-map", default="configs/dataset_class_map.yaml")
    ap.add_argument("--split", default="0.8,0.1,0.1")
    ap.add_argument("--classes", default=None, help="restrict to these canonical classes")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    specs = []
    for s in a.source:
        p = s.split(":")
        specs.append(SourceSpec(Path(p[1]), p[0], p[2], license=p[3] if len(p) > 3 else "unknown",
                                coco_json=p[4] if len(p) > 4 else None, images_dir=p[5] if len(p) > 5 else None))
    tax = load_taxonomy(str(get_settings().taxonomy_config))
    man = build_dataset(specs, Path(a.out), Path(a.class_map), tax.names(),
                        tuple(float(x) for x in a.split.split(",")),
                        classes=a.classes.split(",") if a.classes else None)
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
