#!/usr/bin/env python
"""Fine-tune a detector on a prepared dataset (see scripts/prepare_dataset.py).
Only do this when the pretrained open-vocabulary models are insufficient for a class.

    python scripts/train.py datasets/campus_v1/data.yaml --base yolo11s.pt --epochs 50
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.training.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_yaml")
    ap.add_argument("--base", default="yolo11s.pt", help="yolo11{n,s,m}.pt or yolov8s-worldv2.pt")
    ap.add_argument("--name", default="campus")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze", type=int, default=None, help="freeze first N layers (transfer learning)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    card = train(a.data_yaml, base=a.base, name=a.name, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                 freeze=a.freeze, device=a.device)
    print(json.dumps(card["metrics"], indent=2))


if __name__ == "__main__":
    main()
