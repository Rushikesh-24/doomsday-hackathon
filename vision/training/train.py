"""Transfer-learning wrapper around Ultralytics with model versioning.

Each run writes `runs/train/<name>-v<N>/model_card.json` recording base weights (+sha256),
dataset manifest hash, hyper-parameters, device and the final validation/test metrics that
Ultralytics computed on the held-out splits. It never edits configs/models.yaml itself; it
prints the registry entry to add once you have reviewed the metrics.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from vision.config import get_settings
from vision.device import select_device
from vision.models.base import sha256_file
from vision.models.ultralytics_models import resolve_ultralytics_weights

log = logging.getLogger(__name__)


def next_version(runs_dir: Path, name: str) -> int:
    existing = [p.name for p in runs_dir.glob(f"{name}-v*")]
    nums = [int(n.rsplit("-v", 1)[1]) for n in existing if n.rsplit("-v", 1)[1].isdigit()]
    return max(nums, default=0) + 1


def train(data_yaml: str | Path, base: str = "yolo11s.pt", name: str = "campus", epochs: int = 50,
          imgsz: int = 640, batch: int = 16, patience: int = 15, freeze: int | None = None,
          runs_dir: str | Path = "runs/train", device: str | None = None, **extra) -> dict:
    from ultralytics import YOLO, YOLOWorld

    s = get_settings()
    data_yaml = Path(data_yaml).resolve()
    manifest_p = data_yaml.parent / "manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    runs_dir = Path(runs_dir).resolve()
    v = next_version(runs_dir, name)
    run_name = f"{name}-v{v}"
    dev = device or select_device(s.device)
    base_path = resolve_ultralytics_weights(base, s.cache_dir, s.offline, base)
    Model = YOLOWorld if "world" in base.lower() else YOLO
    model = Model(str(base_path))
    hp = dict(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, patience=patience, device=dev,
              project=str(runs_dir), name=run_name, exist_ok=False, seed=0, deterministic=True, **extra)
    if freeze is not None:
        hp["freeze"] = freeze
    log.info("Training %s from %s on %s", run_name, base, dev)
    model.train(**hp)
    run = runs_dir / run_name
    best = run / "weights" / "best.pt"
    metrics = {}
    names = yaml.safe_load(data_yaml.read_text())["names"]
    for split in ("val", "test"):
        try:
            r = YOLO(str(best)).val(data=str(data_yaml), split=split, imgsz=imgsz, device=dev, verbose=False)
            metrics[split] = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
                              "precision": float(r.box.mp), "recall": float(r.box.mr),
                              "per_class_mAP50_95": {names[int(c)]: float(m) for c, m in zip(r.box.ap_class_index, r.box.maps[r.box.ap_class_index])}}
        except Exception as e:  # e.g. empty test split
            metrics[split] = {"error": str(e)}
    card = {"name": run_name, "version": v, "created": datetime.now(timezone.utc).isoformat(),
            "base_weights": base, "base_sha256": sha256_file(base_path), "weights": str(best),
            "weights_sha256": sha256_file(best) if best.exists() else None,
            "dataset": {"data_yaml": str(data_yaml), "content_hash": manifest.get("content_hash"),
                        "classes": manifest.get("classes"), "sources": manifest.get("sources")},
            "hyperparameters": {k: v for k, v in hp.items() if k not in ("project",)}, "device": dev,
            "metrics": metrics}
    (run / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    entry = {f"{name}_v{v}": {"type": "yolo_coco", "weights": str(best), "version": f"{run_name} (fine-tuned from {base})",
                              "source": "local fine-tune", "license": "AGPL-3.0 (Ultralytics base) + dataset licences",
                              "conf": 0.25, "imgsz": imgsz, "open_vocabulary": False}}
    log.info("Review %s, then add to configs/models.yaml:\n%s", run / "model_card.json", yaml.safe_dump({"models": entry}))
    return card
