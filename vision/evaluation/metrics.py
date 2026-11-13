"""Detection evaluation: statistics without ground truth, and precision/recall/AP with it.

Ground-truth format (JSON), boxes in absolute pixels xyxy:
    {"images": {"photo.jpg": [{"class": "staircase", "bbox": [x1, y1, x2, y2]}, ...]}}
YOLO txt label folders can be converted with `load_yolo_labels`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vision.postprocessing.nms import box_iou
from vision.schemas import ImageResult


# ------------------------------------------------------------------ no-GT statistics
def summarize_results(results: list[ImageResult], bins: int = 10) -> dict:
    counts: dict[str, int] = defaultdict(int)
    confs: dict[str, list[float]] = defaultdict(list)
    infer, total = [], []
    for r in results:
        infer.append(r.timing.inference_ms)
        total.append(r.timing.total_ms)
        for d in r.detections:
            counts[d.class_name] += 1
            confs[d.class_name].append(d.confidence)
    all_c = [c for v in confs.values() for c in v]
    edges = np.linspace(0, 1, bins + 1)
    hist = np.histogram(all_c, bins=edges)[0].tolist() if all_c else [0] * bins
    pct = lambda a, q: round(float(np.percentile(a, q)), 1) if a else None
    return {
        "images": len(results),
        "detections": int(sum(counts.values())),
        "counts_by_class": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "confidence_by_class": {k: {"n": len(v), "mean": round(float(np.mean(v)), 4), "min": round(min(v), 4),
                                    "max": round(max(v), 4)} for k, v in confs.items()},
        "confidence_histogram": {"bin_edges": [round(e, 2) for e in edges.tolist()], "counts": hist},
        "inference_ms": {"mean": round(float(np.mean(infer)), 1) if infer else None, "p50": pct(infer, 50),
                         "p95": pct(infer, 95)},
        "total_ms": {"mean": round(float(np.mean(total)), 1) if total else None},
    }


# ------------------------------------------------------------------ ground truth
def load_ground_truth(path: str | Path) -> dict[str, list[dict]]:
    data = json.loads(Path(path).read_text())
    return data["images"]


def load_yolo_labels(labels_dir: str | Path, images_dir: str | Path, class_names: list[str]) -> dict[str, list[dict]]:
    """Convert YOLO normalised cx,cy,w,h label files into the GT dict."""
    from PIL import Image
    gt = {}
    for img in Path(images_dir).iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        lbl = Path(labels_dir) / (img.stem + ".txt")
        W, H = Image.open(img).size
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                boxes.append({"class": class_names[c],
                              "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]})
        gt[img.name] = boxes
    return gt


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point interpolated AP (VOC2010+/COCO style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(predictions: dict[str, list[dict]], ground_truth: dict[str, list[dict]],
             iou_threshold: float = 0.5, classes: list[str] | None = None) -> dict:
    """predictions: {image_name: [{"class", "bbox", "confidence"}]}. Only images present in the
    ground truth are scored. Returns per-class TP/FP/FN, precision, recall, F1 and AP@iou."""
    classes = classes or sorted({b["class"] for v in ground_truth.values() for b in v})
    per_class = {}
    for c in classes:
        scored = []  # (confidence, is_tp)
        n_gt = 0
        for img, gts in ground_truth.items():
            g = np.array([b["bbox"] for b in gts if b["class"] == c]).reshape(-1, 4)
            n_gt += len(g)
            preds = sorted([p for p in predictions.get(img, []) if p["class"] == c], key=lambda p: -p["confidence"])
            used = np.zeros(len(g), bool)
            for p in preds:
                if len(g):
                    ious = box_iou(np.array([p["bbox"]]), g)[0]
                    ious[used] = -1
                    j = int(np.argmax(ious))
                    if ious[j] >= iou_threshold:
                        used[j] = True
                        scored.append((p["confidence"], 1))
                        continue
                scored.append((p["confidence"], 0))
        scored.sort(key=lambda t: -t[0])
        tp = np.cumsum([s[1] for s in scored]) if scored else np.array([])
        fp = np.cumsum([1 - s[1] for s in scored]) if scored else np.array([])
        TP = int(tp[-1]) if len(tp) else 0
        FP = int(fp[-1]) if len(fp) else 0
        prec = TP / max(TP + FP, 1)
        rec = TP / n_gt if n_gt else None
        ap = _ap(tp / n_gt, tp / np.maximum(tp + fp, 1e-9)) if n_gt and len(tp) else (0.0 if n_gt else None)
        per_class[c] = {"gt": n_gt, "tp": TP, "fp": FP, "fn": n_gt - TP, "precision": round(prec, 4),
                        "recall": None if rec is None else round(rec, 4),
                        "f1": round(2 * prec * rec / (prec + rec), 4) if rec and (prec + rec) else 0.0,
                        f"ap@{iou_threshold}": None if ap is None else round(ap, 4)}
    valid = [v[f"ap@{iou_threshold}"] for v in per_class.values() if v[f"ap@{iou_threshold}"] is not None]
    return {"iou_threshold": iou_threshold, "images_scored": len(ground_truth), "per_class": per_class,
            f"mAP@{iou_threshold}": round(float(np.mean(valid)), 4) if valid else None}


def predictions_from_results(results: list[ImageResult]) -> dict[str, list[dict]]:
    return {r.image.name: [{"class": d.class_name, "bbox": d.bbox.xyxy(), "confidence": d.confidence}
                           for d in r.detections] for r in results}
