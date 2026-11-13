"""Threshold filtering and de-duplication of raw detections."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torchvision.ops import batched_nms

from vision.models.base import RawDetection


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N,4) and (M,4) xyxy boxes."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def intersection_over_first(a, b) -> float:
    """Fraction of box `a` covered by box `b`."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    return float(ix * iy / area)


def filter_by_threshold(dets: list[RawDetection], threshold_for: Callable[[RawDetection], float]) -> list[RawDetection]:
    return [d for d in dets if d.score >= threshold_for(d)]


def drop_degenerate(dets: list[RawDetection], min_size: float = 4.0) -> list[RawDetection]:
    return [d for d in dets if (d.xyxy[2] - d.xyxy[0]) >= min_size and (d.xyxy[3] - d.xyxy[1]) >= min_size]


def classwise_nms(dets: list[RawDetection], iou: float = 0.5) -> list[RawDetection]:
    """NMS within each canonical class. Different prompts of the same class
    ("stairs", "staircase") and different ensemble members collapse into one box;
    the highest-scoring box (and its prompt) wins."""
    if not dets:
        return []
    boxes = torch.tensor(np.stack([d.xyxy for d in dets]), dtype=torch.float32)
    scores = torch.tensor([d.score for d in dets], dtype=torch.float32)
    names = sorted({d.class_name for d in dets})
    idx = torch.tensor([names.index(d.class_name) for d in dets])
    keep = batched_nms(boxes, scores, idx, iou).tolist()
    return [dets[i] for i in keep]


def suppress_contained_duplicates(dets: list[RawDetection], containment: float = 0.9) -> list[RawDetection]:
    """Second pass within a class: drop a lower-scoring box that is almost entirely inside a
    higher-scoring box of the same class *and* nearly the same size (IoU NMS misses these when
    the boxes differ in aspect ratio). Genuinely nested distinct objects survive because of
    the size-ratio test."""
    dets = sorted(dets, key=lambda d: -d.score)
    kept: list[RawDetection] = []
    for d in dets:
        dup = False
        for k in kept:
            if k.class_name != d.class_name:
                continue
            area_d = (d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1])
            area_k = (k.xyxy[2] - k.xyxy[0]) * (k.xyxy[3] - k.xyxy[1])
            if intersection_over_first(d.xyxy, k.xyxy) >= containment and area_d / max(area_k, 1e-9) > 0.6:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def merge_fragments(dets: list[RawDetection], classes: set[str], gap_frac: float = 0.15) -> list[RawDetection]:
    """Union-merge boxes of the given classes that overlap or are vertically stacked with a small
    gap (e.g. a staircase detected as one box per step). The merged box takes the max score and
    the prompt of the best fragment. Other classes pass through unchanged."""
    out = [d for d in dets if d.class_name not in classes]
    for cls in classes:
        group = [d for d in dets if d.class_name == cls]
        if len(group) < 2:
            out.extend(group)
            continue
        n = len(group)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            a = group[i].xyxy
            for j in range(i + 1, n):
                b = group[j].xyxy
                ow = min(a[2], b[2]) - max(a[0], b[0])           # horizontal overlap
                gap = max(a[1], b[1]) - min(a[3], b[3])           # vertical gap (<0 = overlap)
                minw = min(a[2] - a[0], b[2] - b[0])
                minh = min(a[3] - a[1], b[3] - b[1])
                if ow > 0.5 * minw and gap < gap_frac * max(minh, 1.0):
                    parent[find(i)] = find(j)
        clusters: dict[int, list[RawDetection]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(group[i])
        for members in clusters.values():
            if len(members) == 1:
                out.append(members[0])
                continue
            best = max(members, key=lambda d: d.score)
            boxes = np.stack([m.xyxy for m in members])
            u = np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()])
            out.append(RawDetection(u, best.score, cls, best.prompt, best.detector))
    return out
