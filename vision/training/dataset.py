"""Dataset preparation for fine-tuning: read YOLO (Roboflow "YOLOv8" export) or COCO-JSON
datasets, remap their labels into the canonical taxonomy, merge multiple sources,
de-duplicate identical images, split deterministically into train/val/test and write an
Ultralytics-ready dataset with a manifest (class counts, sources, licences, content hash)."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

log = logging.getLogger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image: Path
    boxes: list[tuple[str, float, float, float, float]]  # (canonical_class, cx, cy, w, h) normalised
    source: str
    sha1: str = ""


@dataclass
class SourceSpec:
    path: Path
    format: str              # "yolo" | "coco"
    name: str
    license: str = "unknown"
    url: str | None = None
    coco_json: str | None = None   # for coco: annotation file relative to path
    images_dir: str | None = None  # for coco
    report: dict = field(default_factory=dict)


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("-", " ").replace("_", " ").split())


def load_class_map(path: str | Path) -> dict[str, str | None]:
    data = yaml.safe_load(Path(path).read_text())["mapping"]
    return {_norm(k): v for k, v in data.items()}


def _map(label: str, cmap: dict[str, str | None], canonical: set[str], unknown: Counter) -> str | None:
    n = _norm(label)
    if n in cmap:
        return cmap[n]
    if n.replace(" ", "_") in canonical:
        return n.replace(" ", "_")
    unknown[label] += 1
    return None


def _read_yolo(spec: SourceSpec, cmap, canonical, unknown) -> list[Sample]:
    root = spec.path
    dy = next((p for p in [root / "data.yaml", root / "dataset.yaml"] if p.exists()), None)
    if dy is None:
        raise FileNotFoundError(f"{root}: no data.yaml")
    meta = yaml.safe_load(dy.read_text())
    names = meta["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    out = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS or "images" not in img.parts:
            continue
        parts = list(img.parts)
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) != 5:  # skip polygons/segments
                    continue
                c = _map(names[int(p[0])], cmap, canonical, unknown)
                if c:
                    boxes.append((c, *map(float, p[1:])))
        out.append(Sample(img, boxes, spec.name))
    return out


def _read_coco(spec: SourceSpec, cmap, canonical, unknown) -> list[Sample]:
    ann = json.loads((spec.path / (spec.coco_json or "annotations.json")).read_text())
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {i["id"]: i for i in ann["images"]}
    by_img = defaultdict(list)
    for a in ann["annotations"]:
        if a.get("iscrowd"):
            continue
        by_img[a["image_id"]].append(a)
    idir = spec.path / (spec.images_dir or "")
    out = []
    for iid, info in imgs.items():
        W, H = info["width"], info["height"]
        boxes = []
        for a in by_img.get(iid, []):
            c = _map(cats[a["category_id"]], cmap, canonical, unknown)
            x, y, w, h = a["bbox"]
            if c and w > 0 and h > 0:
                boxes.append((c, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        out.append(Sample(idir / info["file_name"], boxes, spec.name))
    return out


def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def _split_of(sha1: str, splits: tuple[float, float, float]) -> str:
    """Deterministic split from content hash -> identical images always land in the same split."""
    u = int(sha1[:8], 16) / 0xFFFFFFFF
    return "train" if u < splits[0] else ("val" if u < splits[0] + splits[1] else "test")


def build_dataset(sources: list[SourceSpec], out_dir: Path, class_map_path: Path, taxonomy_names: list[str],
                  splits=(0.8, 0.1, 0.1), keep_empty: bool = True, classes: list[str] | None = None) -> dict:
    assert abs(sum(splits) - 1) < 1e-6
    cmap = load_class_map(class_map_path)
    canonical = set(taxonomy_names)
    samples: list[Sample] = []
    for s in sources:
        unknown = Counter()
        items = (_read_yolo if s.format == "yolo" else _read_coco)(s, cmap, canonical, unknown)
        s.report = {"images": len(items), "unmapped_labels": dict(unknown)}
        log.info("%s: %d images, unmapped labels: %s", s.name, len(items), dict(unknown))
        samples.extend(items)
    used = sorted({b[0] for smp in samples for b in smp.boxes} if classes is None else classes)
    idx = {c: i for i, c in enumerate(used)}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    seen: set[str] = set()
    counts = {k: Counter() for k in ("train", "val", "test")}
    n_img = Counter()
    dup = 0
    for smp in samples:
        if not smp.image.exists():
            continue
        try:
            Image.open(smp.image).verify()
        except Exception:
            log.warning("skipping unreadable image %s", smp.image)
            continue
        smp.sha1 = _sha1(smp.image)
        if smp.sha1 in seen:
            dup += 1
            continue
        seen.add(smp.sha1)
        boxes = [b for b in smp.boxes if b[0] in idx]
        if not boxes and not keep_empty:
            continue
        sp = _split_of(smp.sha1, splits)
        (out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
        stem = f"{smp.source}_{smp.sha1[:12]}"
        shutil.copy2(smp.image, out_dir / "images" / sp / (stem + smp.image.suffix.lower()))
        (out_dir / "labels" / sp / (stem + ".txt")).write_text(
            "".join(f"{idx[c]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes))
        n_img[sp] += 1
        for b in boxes:
            counts[sp][b[0]] += 1
    data_yaml = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
                 "test": "images/test", "names": {i: c for c, i in idx.items()}}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    content_hash = hashlib.sha1("".join(sorted(seen)).encode()).hexdigest()
    manifest = {"classes": used, "images_per_split": dict(n_img), "boxes_per_split": {k: dict(v) for k, v in counts.items()},
                "duplicates_removed": dup, "content_hash": content_hash, "splits": splits,
                "sources": [{"name": s.name, "path": str(s.path), "format": s.format, "license": s.license,
                             "url": s.url, **s.report} for s in sources]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
