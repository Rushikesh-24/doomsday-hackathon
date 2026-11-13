"""Benchmark several registered models on the same images. Reports real measurements only:
latency (warm, excluding the first call), load time, process memory, model size on disk,
per-class detection counts / confidences, and P/R/AP when ground truth is supplied."""
from __future__ import annotations

import gc
import json
import logging
import os
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from vision.evaluation.metrics import evaluate, predictions_from_results, summarize_results
from vision.inference.pipeline import VisionPipeline

log = logging.getLogger(__name__)


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if platform.system() == "Darwin" else r / 1e3  # macOS bytes, Linux KB


def _accel_mem_mb(device: str) -> float | None:
    try:
        if device.startswith("cuda"):
            return torch.cuda.max_memory_allocated() / 1e6
        if device == "mps":
            return torch.mps.driver_allocated_memory() / 1e6
    except Exception:
        pass
    return None


def _disk_mb(pipe: VisionPipeline, key: str) -> float | None:
    spec = pipe.registry.specs[key]
    s = pipe.settings
    if "weights" in spec:
        p = s.cache_dir / "ultralytics" / spec["weights"]
        return round(p.stat().st_size / 1e6, 1) if p.exists() else None
    if "hf_id" in spec:
        d = Path(os.environ.get("HF_HOME", s.cache_dir / "hf")) / "hub" / ("models--" + spec["hf_id"].replace("/", "--"))
        if d.exists():
            return round(sum(f.stat().st_size for f in (d / "blobs").glob("*")) / 1e6, 1)
    if spec["type"] == "ensemble":
        parts = [_disk_mb(pipe, m) for m in spec["members"]]
        return round(sum(p for p in parts if p), 1)
    return None


def run_benchmark(images: list[Path], models: list[str], out_dir: Path, gt_path: Path | None = None,
                  warmup: int = 1) -> dict:
    pipe = VisionPipeline()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"created": datetime.now(timezone.utc).isoformat(), "device": pipe.registry.device,
              "torch": torch.__version__, "platform": platform.platform(), "images": [p.name for p in images],
              "models": {}}
    gt = None
    if gt_path:
        from vision.evaluation.metrics import load_ground_truth
        gt = load_ground_truth(gt_path)
    for key in models:
        log.info("=== benchmarking %s", key)
        rss0 = _rss_mb()
        t = time.perf_counter()
        try:
            det = pipe.registry.get(key)
        except Exception as e:
            report["models"][key] = {"error": str(e)}
            continue
        load_s = time.perf_counter() - t
        for _ in range(warmup):  # first call includes kernel compilation / CLIP / caches
            pipe.analyze_image(images[0], model=key, save_dir=None, save_annotated=False, save_json=False)
        results = [pipe.analyze_image(p, model=key, save_dir=out_dir / key) for p in images]
        entry = {"model": det.info().model_dump(), "load_seconds": round(load_s, 2),
                 "process_peak_rss_mb_after": round(_rss_mb(), 0), "rss_growth_mb": round(_rss_mb() - rss0, 0),
                 "accelerator_mem_mb": _accel_mem_mb(det.device), "disk_mb": _disk_mb(pipe, key),
                 "supported_classes": det.supported_classes(),
                 "stats": summarize_results(results),
                 "per_image": {r.image.name: {"counts": r.counts_by_class, "inference_ms": r.timing.inference_ms,
                                              "top": [(d.class_name, d.prompt, d.confidence) for d in r.detections[:8]]}
                               for r in results}}
        if gt:
            entry["evaluation"] = evaluate(predictions_from_results(results), gt)
        report["models"][key] = entry
        gc.collect()
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "benchmark.md").write_text(render_markdown(report))
    return report


def render_markdown(rep: dict) -> str:
    L = [f"# Model benchmark ({rep['created'][:19]}Z)", "",
         f"Device: `{rep['device']}` · torch {rep['torch']} · {rep['platform']}", "",
         f"Images ({len(rep['images'])}): " + ", ".join(rep["images"]), "",
         "Latency is warm (first call excluded). Memory figures are process-level and cumulative across "
         "models in one run, so treat them as indicative. No ground truth ⇒ no precision/recall reported.", "",
         "| model | load s | infer ms mean | infer ms p95 | disk MB | detections | classes found |",
         "|---|---|---|---|---|---|---|"]
    for k, m in rep["models"].items():
        if "error" in m:
            L.append(f"| {k} | error: {m['error']} | | | | | |")
            continue
        s = m["stats"]
        L.append(f"| {k} | {m['load_seconds']} | {s['inference_ms']['mean']} | {s['inference_ms']['p95']} | "
                 f"{m['disk_mb']} | {s['detections']} | {', '.join(f'{c}:{n}' for c, n in s['counts_by_class'].items())} |")
    L += ["", "## Per-image top detections", ""]
    for k, m in rep["models"].items():
        if "error" in m:
            continue
        L.append(f"### {k}")
        for img, v in m["per_image"].items():
            top = "; ".join(f"{c}({p}) {s:.2f}" for c, p, s in v["top"]) or "—"
            L.append(f"- **{img}** ({v['inference_ms']} ms): {top}")
        if "evaluation" in m:
            L.append("")
            L.append("```json\n" + json.dumps(m["evaluation"], indent=2) + "\n```")
        L.append("")
    return "\n".join(L)
