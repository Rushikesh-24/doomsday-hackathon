# PS32 — Campus Accessibility Intelligence (vision engine + web platform)

AI computer-vision engine that finds accessibility-related infrastructure (stairs, ramps, lifts,
doors, entrances, handrails, tactile paving, corridors, walkways, accessibility symbols, signs,
obstacles) in campus **photos and videos**, and returns bounding boxes, labels, confidence scores,
accessibility attributes, annotated images/videos and structured JSON.

It is a Python package + FastAPI backend, plus an independent Next.js web platform in `web/`
(AI Studio, 3D Explorer, Facility Intelligence, Campus Explorer, Navigation, Compliance, AI Lab) —
see [docs/PLATFORM.md](docs/PLATFORM.md).

* Default model: **`campus_ensemble`** = OWLv2 (open-vocabulary facilities, Apache-2.0) +
  YOLO11m COCO (people & everyday obstacles). Why: [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md).
* Runs on CUDA, Apple MPS or CPU. Weights are downloaded once into `models_cache/`.
* Status, limitations, next steps: [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md).

## Setup

```bash
python3.11 -m venv .venv
```
```bash
.venv/bin/pip install -r requirements.txt
```
```bash
.venv/bin/python scripts/download_models.py          # default model; add --all for every model
```

Requires Python 3.11+ (the pinned transformers/numpy need it) and `ffmpeg` (for browser-playable H.264
annotated videos and audio transcription; without it a MPEG-4 file is still written).

**Windows + NVIDIA:** torch 2.15 has no CUDA wheel for Windows yet, so install the CUDA 13.0 build first, then
the rest (paths use `.venv\Scripts\` instead of `.venv/bin/`):

```bash
py -3.11 -m venv .venv
```
```bash
.venv/Scripts/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```
```bash
grep -v -E "^(torch|torchvision)==" requirements.txt > req-win.txt && .venv/Scripts/python -m pip install -r req-win.txt
```

Stage demo (pre-built 3D scan with tagged rooms, sample floor plan, 5-minute script): [demo/README.md](demo/README.md).

**Phone app** (Android, offline-first wayfinding, Groq assistant): [mobile/README.md](mobile/README.md).
**AI assistant:** put your Groq key in `.env` as `GROQ_API_KEY=` (see `.env.example`). It is used only by the server.

## Run the platform (API + web app)

```bash
VISION_BRUSH_BIN=tools/brush/brush-app-aarch64-apple-darwin/brush_app .venv/bin/uvicorn api.app:app --port 8000
```
```bash
pnpm --dir web install
```
```bash
pnpm --dir web dev
```

Then open http://localhost:3000. Node 20+ and pnpm are needed for the web app only; the API runs
without it. Demo flow and screen→endpoint map: [docs/PLATFORM.md](docs/PLATFORM.md).

## Demo with real images

Ten openly-licensed real photos are in `samples/images/` (sources/licences in `SOURCES.json`).

```bash
.venv/bin/python scripts/analyze_image.py samples/images/stairs_oau_campus.jpg
```

Prints every detection with its attributes and writes `outputs/demo/<name>_annotated.jpg` +
`<name>.json`. Pre-generated results for all ten photos are in `samples/demo_outputs/`.

### Video

```bash
.venv/bin/python scripts/download_samples.py --video-only     # 60 s CC-BY accessibility video
```
```bash
.venv/bin/python scripts/analyze_video.py samples/videos/access_city_logrono.mp4 --sample-fps 2
```

Writes `outputs/videos/<id>/annotated.mp4`, `frames/*.jpg` and `result.json` with unique tracked
facilities (first/last seen, observations, best frame).

### API

```bash
.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
```

| Method | Path | Notes |
|---|---|---|
| POST | `/vision/analyze-image` | multipart `file`; query `model`, `conf`, `classes=staircase,ramp`, `iou`, `include_auxiliary` |
| POST | `/vision/analyze-batch` | multipart `files` (≤50); invalid files are reported in `errors`, others still processed |
| POST | `/vision/analyze-video` | multipart `file` (MP4/MOV); query `sample_fps`, `max_frames`, `min_hits`, `write_video` |
| GET | `/vision/models` | registered models (licence, source, loaded/error state) + taxonomy and prompts |
| GET | `/vision/health` | device, loaded models, model load errors |
| GET | `/vision/outputs/...` | annotated images / videos / JSON referenced by responses |

```bash
curl -F "file=@samples/images/ramp_hull_museum.jpg" "http://localhost:8000/vision/analyze-image"
```

Interactive docs: `http://localhost:8000/docs`.

## Output contract (abridged)

```jsonc
{
  "schema_version": "1.0.0",
  "request_id": "6c31e2f71b2e",
  "image": {"name": "stairs_oau_campus.jpg", "width": 720, "height": 1280, "exif_transposed": true},
  "model": {"key": "campus_ensemble", "name": "google/owlv2-base-patch16-ensemble+yolo11m.pt",
            "version": "ensemble(owlv2_base + yolo11_coco)", "license": "...", "device": "mps",
            "revision": "owlv2_base:<hf commit>;yolo11_coco:<sha256>"},
  "thresholds": {"staircase": 0.15, "...": 0.0},
  "detections": [{
    "id": "6c31e2f71b2e-000", "class_name": "staircase", "category": "facility",
    "confidence": 0.5681, "prompt": "stairs", "detector": "owlv2_base",
    "bbox": {"x1": 162.87, "y1": 648.24, "x2": 600.39, "y2": 1112.71,          // pixels of the
             "x1n": 0.2262, "y1n": 0.5064, "x2n": 0.8339, "y2n": 0.8693, ...},  // displayed image
    "source": {"name": "stairs_oau_campus.jpg", "kind": "image", "frame_index": null},
    "attributes": {
      "handrail_present":   {"value": true,  "status": "detected", "method": "handrail detection overlaps expanded stair box"},
      "visible_step_count": {"value": 15,    "status": "estimated", "confidence": 0.6, "note": "counts visible step edges ..."},
      "riser_height":       {"value": null,  "status": "requires_measurement"},
      "direction_up_or_down": {"value": null, "status": "not_determinable"}
    },
    "model_name": "...", "model_version": "...", "timestamp": "2026-11-13T11:10:09.114+00:00"
  }],
  "counts_by_class": {"staircase": 1, "handrail": 1, "...": 1},
  "findings": [{"code": "NO_STEP_FREE_ALTERNATIVE_IN_VIEW", "severity": "warning", "detection_ids": ["..."]}],
  "timing": {"preprocess_ms": 9.0, "inference_ms": 1258.1, "postprocess_ms": 0.8, "attributes_ms": 1.1, "total_ms": 1269.1},
  "annotated_image": "/vision/outputs/images/20261113/stairs_oau_campus_annotated.jpg"
}
```

Boxes are in pixels of the image **after EXIF orientation** (what a phone gallery displays) plus
normalised `[0,1]` coordinates, so Flutter / Next.js clients can overlay them at any display size.

### Attribute status (never invent measurements)

| status | meaning |
|---|---|
| `detected` | directly supported by model detections (e.g. a handrail box overlaps the stair box) |
| `estimated` | 2D image-plane heuristic; approximate, carries its method and a confidence |
| `not_detected` | looked for it, found nothing. **Not** proof of absence |
| `requires_measurement` | physical quantity (riser height, ramp gradient, door width) that needs on-site measurement / depth + calibration |
| `not_determinable` | cannot be inferred from a single 2D image by this pipeline |

## Measurement engine (accessibility geometry)

Estimates physical properties of detected facilities — ramp slope/length/width, step count, riser/tread,
stair width, door width/height, corridor clear width, obstacle clearance, handrail evidence — and screens
them against configurable standards (`configs/accessibility_rules.yaml`: ADA 2010, India HG 2021).
Method, models and validation: [docs/MEASUREMENT.md](docs/MEASUREMENT.md).

```bash
.venv/bin/python scripts/measure_image.py samples/images/ramp_hull_museum.jpg
```
```bash
.venv/bin/python scripts/measure_image.py photo.jpg --ref "812,940,1210,944=1.00" --no-priors
```

`--ref` is a tape-measured segment in original-image pixels (must lie on a visible surface);
`--object <detection_id>:height=2.10` uses a measured object; `--marker DICT_4X4_50:0.15` uses a printed ArUco
marker. **Without a reference, absolute lengths are withheld (`REQUIRES_CALIBRATION`)**; scale-free values
(slope, step count, ratios) are still reported. Pre-generated real-photo outputs: `samples/measurement_demo/`.

API (detections are reused, never re-run when supplied):

| Method | Path | Notes |
|---|---|---|
| POST | `/measurements/analyze` | multipart `file`; form `vision_request_id` **or** `detections` (ImageResult JSON), `calibration` JSON, `ground_truth` JSON; query `depth_model`, `rules_profile`, `detection_ids` |
| POST | `/measurements/analyze-batch` | multipart `files`; form `calibrations`, `detections`, `facility_groups` (same facility across photos → scale transfer + fused values), `ground_truth` |
| GET | `/measurements/{detection_id}` | latest measurements + assessment for a detection |
| GET | `/measurements/{detection_id}/history` | all runs for that detection |

```bash
curl -F file=@samples/images/campus_entrance_rmcad.jpg -F vision_request_id=<id> \
  -F 'calibration={"use_priors":false,"references":[{"type":"object","detection_id":"<id>-004","dimension":"height","length_m":2.13}]}' \
  localhost:8000/measurements/analyze
```

Accuracy you can expect from one phone photo (measured, see docs): step counts within ±1, absolute lengths
±20–25 % (1σ) after calibration, ramp slope ±7° — screening only; flagged items need on-site measurement.

## Spatial pipeline (walkthrough video → 3D Gaussian splat + registered facilities)

```
upload → quality analysis → keyframes → facility detection+tracking → COLMAP SfM → Gaussian-splat training
       → scale/gravity alignment → facility-to-3D registration (+ measurements) → export (PLY, trajectory, JSON)
```

Design, tool selection and measured results: [docs/SPATIAL.md](docs/SPATIAL.md).

```bash
.venv/bin/python scripts/download_samples.py --video-only          # real CC-licensed walkthroughs
```
```bash
.venv/bin/python scripts/spatial_run.py samples/videos/walkthrough_stairway.mp4 --preset fast
```
```bash
.venv/bin/python scripts/spatial_run.py --resume <job_id> --force reconstruction     # retry one stage
```

Add `--marker DICT_4X4_50:0.15` if a printed ArUco marker (15 cm) is visible in the video: it is the only way
(besides user reference segments) to get a **scale-calibrated** reconstruction. Without it the result is
`relative` or `approximate_metric` (phone-height / monocular-depth priors), and it is labelled so.

| Method | Path | Notes |
|---|---|---|
| POST | `/spatial/videos/upload` | MP4/MOV; returns `video_id` + metadata (ffprobe) |
| POST | `/spatial/videos/{id}/analyze` | JSON: `preset` fast/balanced/high, `tracker`, `markers`, `references`, `gcps`, `resume_job_id`, `force_stages`; returns `job_id` (202) |
| GET | `/spatial/jobs/{id}` / `/progress` | stage-by-stage status, progress, errors |
| GET | `/spatial/videos/{id}/quality` | capture quality report with recommendations |
| GET | `/spatial/reconstructions[/{id}]` | list / metadata (registration stats, splat metrics, scale status, transforms) |
| GET | `/spatial/reconstructions/{id}/assets[/{name}]` | download `splat_ply`, `splat_local_ply`, `sparse_points_ply`, `trajectory_json`, … |
| GET | `/spatial/reconstructions/{id}/facilities` | persistent facilities with 3D positions/boxes, provenance, detection IDs |

The splat PLY uses the standard 3DGS layout (SH degree 0) and opens in common splat viewers. A finished real run
(stairway walkthrough) is in `samples/spatial_demo/`.

## Evaluation

```bash
.venv/bin/python scripts/benchmark_models.py --images samples/images --models owlv2_base campus_ensemble
```

Reports latency, load time, disk size, memory, per-class counts and confidence histograms.
Precision / recall / F1 / AP@0.5 are reported **only** when ground truth is passed with `--gt`
(JSON `{"images": {"x.jpg": [{"class": "staircase", "bbox": [x1,y1,x2,y2]}]}}`, or convert YOLO labels
with `vision.evaluation.metrics.load_yolo_labels`).

## Fine-tuning (only when needed)

```bash
.venv/bin/python scripts/prepare_dataset.py --out datasets/campus_v1 --source yolo:path/to/roboflow_export:rf_stairs:"CC BY 4.0"
```
```bash
.venv/bin/python scripts/train.py datasets/campus_v1/data.yaml --base yolo11s.pt --epochs 50
```

Each run is versioned (`runs/train/campus-vN/model_card.json` with dataset hash, base-weight sha256,
hyper-parameters and held-out metrics). Add the printed entry to `configs/models.yaml` to serve it.

## Layout

```
vision/
  config.py  device.py  taxonomy.py  schemas.py
  models/          detector interface, YOLO-World / YOLO / OWLv2 / Grounding DINO adapters, ensemble, registry (load-once cache)
  preprocessing/   decode, validate, EXIF, colour-mode, downscale
  postprocessing/  per-class thresholds, class-wise NMS, fragment merging, contained-duplicate removal
  inference/       VisionPipeline (image, array, batch)
  accessibility/   attribute derivation + scene findings, pixel geometry heuristics
  visualization/   annotation rendering
  video/           sampling, static-scene skipping, annotated video
  tracking/        IoU + centre-distance tracker for sparse frames
  evaluation/      stats, P/R/AP, multi-model benchmark
  training/        dataset merge/remap/split, Ultralytics fine-tuning with model cards
  measurement/     depth/segmentation backends, camera, scale recovery, geometry analyzers, rules, store
  spatial/         video quality, keyframes, tracking, COLMAP SfM worker, splat trainer, alignment, registration, jobs
  platform/        reviews + campus store (SQLite), read-only aggregation for the web app
  navigation/      walkable graph from a reconstruction, profile-aware routing
  ailab/           Florence-2 (OCR/captions), Whisper (voice intents), DINOv2 (visual localisation)
api/app.py, api/measurements.py, api/spatial.py, api/platform.py   FastAPI routes only
web/               Next.js web platform (independent; talks to the API over HTTP)
configs/           models.yaml, taxonomy.yaml (prompts, thresholds), dataset_class_map.yaml
scripts/           CLI entry points
tests/             unit tests (no weights needed) + real-model integration tests
```

Configuration via env vars: `VISION_DEVICE` (auto|cuda|mps|cpu), `VISION_DEFAULT_MODEL`,
`VISION_OFFLINE=1` (never download), `VISION_CACHE_DIR`, `VISION_OUTPUT_DIR`,
`VISION_MAX_UPLOAD_MB`, `VISION_MAX_VIDEO_MB`, `VISION_MAX_VIDEO_SECONDS`, `VISION_PRELOAD`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Sample media © their authors, see `samples/*/SOURCES.json` (CC0 / CC BY / CC BY-SA).
