# DEVELOPMENT_STATUS — PS32 Campus Facility Vision Engine

Last updated: 2026-11-13 (prompt 04: web platform + platform/navigation/AI-lab APIs; prompt 03: spatial pipeline; prompt 02: measurement engine). Repository started empty, so nothing pre-existing had to be preserved; prompt-01 functionality is unchanged except that saved vision results are now named `<image>_<request_id>.*` so they are never overwritten.

## Completed and verified

| Area | State | How it was verified |
|---|---|---|
| Image pipeline (validate → EXIF/colour normalise → downscale → detect → per-class thresholds → class-wise NMS → fragment merge → duplicate removal → attributes → JSON + annotated JPEG) | ✅ | unit tests + real photos (`samples/demo_outputs/`) |
| Batch images (per-file errors don't abort the batch) | ✅ | unit + HTTP test |
| Video: MP4/MOV validation, fixed-rate sampling, static-scene skipping, tracking, unique-facility summary, annotated frames, H.264 annotated video | ✅ | synthetic video test + real 60 s CC-BY video |
| Tracker (per-class greedy IoU + centre-distance fallback, min_hits confirmation, max_age retirement) | ✅ | unit tests; real video: 872 per-frame detections → 158 confirmed tracks |
| Accessibility attribute layer with explicit `detected / estimated / not_detected / requires_measurement / not_determinable` | ✅ | unit tests + real photos |
| Model registry (YAML), load-once caching, re-entrant lock, lazy loading, provenance (sha256 / HF commit), graceful `ModelUnavailableError` (HTTP 503) | ✅ | tests incl. offline-missing-weights |
| Device selection CUDA > MPS > CPU; HF models fall back to CPU if an op fails on the accelerator | ✅ (MPS + CPU exercised; CUDA path not run, no NVIDIA GPU here) | |
| FastAPI: `/vision/analyze-image`, `/analyze-batch`, `/analyze-video`, `/models`, `/health`, static `/vision/outputs` | ✅ | TestClient tests + live `uvicorn` with real photos and a real video clip |
| Evaluation: counts by class, confidence histograms, latency, P/R/F1/AP@IoU with ground truth, multi-model benchmark report | ✅ | unit test with hand-computed AP; benchmark run on 10 photos |
| Training pipeline: YOLO/COCO-JSON readers, class remapping to taxonomy, de-dup, deterministic hash split, Ultralytics transfer learning, versioned model cards | ✅ code, ⚠️ **training not executed** (no campus dataset yet; by design) | dataset-prep unit test |
| Model discovery, benchmark and selection | ✅ | [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md), `outputs/benchmark/benchmark.md` |

## Models

| key | role | licence | status |
|---|---|---|---|
| `campus_ensemble` (default) | OWLv2 facilities/signage/non-COCO obstacles + YOLO11m people & COCO obstacles | Apache-2.0 + AGPL-3.0 | in use |
| `owlv2_base` | `google/owlv2-base-patch16-ensemble` | Apache-2.0 | in use (via ensemble) |
| `yolo11_coco` | `yolo11m.pt` | AGPL-3.0 | in use (via ensemble) |
| `campus_ensemble_fast` | YOLO-World-L + YOLO11m, ~6× faster, weaker facility recall | AGPL-3.0 | available |
| `grounding_dino_tiny` | `IDEA-Research/grounding-dino-tiny` | Apache-2.0 | available, ~4.4 s/img on M2 |
| `yolo_world_s`, `yolo_world_l` | YOLO-World v2 | AGPL-3.0 | available |

Measured on Apple M2 (MPS), warm, mean per ≤1280-px image (final benchmark run): ensemble ≈ 0.88 s,
OWLv2 ≈ 0.81 s, YOLO-World-L ≈ 0.13 s, YOLO11m ≈ 0.09 s, Grounding DINO tiny ≈ 4.4 s. Video with the default
model at 2 fps sampling: ≈ 0.83 s per processed frame (60 s video → 110 frames, ~100 s total).

## Measurement engine (prompt 02) — completed and verified

Details, model comparison and full validation tables: [docs/MEASUREMENT.md](docs/MEASUREMENT.md).

| Area | State | How it was verified |
|---|---|---|
| Pipeline image → reused detections → SAM masks → ADE20K context → metric depth → intrinsics → gravity → scale → analyzers → rules → JSON + annotated image | ✅ | real photos (`samples/measurement_demo/`), live `uvicorn` |
| Detection reuse: inline `detections` JSON or stored `vision_request_id` (stale/overwritten IDs rejected); detection runs only if neither is given | ✅ | API test + live HTTP (`detections_source: stored`) |
| Models: DA-V2 metric indoor/outdoor Small (default), Depth Pro (optional, GPU-only, estimates FOV), UperNet-ConvNeXt-T ADE20K, SlimSAM; cached, MPS with CPU fallback, `ModelUnavailableError` on missing weights | ✅ | run on MPS; UperNet falls back to CPU (MPS pooling limitation) |
| Scale recovery: user segments, user objects (detection_id), ArUco markers, door/person priors, multi-image transfer, camera intrinsics (user/FOV/35mm, EXIF, Depth Pro FOV, prior), lens undistortion | ✅ | synthetic tests incl. a rendered ArUco marker; real-photo user-reference demo |
| Ramp: slope (°, %, 1:n) with focal-length sensitivity, length/run/rise/width, handrail presence/coverage/continuity, surface indicators | ✅ | synthetic exact GT; real photos |
| Stairs: step count (perspective-corrected per-strip method + 2D cross-check, lower bound when truncated), pitch, riser/tread, width, handrails per side, tactile | ✅ | synthetic exact GT; real photos (OAU, Sydney, Hull, Arts) |
| Doors/lifts: doorway width, height, width:height, clear-width estimate; corridors/walkways: clear width (occlusion-aware), obstacle clearance, permanent-obstruction flag | ✅ | synthetic exact GT; real photos |
| Statuses VERIFIED / ESTIMATED / REQUIRES_CALIBRATION / REQUIRES_MANUAL_MEASUREMENT / INSUFFICIENT_DATA with value, unit, σ, 95 % interval, confidence, method, models, scale status, verification requirement | ✅ | tests |
| Refusals: tiny objects, truncated extents, non-planar/grazing surfaces, no gravity, irregular step edges, non-horizontal floor, off-image/too-short references | ✅ | tests + real photos (RMCAD small stairs, lift-lobby floor, 3-px reference) |
| Rules engine (ADA 2010 verified values; India HG 2021 only verified values, rest NOT_CONFIGURED), never declares compliance | ✅ | parametrised outcome tests |
| Storage/history (SQLite) and API: `POST /measurements/analyze`, `POST /measurements/analyze-batch`, `GET /measurements/{detection_id}`, `GET /measurements/{detection_id}/history` | ✅ | TestClient + live HTTP |
| Validation tooling: synthetic exact GT, DIODE laser depth (geometry + detection level), manual tape-measured GT file | ✅ tooling; ⚠️ no tape-measured campus GT exists yet | `scripts/validate_measurements.py` |

### Measurement validation results (actual)

* **Synthetic exact GT** (27 scene variants, injected depth-scale error ×0.75–×1.35, one 0.6 m reference):
  slope MAE 0.002°, door width 0.9 %, riser/tread 2 %, ramp/corridor width 1.2 %, ramp length 0.8 %,
  stair width 3.9 %, step count MAE 0.5 (max 1). Truth inside reported 95 % band: 100 %.
* **Real photos with laser depth** (DIODE, only 8 images of one outdoor scene — download throttled to
  0.03 MB/s): uncalibrated network scale off by ×0.55–×2.09; ground-plane orientation error 6.8°/8.8°;
  lateral lengths after a single reference: median 23 % error; along-view lengths: 69 %. Depth Pro: 21 % / 44 %,
  6.5° but worse raw scale. **The uncertainty priors were reset to these measured values.**
* **Tape-measured campus ground truth: none available → no field accuracy is claimed.**

Practical meaning: from one monocular photo the engine can reliably *count steps, detect missing
handrails, give ratios and flag gross problems*; absolute dimensions carry ~±20–25 % (1σ) even when
calibrated with one reference, and ramp slopes carry ~±7°, which cannot resolve a 1:12 limit.

## Spatial pipeline (prompt 03) — video → SfM → Gaussian splat → registered facilities

Details, tool selection and results: [docs/SPATIAL.md](docs/SPATIAL.md).

| Area | State | How it was verified |
|---|---|---|
| Ingestion (MP4/MOV, ffprobe metadata incl. rotation/creation/GPS tag), letterbox detection & cropping | ✅ | tests + 3 real videos |
| Quality analysis: sharpness, exposure, LK+forward-backward motion & overlap, shake, scene cuts, near-duplicates, revisits, suitability score, recommendations | ✅ | synthetic dark/static/shaky/short tests; real montage & metro diagnosed correctly |
| Keyframe selection (parallax/overlap windows, sharpest frame, gap forcing, hard cap) & extraction | ✅ | tests; real runs |
| Facility tracking reusing the detection engine (cached per frame), IoU+centre vs ByteTrack compared, detection IDs preserved | ✅ | 20-frame identity test for both trackers; real stairway |
| SfM: pycolmap worker process, sequential → exhaustive fallback, temporal-consistency check (aliasing), multiple models kept separate, COLMAP binary + JSON/NPZ export, sparse PLY | ✅ | synthetic GT (36/36 registered, 1.2 % trajectory RMSE); real stairway |
| Gaussian splat: in-house PyTorch 3DGS (custom memory-bounded compositing with analytic backward, gradcheck-verified), presets, MPS/CUDA/CPU fallback, quality gate, 3DGS PLY export, held-out PSNR/SSIM | ✅ | tests; real stairway reconstruction |
| Brush backend (approved): holdout-free COLMAP copy, shared SH-3 evaluator, `splat_backend=auto` | ✅ | real stairway: 19.53 dB / SSIM 0.60 held-out in 149 s vs in-house 17.69 dB in 288 s; Brush test runs when `VISION_BRUSH_BIN` is set |
| Scale/gravity alignment with explicit transforms; ArUco triangulation, user segments, camera-height and monocular priors, prior-conflict rule, GCP georeferencing | ✅ | synthetic GT: ArUco scale error 1.9 %; real stairway: approximate only (no marker in footage) |
| Facility-to-3D registration (ray ∩ support surface, extended vs compact objects, 3D track merging, unregistered kept with reason) + measurement engine on best frame | ✅ | synthetic GT (within 0.15 m of triangulated marker); real stairway |
| Resumable job system (state.json per stage, force/retry, blocked stages, PID liveness for interrupted jobs), SQLite store, background runner | ✅ | tests (resume skips done stages; failure blocks downstream) |
| API: upload, analyze, jobs, progress, quality, reconstructions, assets (download), facilities | ✅ | TestClient + live HTTP (PLY download) |

**Real-footage result (stairway, fast preset):** quality `good`; 156 keyframes; 36 persistent tracks; SfM
`partial` (44/156 frames in the primary model, 0.81 px); splat (Brush) 32,341 Gaussians, held-out PSNR 19.5 dB / SSIM 0.60 (in-house trainer 17.7 dB / 0.50); scale `relative` (priors contradict x6.4 → no metric claim); 5 of 31 facilities registered in 3D.
Artifacts: `samples/spatial_demo/`. **No real metric accuracy is claimed** (no marker / measured distances in
the footage); synthetic GT: trajectory RMSE 1.2 %, ArUco scale error 1.9 %, facility position < 0.15 m.

Issues found and fixed while running on real footage: keyframes too sparse for SfM (16/90 registered → raised
budgets); splat optimisation collapse from unstable transmittance division (→ per-pixel termination + stored
transmittance, quality gate); exhaustive matching aliasing two similar stair flights (13 "teleports" →
temporal-consistency check and model selection by consistent frames); stale frames/outputs reused across
re-runs; API marking a live CLI job as interrupted; Ultralytics silently pip-installing `lap`
(→ `YOLO_AUTOINSTALL=false`, `lap` pinned).

## Web platform (prompt 04) — completed and verified in a browser

Next.js 16 app in `web/` + `api/platform.py` (details, demo flow, endpoint map: [docs/PLATFORM.md](docs/PLATFORM.md)).
Frontend and backend run independently; nothing existing was rebuilt (screens call the prompt 01-03 endpoints).

* **Overview** — live counts from stored results (10 distinct real photos / 50 analysis runs, measurement
  status mix, 1 reconstruction with 5/31 facilities placed in 3D, 19 possible issues), recent activity.
* **AI Studio** — photo library with provenance badges, upload → detection overlay (click boxes), attributes,
  findings; measurement with scale source + rule profile on the stored image (`/measurements/analyze-stored`,
  no re-upload; source images are now saved with each analysis); Florence-2 sign reading; video upload →
  reconstruction with live stage progress.
* **3D Explorer** — real Brush splat (32,341 Gaussians), COLMAP points, camera path, registered facility
  boxes in the gravity-aligned frame; metrics and downloadable assets.
* **Facility Intelligence** — search/filter 140+ facilities (photos + 3D); review trail
  (verified / needs_review / rejected / fixed) stored in `outputs/platform.db`.
* **Campus Explorer** — user-created buildings/floors; assign photos and reconstructions. Starts empty.
* **Navigation** — graph over the walked path + registered facilities; standard / wheelchair (no stairs) /
  low-vision profiles; voice destination (Whisper), "locate me" (DINOv2), route drawn in 3D, read aloud
  with browser speech synthesis. Verified: on the stairway video, wheelchair → "no route (only via stairs)",
  standard → 8.3 units with "Stairs - no handrail registered in 3D (verify on site)".
* **Compliance** — possible issues by rule with evidence image, gaps, outcome counts, review/fix progress.
* **AI Lab** — model/licence table; Florence-2 caption/OCR with region overlay (toilet signs read in 2.1 s),
  Whisper record/upload with language hint, DINOv2 localisation with matched keyframes.

Fixed along the way: drei `<Html>` labels created nested React roots ("synchronously unmount a root" errors)
→ canvas-sprite labels; React-Compiler lint violations (setState in effects) → derived state; Overview and
library lookups crashed on a fresh install with no measurement DB → guarded (covered by tests).

Not done: automated UI tests (lint + build + manual browser checks only); map/geo view of buildings
(buildings have optional lat/lon in the API but no map, since no campus is georeferenced); mobile layout is
usable but not tuned; no auth — reviews are attributed to "web user".

## Tests

`pytest -q` → **104 passed, 1 skipped** (105 with `VISION_BRUSH_BIN` set: +Brush end-to-end test) (2026-11-13): 11 platform API tests (overview on an empty store, reviews, campus CRUD, navigation profiles + handrail overlap, frame-name validation, model licences, bad input), 43 vision unit tests, 5 real-model integration tests on real
photos (skipped when weights are not cached), 26 measurement tests on analytic ray-cast scenes with exact
ground truth, 19 spatial tests (compositing gradcheck, splat training + PLY, quality flags, keyframes,
tracker identity, real COLMAP on a synthetic walk vs exact GT, ArUco scale, registration, resume/failure
handling, failed-reconstruction reporting, API).

## Supported classes and how trustworthy they are (qualitative, from 10 photos + 1 video)

No ground truth exists yet, so **no accuracy / mAP numbers are claimed**.

| class | source | observed behaviour |
|---|---|---|
| staircase | OWLv2 | found in all 3 stair photos; per-step boxes merged into one region |
| handrail | OWLv2 | usually found on stairs/ramps; also fires on balustrades/fences |
| tactile_paving | OWLv2 | found in both station photos; one low-score (0.16) false positive on plain paving |
| elevator | OWLv2 | found in the lift photo (Grounding DINO / YOLO-World did not label it correctly) |
| ramp | OWLv2 | weakest facility class: small/partial boxes on the two ramp photos → **fine-tuning candidate** |
| door / entrance_exit | OWLv2 | doors found; entrance_exit fires on many window/door frames (low confidence) |
| corridor / walkway | OWLv2 | scene-level boxes, coarse |
| accessibility_symbol / signboard | OWLv2 | real symbols ≈0.5; subtitles/text panels ≈0.2–0.3 → thresholds raised (×2.0 / ×1.3) |
| obstacle / person | YOLO11 COCO (+ OWLv2 for cones/barriers/bins/boxes) | reliable for people/chairs/benches/bicycles |
| wheelchair_user | OWLv2 | prompt `wheelchair`; untested on real photos |
| elevator_panel (auxiliary) | OWLv2 | used only for lift `status_indicators`; small panels often missed |

## Known limitations

### Measurement engine
* Monocular depth is the accuracy bottleneck (see validation above); sample for that validation is small (8 images, one scene).
* Ramp slopes cannot be resolved to the 1:12 threshold from a single photo; inclinometer readings should be entered as ground truth.
* Short (≤3-step) and distant flights are refused; individual riser variation is not measured (mean only).
* Clear door width uses a fixed 0.06 m deduction; door open/closed state unknown; thresholds, slip resistance, handrail height, lift car size are REQUIRES_MANUAL_MEASUREMENT.
* Corridor width depends on ADE20K floor segmentation; crowded or monochrome scenes give lower bounds or underestimates.
* UperNet runs on CPU under MPS (~2 s); full measurement ≈ 5–10 s/image on M2 (detection reused: ~4–7 s).
* Depth Pro is research-licensed (apple-amlr) and GPU-only; DA-V2 metric checkpoints lack an explicit licence tag (upstream says Apache-2.0 for Small).
* MoGe-2 (best-suited geometry model, MIT) not integrated: its installer pulls unvetted git dependencies.

### Vision engine

* Open-vocabulary scores are not calibrated probabilities and are not comparable across models;
  thresholds were set by inspection on 10 photos, not tuned on labelled data.
* `visible_step_count` is a 2D edge-periodicity heuristic: OAU photo → 15–18 (plausible),
  Sydney photo → 6 while ~10 steps are visible (undercount), and it refuses (<150 px tall) on
  small staircases. Treat it as indicative only.
* Ramp orientation is an **image-plane** edge angle, not a gradient. Riser height, ramp gradient,
  door/corridor width, handrail height are always `requires_measurement`.
* Door swing direction, tactile pattern type (warning vs guidance), stair direction and
  entrance-vs-exit are `not_determinable` without fine-tuned models / OCR / depth.
* Obstacle-on-path uses the obstacle's ground-contact point against walkway/corridor boxes; those
  boxes are coarse, so `potential_obstruction` is an estimate.
* Tracker has no appearance model; hard scene cuts (montage videos) fragment tracks. Fine for
  continuous walk-through footage; the demo video is a promotional montage (not a campus walk).
* Inference is serialised per process (one semaphore). Video endpoint is synchronous; long videos
  hold the request open (limits: 300 MB, 600 s, configurable).
* CUDA path is implemented but was not executed on this machine.
* Licence: default ensemble includes AGPL-3.0 Ultralytics components (see MODEL_SELECTION.md).

## Exact next steps (spatial)

1. Record a real campus walkthrough (slow walk, 30-60 s, phone upright) with a printed 15 cm ArUco marker
   (DICT_4X4_50) visible from several positions → first `scale_calibrated` real reconstruction; tape-measure
   2-3 distances between facilities and pass them as `ground_truth` to get real metric errors.
2. On a CUDA machine, add a gsplat backend and compare with Brush using `splat/evaluate.py`.
3. Replace SIFT with ALIKED+LightGlue (licence-clean learned matching) and add vocabulary-tree loop detection
   to reduce fragmentation; merge disconnected models when they share facilities/markers.
4. Add SH degree ≥ 1 and opacity reset to the trainer; train longer presets on a GPU.
5. Make the navigation graph consume `facilities.json` (`position_local`, `bbox_local`) and `trajectory.json`.

## Exact next steps (measurement)

1. **Collect tape-measured ground truth on campus** (10+ doors, 5+ corridors, 5+ stairs with riser/tread,
   3+ ramps with inclinometer slope), photograph each from 2 viewpoints with a known reference (ArUco
   DICT_4X4_50 printed at 15 cm, or a tape-measured segment) and run
   `scripts/validate_measurements.py manual --file campus_gt.json`; re-set `configs/measurement.yaml`
   uncertainty priors from those errors.
2. Finish the DIODE validation on a faster connection (`scripts/fetch_diode_subset.py`, then
   `validate_measurements.py diode-geometry` and `diode`) to get a larger real-depth sample incl. indoors.
3. Integrate MoGe-2 (MIT; metric point maps + FOV) via a vetted, pinned install and benchmark it with
   `diode-geometry`; it targets exactly the shape error that dominates here.
4. Use multi-view reconstruction (two photos + SfM/feature matching) for ramps where single-view slope
   error (~7°) is too large; or require inclinometer input.
5. Fill India HG 2021 null thresholds from the official MoHUA PDF.

## Exact next steps (vision)

1. **Collect & label campus ground truth** (≈200 photos from our campus, CVAT/Roboflow, export
   COCO or YOLO using the canonical class names) and run
   `scripts/benchmark_models.py --gt gt.json --models owlv2_base grounding_dino_tiny campus_ensemble`
   to get real precision/recall per class; then tune `threshold_scale` per class in
   `configs/taxonomy.yaml` from the PR curves.
2. **Fine-tune where recall is poor** (expected: ramp, tactile_paving, door, elevator_panel):
   download Zenodo 6382090 + Roboflow `drone-accessibility-mapping`, `Data_Walk_Asst`,
   tactile-paving sets (check licences), `scripts/prepare_dataset.py` → `scripts/train.py --base yolo11s.pt`,
   add the model to `configs/models.yaml` and include it as an ensemble member for those classes.
3. Benchmark the other native transformers open-vocab models (`omdet_turbo`, `mm_grounding_dino`)
   with the same script; OmDet-Turbo may give OWLv2-level recall at real-time speed.
4. Add OCR for signboards (Florence-2 `<OCR>` or PaddleOCR) → fills `text_content` and
   `entrance_or_exit`.
5. Add monocular depth (e.g. Depth Anything V2) to turn some `requires_measurement` attributes
   into calibrated estimates (relative clear width, step height ratio), still flagged as estimates.
6. Make `/analyze-video` asynchronous (job id + polling) for long videos; add an appearance
   embedding to the tracker for scene-cut robustness.
7. Run the test-suite and benchmark on an NVIDIA GPU to validate the CUDA path and record timings.
