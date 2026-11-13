# Video → spatial intelligence & 3D Gaussian splats (PS32 prompt 03)

Status date: 2026-11-13. Hardware: Apple M2 (16 GB, MPS), **no CUDA GPU**. All numbers below are measured.

## 1. Pipeline

```
upload ─► quality analysis ─► keyframes ─► detection+tracking ─► SfM (COLMAP worker) ─► splat training (worker)
       ─► gravity/scale alignment ─► facility→3D registration (+ measurement engine) ─► export
```
`vision/spatial/`: `quality.py`, `frames.py`, `tracking.py`, `sfm.py`, `splat/{trainer,composite,run,brush}.py`,
`alignment.py`, `registration.py`, `pipeline.py` (resumable stages), `jobs.py` (SQLite + background runner),
`synthetic_walk.py` (exact-GT test scenes). API: `api/spatial.py`. CLI: `scripts/spatial_run.py`.

Each stage writes artifacts + status into `outputs/spatial/jobs/<job>/state.json`; completed stages are
skipped on re-run, failed ones can be retried (`resume_job_id`, `force_stages`), dependent stages of a failed
stage are marked `blocked` (never faked). COLMAP and splat training run in **worker processes** (pycolmap and
PyTorch bundle different OpenMP runtimes that abort when loaded together; it also isolates crashes).

## 2. Tool & model selection

| Need | Candidates examined | Chosen | Why / why not |
|---|---|---|---|
| SfM / poses | COLMAP (pycolmap), hloc (SuperPoint/LightGlue), GLOMAP, OpenMVG, DUSt3R/MASt3R | **pycolmap 4.2.1** (BSD-3, official wheels for macOS arm64) | reproducible pip install, CPU-capable, standard model format. hloc's SuperPoint weights are non-commercial (Magic Leap licence); DUSt3R/MASt3R are CC BY-NC; GLOMAP needs a source build |
| Feature matching | SIFT (COLMAP), SuperPoint+LightGlue, ALIKED+LightGlue | SIFT, sequential + quadratic overlap, exhaustive fallback | licence-clean, robust; learned matchers listed as next step |
| 3DGS training | graphdeco diff-gaussian-rasterization, gsplat / Nerfstudio Splatfacto, OpenSplat, Brush | **Brush v0.3.0** (Apache-2.0, approved by the operator 2026-11-13, archive sha256 pinned) when `VISION_BRUSH_BIN` is set (`splat_backend=auto`), else the **in-house PyTorch 3DGS** | reference rasterizers and Splatfacto are CUDA-only; OpenSplat publishes no binaries |
| 2D tracking | ByteTrack, BoT-SORT (Ultralytics), existing IoU+centre tracker | **IoU+centre** default, ByteTrack selectable | see §4: ByteTrack's high-confidence gate discards most open-vocabulary detections |
| Metric scale | ArUco, user segments, camera-height prior, monocular metric depth (DA-V2) | all, with strict labelling | only ArUco/segments yield `scale_calibrated` |

### In-house Gaussian splatting (what it is and is not)
Faithful 3DGS model: anisotropic Gaussians (mean, log-scale, quaternion, opacity, SH-0 colour), EWA projection
with 0.3 px low-pass, 16×16 tiles, per-tile front-to-back sort, **all** overlapping Gaussians composited with
early termination, custom autograd backward implementing the reference back-to-front recurrence
(verified with `torch.autograd.gradcheck` and against a plain-autograd reference to 1e-15), L1+SSIM loss,
paper learning rates, clone/split/prune densification, valid-pixel masks after undistortion.
Not implemented: higher-order SH (view-dependent colour), opacity reset, exposure compensation.
Speed on M2/MPS: ~0.5 s/iteration at 320 px with ~16k Gaussians (rises with Gaussian count) — roughly
50–100× slower than CUDA gsplat, so presets are short (fast = 1000 iterations).

## 3. Scale & coordinate frames (stored in `alignment.json`)
* `sfm`: COLMAP frame (arbitrary similarity). Cameras: x_cam = R·x_sfm + t.
* `local`: x_local = s·R_g·(x_sfm − c₀) — gravity-aligned (+Y up) from a RANSAC ground plane of sparse points
  (fallback: mean camera up), origin at the first camera, metres if a scale exists. Stored as `T_local_from_sfm`.
* `geo`: only with ≥ 3 ground-control points **and** a calibrated scale (Umeyama; residuals reported).
* `scale_status`: `relative` | `approximate_metric` (camera-height 1.40 ± 0.15 m prior and/or DA-V2 depth vs SfM
  depth, ±45 % measured) | `scale_calibrated` (ArUco corners triangulated across frames, user segments) |
  `georeferenced`. A phone GPS tag is kept as metadata only.

## 4. Results on real footage
Artifacts of the final stairway run are copied to `samples/spatial_demo/` (splat PLY, sparse PLY, trajectory
JSON/PNG, facilities, metadata, keyframe sheet, held-out render comparison).

### 4a. "Up the Lake Creek Falls Stairway" (Wikimedia Commons, CC BY-SA 3.0, 50 s, 640x480 upscaled, letterboxed)
| Stage | Result |
|---|---|
| Quality | `good` 82.6/100; CAMERA_SHAKE, 2 SCENE_CUTS; letterbox detected and cropped |
| Keyframes | 156 (13 forced-imperfect) |
| Tracking | 354 detections → 36 confirmed tracks (IoU+centre) vs 11 (ByteTrack: its 0.25 confidence gate drops most open-vocab detections) |
| SfM | **partial**: primary model 44/156 frames (28 %), 3,664 points, reprojection 0.81 px, 0 temporal jumps; 8 disconnected models in total |
| Gaussian splat | 25,286 Gaussians, 1000 iterations @ 320 px on MPS in 288 s; **held-out PSNR 17.7 dB, SSIM 0.50** (train 19.1 dB). Renders reproduce stairs and both handrails from held-out views, blurry |
| Alignment | `relative`: camera-height prior (1.46 m/unit) and monocular prior (0.23 m/unit) disagree x6.4 → no metric claim; gravity from ground plane |
| Registration | 31 facilities, 5 registered in 3D (handrail merged from 4 tracks / 53 observations, conf 0.73; staircase 26 observations, conf 0.49; walkway 0.81), 26 unregistered with reason (frames outside the primary model / no surface support) |
| Measurements on facilities | staircase step count 9 (ESTIMATED); lengths REQUIRES_CALIBRATION / INSUFFICIENT_DATA |

**What went wrong first and was fixed (all on this real video):** a 90-keyframe selection registered only 16
frames → denser keyframes; the splat optimisation collapsed (50 Gaussians, PSNR 8.4) because transmittance was
recovered by division → per-pixel termination + quality gate; exhaustive matching fused two similar stair
flights into a 93-frame model with 13 "teleports" (a nicer-looking number that was wrong) → temporal-consistency
check, primary model chosen by consistent frames; stale keyframes from an earlier run leaked into SfM → frames
directory cleared on re-extraction.

### 4a-bis. Brush vs in-house trainer (same job, same 6 held-out frames, same evaluator)
Brush trains on a copy of the COLMAP model with the held-out frames deregistered; both PLYs are scored by
`vision/spatial/splat/evaluate.py` (SH 0-3, identical views/masks; it reproduces the in-house trainer's own
score exactly).

| Backend | Steps @ res | Time (M2) | Gaussians | Held-out PSNR / SSIM | Train PSNR |
|---|---|---|---|---|---|
| in-house PyTorch | 1000 @ 320 | 288 s | 25,286 | 17.69 / 0.497 | 19.08 |
| Brush (matched budget) | 1000 @ 320 | 10 s | 5,199 | 17.61 / 0.488 | 18.21 |
| **Brush fast preset** (job default now) | 5000 @ 640 | 149 s | 32,341 | **19.53 / 0.604** | 20.67 |

Same quality at equal budget (independent validation of the in-house trainer); Brush is ~28x faster, so its
preset trains longer at higher resolution for +1.8 dB. Brush PLYs are SH degree 3 in the COLMAP frame
(verified); the local-frame export keeps base colour only. Both PLYs are in `samples/spatial_demo/`.

### 4b. Metro corridor walkthrough (CC BY 4.0, 27 s portrait)
Quality `fair` 59.4 (LOW_OVERLAP: 76 % of frame steps keep < 35 % of features). An earlier SfM probe fragmented
it into ≤ 32-frame models - consistent with the quality warning. Full splat run not repeated for time.

### 4c. Failure scenarios (real and synthetic) - what the system does
| Scenario | Input | Observed behaviour |
|---|---|---|
| Edited montage / little movement | real "Access City Logroño" CC-BY video | `fair` 58.7: LITTLE_MOVEMENT (74 % near-duplicate frames), SCENE_CUTS (12), REPEATED_VIEWS; 81 of 124 keyframes forced-imperfect; recommendations to record one continuous walking take |
| Fast hand-held dash, low overlap | real metro corridor walkthrough | `fair` 59.4: LOW_OVERLAP (76 % of steps keep < 35 % of tracked features); COLMAP splits it into ≤ 32-frame fragments (6 models) - reported as `partial`, each fragment keeps its own frame |
| Dark / blurry / shaky / static / too short | synthetic variants (tests) | POOR_LIGHTING, CAMERA_SHAKE, LITTLE_MOVEMENT, TOO_SHORT flagged with recommendations; good capture scores higher |
| Insufficient frames | 6-frame static clip (test) | `frames` fails ("only N usable keyframes"); `sfm`, `reconstruction` → `blocked`; no PLY written |
| Failed reconstruction | SfM model with 0 points (test) | worker exits non-zero, `splat.json` status `failed` with error; no PLY; stage `failed` |
| Diverged optimisation | real stairway, 16-view model (first run) | produced 50 Gaussians / train PSNR 8.4 → now caught by the quality gate (< max(200, 10 % of init points) or PSNR < 12 dB ⇒ `failed`) |
| Missing GPU | `VISION_DEVICE=cpu` | trainer runs on CPU but presets are reduced to 400 iterations @ 240 px and the run is flagged `cpu_fallback` |
| Insufficient scale references | no marker/segments | `relative` or `approximate_metric` only; contradictory priors (camera-height vs monocular, x7 apart on the stairway) ⇒ `relative` with `prior_conflict` note; never `scale_calibrated` |
| Brush backend not approved | `splat_backend=brush` without `VISION_BRUSH_BIN` | API 400 with reason |

## 5. Validation against exact ground truth (synthetic ray-cast walkthrough, `tests/test_spatial.py`)
36-frame textured corridor walk (640×480, known poses) with a 0.40 m ArUco marker:
* COLMAP registered 36/36 frames, 9.5k points, reprojection error 0.30 px.
* Trajectory RMSE after similarity alignment: 3.9 cm over a 3.22 m path (1.2 %).
* ArUco scale: 0.2559 vs true 0.2512 m/unit → **1.9 % error** (uncertainty floor set to 2 % accordingly).
* Facility registration (marker as a "signboard" seen in all frames): within 0.15 m of the independently
  triangulated marker centre; two duplicate tracks merged into one facility; a box without surface support was
  left unregistered with a reason.

## 6. Limitations (actual)
* Single real walkthrough reconstructed end-to-end; no real footage with an ArUco marker or measured distances, so
  **no real metric accuracy is claimed**; only synthetic GT gives metric errors.
* In-house 3DGS on MPS is slow (~5 min for a fast preset) and blurry at 1000 iterations; SH-0 only.
* SIFT + incremental COLMAP fragments fast/hand-held footage; disconnected models are not merged.
* Registration needs sparse points inside boxes: thin objects (handrails at distance), sky-backed signs and
  frames outside the primary model stay unregistered.
* Gravity from the camera-up fallback assumes an upright phone.
