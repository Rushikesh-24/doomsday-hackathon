# Measurement & geometric intelligence (PS32 prompt 02)

Status date: 2026-11-13. Hardware for all timings: Apple M2 16 GB, PyTorch 2.15, MPS.

## 1. Pipeline

```
image ─► facility detections (reused: supplied JSON, or stored vision_request_id; detection runs only if neither is given)
      ─► per-object masks: SlimSAM box prompts (+ ADE20K labels as fallback / union for stairs)
      ─► scene parsing: UperNet-ConvNeXt ADE20K (floor/road/sidewalk/path, stairs, railing/bannister, door, sky)
      ─► metric depth: Depth-Anything-V2 metric Small (indoor or outdoor chosen from sky fraction) | Depth Pro (opt.)
      ─► intrinsics: user K / FOV / 35mm-focal  >  EXIF FocalLengthIn35mmFilm  >  Depth Pro FOV  >  26 mm prior (±15%)
      ─► gravity: RANSAC ground plane (ADE20K ground pixels minus objects), cross-checked with the vertical
                  vanishing point; VP alone if no ground; refuse slope if neither
      ─► scale recovery (section 3)  ─► analyzers (section 4)  ─► ground-truth comparison  ─► rules  ─► JSON + image
```

Code: `vision/measurement/` (`engine.py` orchestration, `backends.py` models, `camera.py`, `geometry.py`,
`scale.py`, `analyzers.py`, `rules.py`, `store.py`, `visualize.py`, `synthetic.py`). Measurements keep the
original `detection_id`; every record lists all contributing models (detector, depth, segmentation, SAM).

## 2. Model discovery & choice

| Model | Output | Licence | Size | Measured here | Decision |
|---|---|---|---|---|---|
| **Depth-Anything-V2 Metric Indoor/Outdoor Small** (HF) | metric depth | Apache-2.0 per upstream README for Small (metric cards carry no tag - verify) | 99 MB each | 0.3–1.8 s/img (MPS) | **default** |
| Depth-Anything-V2 Base/Large (relative & metric) | depth | **CC-BY-NC-4.0** | 0.4–1.3 GB | – | rejected: non-commercial |
| **Apple Depth Pro** (HF) | metric depth **+ field of view** | apple-amlr (research only) | 1.9 GB | 11 s (MPS), **496 s (CPU)** | optional (`depth_model=depth_pro`), GPU-only |
| ZoeDepth NYU-KITTI | metric depth | MIT | 1.4 GB | – | not integrated (older, superseded) |
| MoGe-2 (Microsoft) | metric point map + FOV + normals | MIT | 141 MB (S) | – | **not integrated**: current package pulls CUDA kernel deps (FlexGEMM), Gradio and two personal-account git repos; revisit with a vetted install |
| UniDepth v2 / Metric3D v2 | metric depth + intrinsics | CC-BY-NC / unclear on HF | – | – | rejected: licence / remote-code loaders |
| Marigold | affine depth (diffusion) | OpenRAIL++ | 13.7 GB | – | rejected: size, not metric |
| **UperNet ConvNeXt-T ADE20K** | 150-class semantics | MIT | 241 MB | ~2.2 s (CPU fallback: MPS lacks this pooling op) | **used** |
| UperNet ConvNeXt-S, OneFormer, Mask2Former ADE20K | semantics | MIT / MIT / "other" | | | only pickle `.bin` weights (not loaded) or unclear licence |
| SegFormer ADE20K | semantics | NVIDIA licence (non-commercial) | | | rejected |
| **SlimSAM-uniform-50** | box-prompted masks | Apache-2.0 | 112 MB | 1.4–1.8 s (MPS) | **used** |
| SAM ViT-B / SAM2.1-S | masks | Apache-2.0 | 0.4–0.75 GB | – | heavier alternatives |

Surface/plane detection, camera pose and perspective geometry are implemented in code (RANSAC planes,
vanishing points, ray–plane casting) rather than with extra networks.

**Measured depth-scale behaviour on real photos (pinhole check against a 2.1 m door, RMCAD photo):**
DA-V2 outdoor put the door at ~18 m (×2.4 too far), Depth Pro at 2.3 m (×3.3 too near, identical on CPU and
MPS). Both are far outside "a few percent" — the reason absolute lengths are withheld until a scale reference
exists.

## 3. Scale recovery

Network depth is treated as correct **up to one global scale factor** s. Each reference yields
s = true_length / network_length; references are combined in log space with inverse-variance weights,
mutually checked (>3σ disagreements are reported), and the result multiplies the depth map.

| Source | How | Status | Typical 1σ |
|---|---|---|---|
| user segment `{"type":"segment","points":[[x,y],[x,y]],"length_m":L}` | endpoints ray-cast onto a plane fitted to the band around the segment | `calibrated_reference` | 0.5 % tape + 2 px/len + plane fit |
| user object `{"type":"object","detection_id":..,"dimension":"height","length_m":L}` | in-plane extent of the object's SAM mask (doors) or single-view metrology (people) | `calibrated_reference` | as above |
| ArUco `{"type":"marker","dictionary":"DICT_4X4_50","size_m":s}` | OpenCV ArUco corners → ray-plane | `calibrated_marker` | <1 % |
| known-object priors (`use_priors`, default on) | doors 2.06 ± 0.06 m (only frontal, untruncated, aspect 1.7–3.3); standing adults 1.68 ± 0.10 m (feet visible, on ground plane) | `prior_reference_object` | 3–7 % |
| multiple images | images of the same facility (`facility_groups`) inherit the scale of a calibrated image via shared measurements; measurements are then fused (inverse-variance, χ² consistency) | `calibrated_multi_image` | ref ⊕ spread ⊕ 4 % |
| camera calibration | user `fx,fy,cx,cy,dist` (checkerboard), `hfov_deg`, `focal_35mm`; EXIF 35 mm focal; Depth Pro FOV | (intrinsics) | |
| none | – | `uncalibrated_monocular` | **lengths withheld** |

**References must lie on a visible surface.** A segment drawn over a point hidden behind something else is
measured on whatever surface is visible there (this was observed in testing).

## 4. Analyzers (what is measured, and how)

| Facility | Measurement | Method | Scale-free? |
|---|---|---|---|
| ramp | running slope (deg, %, 1:n) | RANSAC plane on ramp-mask points vs gravity; σ = plane-fit ⊕ gravity ⊕ **focal-length sensitivity (recomputed at f·(1±σf))** ⊕ 7° depth-model floor (measured, §7b) | yes |
| | length, horizontal run, rise, clear width | mask pixels ray-cast onto the ramp plane; extents along uphill/lateral axes; lower bound if truncated | no |
| | handrail presence/coverage/continuity | ADE20K railing/bannister + handrail detections mapped onto the ramp axis, per side | yes |
| | surface condition | plane-residual/length, texture energy, specular fraction → indicators only; slip resistance = manual | yes |
| staircase | step count | horizontal edges in 7 vertical strips → each strip's rows mapped through a smooth 3D depth curve (perspective-correct) → longest regular run per strip (1 or 2 edges/step) → median over strips; 2D periodicity cross-check; lower bound if truncated | yes |
| | pitch, riser:tread ratio | 3D nosing line vs gravity | yes |
| | riser height, tread depth | mean nosing spacing × sin/cos(pitch) | no |
| | width | lateral extent of stair-mask depth points on the nosing plane | no |
| | handrails (sides), tactile warning | railing evidence left/right of the run axis; tactile detections (vision engine) | yes |
| door / lift | doorway width, height, width:height | SAM mask ray-cast onto RANSAC door plane; gravity-aligned in-plane axes; refuse if truncated or grazing > 78° | ratio yes |
| | clear width estimate | doorway − 0.06 ± 0.03 m (leaf + stops at 90°); lifts: no deduction | no |
| corridor / walkway | clear width | floor plane; 10 cross-sections along the principal axis; min over sections whose both floor edges are visible (not image border, not an occluding object) else lower bound | no |
| | min available clearance, permanent obstruction | obstacle footprint (box bottom edge on floor plane) vs free lateral gap in that section; people = transient | no / yes |

## 5. Measurement record

Every measurement: `value`, `unit`, `uncertainty` (1σ) and `interval_95`, `lower_bound`, `confidence`,
`status`, `method`, `models[]` (name, version, licence, device, revision), `scale_status`, `scale_dependent`,
`verification_required` + reason, `provisional_value` (uncalibrated only), `refusal_reason`, `ground_truth`
comparison, `overlay` (lines/points in original-image pixels), `detection_id`, `timestamp`.

| Status | When |
|---|---|
| `VERIFIED` | a manual measurement was supplied for this detection+quantity (engine value and error kept in `ground_truth`) |
| `ESTIMATED` | scale-free quantity, or absolute quantity with a calibrated/prior scale |
| `REQUIRES_CALIBRATION` | absolute quantity and no scale reference: `value=null`, `provisional_value` given and labelled "not a measurement" |
| `REQUIRES_MANUAL_MEASUREMENT` | not observable (threshold height, slip resistance, door force/swing, nosing profile) |
| `INSUFFICIENT_DATA` | refused: region < 1200 px, < 400 depth points, non-planar surface, grazing view, truncated extent, no gravity reference, step edges irregular, floor not horizontal… (reason given) |

The uncertainty model (configs/measurement.yaml → `uncertainty`) is first-order. Its depth terms were set
from errors measured on real photos with laser depth (§7b, small sample); the geometry terms were checked
against exact ground truth (§7a). It has **not** been calibrated on tape-measured campus data.

## 6. Rules engine

`configs/accessibility_rules.yaml`, profiles `ada_2010` (values checked against the 2010 ADA text:
405.2 1:12, 405.5/403.5.1 36 in, 404.2.3 32 in, 504.2 risers 4–7 in, treads ≥ 11 in) and `india_hg_2021`
(only ramp 1:12 and 1200 mm confirmed; riser/tread/door/corridor thresholds left **null → NOT_CONFIGURED**
until filled from the official MoHUA text). Outcomes: `POTENTIAL_ISSUE` (violates even at the favourable end
of the 95 % band), `POSSIBLE_ISSUE` (estimate or band crosses threshold; also "handrail not detected"),
`NO_ISSUE_INDICATED` (unverified), `MEETS_THRESHOLD_VERIFIED` (manual measurement, per rule only),
`CANNOT_ASSESS`, `NOT_CONFIGURED`. Lower bounds can only prove a minimum is met or a maximum is exceeded.
No overall compliance verdict is ever produced.

## 7. Validation (actual numbers only)

### 7a. Exact geometry (synthetic ray-cast scenes) — `python scripts/validate_measurements.py synthetic`
9 scenes × 3 injected network-scale errors (×0.75, ×1.0, ×1.35), calibrated with one 0.6 m ground segment.
Isolates the geometric algorithms (masks and labels are exact; depth is exact up to the injected scale).

| quantity | n | MAE | rel. MAE | max abs | truth inside reported 95 % |
|---|---|---|---|---|---|
| running slope | 9 | 0.002° | 0.03 % | 0.002° | 100 % |
| ramp length | 9 | 0.032 m | 0.8 % | 0.039 m | 100 % |
| ramp / corridor clear width | 15 | 0.018 m | 1.2 % | 0.042 m | 100 % |
| doorway width / height | 6 / 6 | 0.009 / 0.018 m | 0.9 % / 0.9 % | 0.011 / 0.022 m | 100 % |
| step count | 6 | 0.5 steps | 6 % | 1 step (steep 0.18/0.26 flight: 7 vs 8) | 100 % |
| pitch | 6 | 0.11° | 0.4 % | 0.14° | 100 % |
| riser / tread | 6 / 6 | 3.5 / 5.5 mm | 2.0 % / 2.0 % | 4 / 7 mm | 100 % |
| stair width | 6 | 0.063 m | 3.9 % | 0.063 m | 100 % |

These numbers say the geometry is right; they say **nothing** about monocular depth on real photos.

### 7b. Real photos with laser depth (DIODE val) — `python scripts/validate_measurements.py diode-geometry [--depth-model ...]`
DIODE validation images come with laser-scanned depth and published intrinsics (fx 886.81, fy 927.06,
cx 512, cy 384; z-depth convention confirmed by a plane-flatness test). The official archive streamed at
~0.03 MB/s from this network, so only **8 images (one outdoor scene, scans 195/196)** were obtained —
a small sample; treat the numbers as indicative. Errors below are of the **monocular depth model only**
(same pixels, laser vs network), computed on ADE20K ground pixels.

| quantity | DA-V2 auto (default) | DA-V2 outdoor forced | Depth Pro |
|---|---|---|---|
| uncalibrated scale ratio net/laser (range) | 0.55 – 2.09 | 0.69 – 2.09 | 0.22 – 1.11 |
| mean \|ln scale error\| | 0.37 | 0.40 | 0.59 |
| ground-plane normal error (n=2 images) | 6.8°, 8.8° | 8.8°, 11.4° | 5.3°, 7.6° |
| lateral lengths, oracle global scale: median \|rel err\| | 20 % | 22 % | 18 % |
| lateral lengths, single 1–3 m reference: median \|rel err\| | 23 % | 25 % | 21 % |
| along-view lengths, single reference: median \|rel err\| | 69 % | 69 % | 44 % |

Consequences applied to the engine (configs/measurement.yaml): uncalibrated scale σ 45 %, residual relative
depth error σ 20 % for calibrated lengths, +45 % for lengths along the viewing direction (ramp length/run/rise,
tread depth), slope floor 7°. Measurements on the same planar object as a user reference keep a 3 % ratio
error (validated on synthetic scenes only). **With these honest bands most real-photo measurements end as
`POSSIBLE_ISSUE` rather than a confident pass/fail — that is the correct outcome for monocular photos.**

Detection-level DIODE check (`... diode`): of the 6 images available at the time, one contained a measurable
facility (a walkway); uncalibrated network width 4.86 m vs 8.69 m on laser depth (−44 %).

**Raw metric scale on a campus photo (pinhole check against a 2.1 m door, RMCAD):** DA-V2 outdoor ×2.4 too
far, Depth Pro ×3.3 too near (identical CPU/MPS).

### 7c. Real campus ground truth (tape) — `python scripts/validate_measurements.py manual --file gt.json`
**Not yet available.** No field measurements of the sample photos exist; no real-world accuracy is claimed.

## 8. Known limitations
* Monocular depth also distorts *relative* geometry (observed: step spacing compressed ~3× from bottom to top
  of a real flight). Calibrated lengths far from the reference depth get larger uncertainty, but this model of
  the error is heuristic.
* **Ramp slope from a single monocular photo cannot resolve 1:12 (4.8°)**: measured plane-orientation errors
  on real photos were 5–11°. Slopes are reported with that uncertainty; a digital inclinometer is needed.
* Short flights (≤ 3 steps) and small/far staircases are refused (not enough step edges).
* Clear door width is derived from the doorway with a fixed deduction; door state (open/closed) is unknown.
* Corridor width depends on floor segmentation; crowded scenes produce lower bounds or underestimates.
* Lift car dimensions, threshold heights, slip resistance, handrail height are not measured.
* UperNet runs on CPU on Apple MPS (unsupported pooling op) → ~2 s/image.
