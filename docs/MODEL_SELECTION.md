# Model & dataset discovery (PS32)

Date: 2026-11-13. Hardware used for all measurements: Apple M2, 16 GB, PyTorch 2.15 on MPS.

## 1. Requirement

Detect stairs, ramps, lifts, doors, entrances, handrails, tactile paving, corridors, walkways,
accessibility symbols, signboards and obstacles in campus photos and videos, locally (GPU with CPU
fallback), without pretending that a model knows a class it was never trained on.

## 2. Candidates examined

| Model | Type | Classes relevant to us | Weights / origin | Licence | Size on disk | Runs locally |
|---|---|---|---|---|---|---|
| **OWLv2** base-patch16-ensemble | open-vocab (ViT, text-conditioned) | any text prompt | `google/owlv2-base-patch16-ensemble` (Google, HF) | Apache-2.0 | 621 MB | yes (MPS/CUDA/CPU) |
| **Grounding DINO** tiny | open-vocab (Swin-T + BERT) | any text prompt (≤256 tokens/pass) | `IDEA-Research/grounding-dino-tiny` (HF) | Apache-2.0 | 690 MB | yes |
| **YOLO-World v2** s / l | open-vocab YOLO (CLIP text) | any text prompt | Ultralytics GitHub release assets | AGPL-3.0 (Ultralytics) | 26 / 94 MB | yes, fastest |
| **YOLO11m** COCO-80 | closed set | person, chair, bench, bicycle, suitcase, potted plant, table, … (no stairs/ramps/doors) | Ultralytics GitHub release assets | AGPL-3.0 | 41 MB | yes |
| Florence-2 | VLM (caption + phrase grounding) | open phrases | `microsoft/Florence-2-*` (native in transformers ≥5) | MIT | 0.5–1.5 GB | **not used as detector**: its `<OD>` / phrase-grounding outputs return boxes **without confidence scores**, so thresholds and the required confidence field would be fabricated. Useful later for captions/OCR (`<OCR>`) of signboards. |
| MM-Grounding-DINO, OmDet-Turbo | open-vocab | any text prompt | native in transformers 5.x | Apache-2.0 | — | **not yet benchmarked**; next candidates (OmDet-Turbo targets real-time) |
| `mudasir13cs/floorcad-yolov8n-detect` (HF) | closed set | door, stairs, elevator **symbols on CAD floor plans** | HF | AGPL-3.0 | small | rejected: trained on drawings, not photos |
| `pauhidalgoo/yolov8-indoor` (HF) | closed set | indoor objects (class list not confirmed to include stairs/doors) | HF | unclear | small | not adopted (unverified classes/licence) |

No public, licence-clear *pretrained checkpoint* was found that covers our accessibility taxonomy on
real photos. Therefore the engine is built on **open-vocabulary detection**, with closed-set COCO
detection for everyday obstacles, and a fine-tuning path for when campus data is available.

## 3. Benchmark (same 10 real photos, same taxonomy prompts)

Images: `samples/images/` (Wikimedia Commons, licences in `SOURCES.json`). Script:
`python scripts/benchmark_models.py`. Full report of the final run: [benchmark_2026-11-13.md](benchmark_2026-11-13.md)
(`outputs/benchmark/benchmark.md` is regenerated on each run).

Latency is warm mean inference per image on MPS:

| model | infer ms (mean) | disk MB | facility classes found across the 10 images |
|---|---|---|---|
| yolo_world_s | ~106 | 26 | staircase 1, ramp 2, elevator 4 (mostly wrong boxes, see below) |
| yolo_world_l | ~134 | 94 | staircase 1, elevator 1 |
| grounding_dino_tiny | ~4400 | 690 | staircase 2, ramp 4, elevator 15, handrail 9, tactile 10 |
| owlv2_base | ~812 | 622 | staircase 6→merged, ramp 2, elevator 1, door 4, handrail 25, tactile 4 |
| yolo11_coco | ~91 | 41 | (obstacles/people only, by design) |
| **campus_ensemble** (OWLv2 + YOLO11) | ~881 | 662 | as OWLv2 for facilities; COCO for obstacles/people |

**No precision/recall is reported** because these photos have no ground-truth boxes. Detection
quality was judged by visual inspection of the annotated outputs (side-by-side sheets in
`docs/compare_*.jpg`), which is qualitative:

* **YOLO-World**: fast, but labelled the Sydney staircase steps as `bench` (×9), missed both ramps
  and the elevator. Unsuitable as the primary facility detector without fine-tuning.
* **Grounding DINO tiny**: found the Hull steps + ramp + railing, but tends to force a nearby prompt
  onto salient objects (elevator → "barrier", floor indicator → "accessibility symbol"), and is ~5×
  slower than OWLv2 because the prompt list must be split into several passes.
* **OWLv2**: best facility recall: OAU and Sydney staircases, tactile paving (Singapore, Doncaster),
  elevator labelled as elevator, handrails. Two systematic issues, both handled in post-processing:
  (a) it boxes a staircase step-by-step → `merge_fragments`; (b) it over-fires on furniture and
  text panels → furniture prompts are delegated to COCO (`exclude_prompts`), and
  `accessibility_symbol`/`signboard` thresholds are raised.
* **YOLO11 COCO**: reliable people/chairs/benches/bicycles; cannot see facilities (not claimed).

**Decision:** default `campus_ensemble` = OWLv2 (facilities, signage, non-COCO obstacles like cones
and barriers) + YOLO11m (people and COCO obstacles). `campus_ensemble_fast` (YOLO-World-L + YOLO11) is
available when latency matters more than facility recall. All models are selectable per request.

### Licence note
OWLv2 and Grounding DINO are Apache-2.0. Anything using Ultralytics code or weights (YOLO11,
YOLO-World) is AGPL-3.0, so a closed-source commercial deployment would need an Ultralytics
Enterprise licence **or** a default of `owlv2_base` alone (Apache-2.0), which loses COCO-quality
person/furniture detection.

## 4. Datasets found (for fine-tuning, not downloaded automatically)

| Dataset | Classes (as listed) | Where | Notes |
|---|---|---|---|
| Barrierfree "accessibility object detection" | ramp, pothole, accessibility symbol, person, stairs | Roboflow Universe `barrierfree/accessibility-object-detection-lrrth` | check licence/size on page |
| Data_Walk_Asst | 10 classes incl. elevator, handrail, door, stairs, crosswalk | Roboflow `shg-ymdrb/data_walk_asst` | handrail + elevator coverage |
| UISTDEMO / drone-accessibility-mapping projects | door, elevator, slope, stairs | Roboflow `drone-accessibility-mapping/*` | indoor building views |
| 無障礙尋找 (c112156211) | objects, ramp, tactile_paving | Roboflow `c112156211/-ognxn` | only tactile-paving source found |
| stair-elevator, Stairs + Road, Stairs_Detection, YOLO door detection | stairs / door | Roboflow (several) | single-class, mixed quality |
| Image Dataset of Accessibility Barriers | steps, stairs, ramps, grab bars (5538 images) | Zenodo record 6382090 | real public-space photos; good for stairs/ramps/handrails |

Roboflow downloads need the user's API key and per-dataset licence acceptance, so they are a
manual step. `configs/dataset_class_map.yaml` already maps the label names used by these datasets
(e.g. `upstair`, `slope`, `grab bar`, `braille block`) to the canonical taxonomy;
`scripts/prepare_dataset.py` merges them, reports unmapped labels, removes duplicate images and
splits deterministically by content hash.

## 5. When to fine-tune

Fine-tuning has **not** been run. Recommended only for classes where open-vocab recall is poor on
real campus data, measured with ground truth (`scripts/benchmark_models.py --gt`). Current
candidates based on qualitative inspection: `ramp`, `tactile_paving` (pattern type), `door`
(opening orientation), `elevator_panel`.
