# Accessibility Intelligence Platform (web app)

Next.js front end (`web/`) over the existing Python backend (`api/`). The two run independently: the web
app only talks to the API over HTTP (`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`), and every
backend feature still works from the CLI and API without it.

## Run

Terminal 1 — API (Python 3.11 venv from the main README):

```bash
VISION_BRUSH_BIN=tools/brush/brush-app-aarch64-apple-darwin/brush_app .venv/bin/uvicorn api.app:app --port 8000
```

(`VISION_BRUSH_BIN` is optional: without it 3D reconstruction uses the in-house PyTorch splat trainer.)

Terminal 2 — web app (Node 20+, pnpm):

```bash
pnpm --dir web install
```
```bash
pnpm --dir web dev
```

Open http://localhost:3000. For a production build: `pnpm --dir web build && pnpm --dir web start`.
To point the web app at another API host, set `NEXT_PUBLIC_API_URL` in `web/.env.local`.

## Judge demo flow

1. **AI Studio → Photos**: pick a stored real sample (e.g. `campus_entrance_rmcad.jpg`) or upload a photo →
   detections with boxes, attributes and scene findings appear.
2. **Measure** (same screen): choose a scale source (priors / uncalibrated / your own reference such as
   "door is 0.9 m wide") → per-facility measurements with status (ESTIMATED / REQUIRES_CALIBRATION / …),
   uncertainty and rule-screening outcomes. **Read signs** OCRs signboard detections with Florence-2.
3. **AI Studio → Video**: upload a walkthrough video, choose a preset, start reconstruction → live stage
   progress (quality → keyframes → tracking → SfM → splat → alignment → registration).
4. **3D Explorer**: the Gaussian splat, COLMAP point cloud, camera path and registered facility boxes;
   click a facility for its 3D evidence.
5. **Facility Intelligence**: search/filter every detected facility; verify / reject / mark fixed with a
   note (stored review trail; detections themselves are never edited).
6. **Navigation**: choose a profile (wheelchair forbids stairs), a destination (or say it — Whisper),
   optionally "locate me" from a photo (DINOv2) → route drawn in 3D, steps read aloud by the browser.
7. **Compliance**: possible issues per rule, accessibility gaps, review/fix progress.
8. **AI Lab**: Florence-2 captions/OCR, Whisper voice, DINOv2 localisation on your own inputs.

## Screens → endpoints

| Screen | Endpoints |
|---|---|
| Overview | `GET /platform/overview` |
| AI Studio (photos) | `GET /platform/library/images`, `GET /platform/library/images/{id}`, `POST /vision/analyze-image`, `POST /measurements/analyze-stored`, `POST /ai/read-signs/{id}` |
| AI Studio (video) | `POST /spatial/videos/upload`, `POST /spatial/videos/{id}/analyze`, `GET /spatial/jobs/{id}/progress` |
| 3D Explorer | `GET /spatial/reconstructions[/{id}[/assets\|/facilities]]`, `GET /spatial/reconstructions/{id}/assets/{name}` |
| Facility Intelligence | `GET /platform/facilities?q&class&status`, `POST /platform/facilities/{id}/review`, `GET /platform/facilities/{id}/reviews` |
| Campus Explorer | `GET /platform/campus`, `POST/DELETE /platform/campus/buildings`, `POST /platform/campus/buildings/{id}/floors`, `POST /platform/campus/links`, `DELETE /platform/campus/links/{kind}/{id}` |
| Navigation | `GET /navigation/{rid}/graph`, `POST /navigation/{rid}/route`, `POST /ai/transcribe`, `POST /ai/localize/{rid}` |
| Compliance | `GET /platform/compliance`, review endpoint above |
| AI Lab | `GET /ai/models`, `POST /ai/describe`, `POST /ai/transcribe`, `POST /ai/localize/{rid}`, `GET /spatial/reconstructions/{rid}/frames/{frame}` |

## Data honesty

* All numbers are computed from stored results; repeated analyses of one photo count once (latest).
* Every photo carries provenance: **Real photo · licence** (from `samples/*/SOURCES.json`) or **Your upload**.
  The synthetic test walk (`vision/spatial/synthetic_walk.py`) is only used by tests and is not shown.
* Absolute sizes are withheld (`REQUIRES_CALIBRATION`) unless a scale source exists; compliance is
  screening only — the UI never says "compliant".
* The campus structure starts empty: buildings and floors are created by the user.
* Navigation covers only the walked path of a video; unexplored space is unknown, not blocked. A
  reconstruction without a scale reference reports distances in "units".

## Additional AI models (`vision/ailab/models.py`)

| Model | Licence (verified on the model card) | Used for | Measured here | Limitations |
|---|---|---|---|---|
| `florence-community/Florence-2-base` (native transformers, no remote code) | MIT | captions, OCR, OCR regions, reading signboard crops | ~2.1 s OCR+regions on the toilet-signs sample (MPS); read all three sign texts | small text and stylised fonts are missed; no translation |
| `openai/whisper-base` | Apache-2.0 | voice destination → facility class (keyword intents in English, Hindi, Urdu, Spanish, French, German) | English "where is the elevator" → `elevator` | base model is weak for Hindi; auto-detect may output Urdu script — pick the language in the UI |
| `facebook/dinov2-small` | Apache-2.0 | photo → nearest reconstruction keyframe | a frame from outside the reconstruction still scored 0.70, so only > 0.9 is "high" | keyframe-level only (no 6-DoF pose); look-alike places confuse it |

Weights download on first use into `models_cache/` (respecting `VISION_OFFLINE=1`).

## Front-end structure (`web/src`)

```
lib/api.ts            centralised client (GET/POST/upload with progress, typed errors)
hooks/use-api.ts      data fetching + polling; use-scene.ts (reconstruction loader); use-recorder.ts (mic)
components/app/       shell, status/provenance badges, loading/error/empty states, bbox overlay
components/studio/    photo workspace, measurement panel, sign reader, video upload + job monitor
components/three/     scene-viewer.tsx (React Three Fiber + @mkkellogg/gaussian-splats-3d, MIT)
app/<section>/page.tsx  one route per section
```

Stack: Next.js 16 (App Router), React 19, TypeScript, Tailwind 4, shadcn/ui (Base UI), three.js,
@react-three/fiber + drei, lucide icons.

## Tests

* Backend: `tests/test_platform.py` (overview, reviews, campus CRUD, navigation profiles and handrail
  overlap, frame-name validation, model licences, bad-input handling).
* Front end: `pnpm --dir web lint` and `pnpm --dir web build` (type-check) are clean; screens were checked
  manually in a browser against the live API. There are no automated UI tests yet.
