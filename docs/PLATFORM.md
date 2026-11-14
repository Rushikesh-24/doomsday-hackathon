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
| AI Studio (video) | `POST /spatial/videos/upload`, `POST /spatial/videos/{id}/analyze`, `GET /spatial/jobs` (processing queue), `GET /spatial/jobs/{id}/progress`, `POST /spatial/jobs/{id}/pause\|resume\|stop\|start`, `GET /spatial/jobs/{id}/thumbnail` |
| 3D Explorer | `GET /spatial/reconstructions[/{id}[/assets\|/facilities]]`, `GET /spatial/reconstructions/{id}/assets/{name}`, places: `GET/POST /navigation/{rid}/places`, `PATCH/DELETE /navigation/{rid}/places/{pid}`, floor plan: `GET /navigation/{rid}/plan`, `POST .../plan/generate`, `POST .../plan/upload`, `PATCH .../plan/transform`, `DELETE .../plan/uploaded`, `GET .../plan/image/{generated\|uploaded}` |
| Facility Intelligence | `GET /platform/facilities?q&class&status`, `POST /platform/facilities/{id}/review`, `GET /platform/facilities/{id}/reviews` |
| Campus Explorer | `GET /platform/campus`, `POST/DELETE /platform/campus/buildings`, `POST /platform/campus/buildings/{id}/floors`, `POST /platform/campus/links`, `DELETE /platform/campus/links/{kind}/{id}` |
| Wayfinding | `GET /navigation/{rid}/graph`, `POST /navigation/{rid}/route` (start/goal = node id, place name/alias, facility class or `start`), `POST /navigation/{rid}/ask` (free text, e.g. "from CC lab to CG lab"), `POST /ai/transcribe`, `POST /ai/localize/{rid}` |
| Compliance | `GET /platform/compliance`, review endpoint above |
| AI Lab | `GET /ai/models`, `POST /ai/describe`, `POST /ai/transcribe`, `POST /ai/localize/{rid}`, `GET /spatial/reconstructions/{rid}/frames/{frame}` |

| Mobile app (sync) | `GET /maps`, `GET /maps/{id}/pack` (ETag / 304), `GET /maps/{id}/plan.png`, `PATCH /maps/{id}` (name, description, hidden, location) |
| Mobile camera guide | `POST /maps/{id}/locate` (photo → nearest keyframe, heading, confidence, scene hints, optional caption) |
| Explorer → Score | `GET /spatial/reconstructions/{id}/accessibility` (score, grade, 6 categories with reasons, located issues, measured corridor) |
| Explorer → Details → Delete | `DELETE /spatial/jobs/{id}` (409 while active; keeps the source video) |
| Assistant (Groq) | `GET /assistant/status`, `POST /assistant/chat` `{map_id, messages, profile, start?, route?}` |

## Job controls

* **Pause** suspends the running worker process (COLMAP or splat training) in place, using psutil
  (NtSuspendProcess on Windows, SIGSTOP on POSIX). In-process stages wait at their next progress report.
* **Resume** continues exactly where it stopped.
* **Stop** kills the worker. The interrupted stage goes back to pending, so **Start** resumes the job from its
  last finished stage (optionally with a new preset). A finished job can be re-run at higher quality
  (`force_stages: ["reconstruction", "export"]`).
* After a server restart, jobs that were running or paused show as *interrupted*; Start resumes them.

## Mobile app and sync

The phone app (`mobile/`, Flutter) is offline-first. `GET /maps` lists every finished scan; each is
downloaded as a versioned *map pack*: graph, places, walked path, plan geometry and the directions wording,
about 25 KB, plus `plan.png`. The version is a content hash, so the phone re-downloads only when places, the
plan or the scan change. Routing runs on the phone (a Dart port tested word-for-word against the Python), so
offline answers equal online ones. The registry of names and visibility lives in `platform.db` (`maps` table).

## Accessibility score and measured floor plan

`vision/navigation/floor.py` measures the corridor along the walk: every 0.25 m the scan's points at body height
are projected onto the path normal and the nearest dense band on each side is taken as the wall (median-filtered;
sides without evidence are "open", drawn dashed). This gives the phone's vector floor plan and a clear-width
estimate per sample.

`vision/navigation/score.py` turns the scan into a 0-100 score (A-E) from six weighted categories: step-free reach
30, lift/ramp alternative 10, stair safety 15, clear width 20 (≥ 0.9 m), obstacle-free path 10, and wayfinding aids 15.
Each category comes with its plain-language reason, and each issue with a position, which the Explorer (3D and plan)
and the phone show as pins. Anything the scan did not see is reported as unknown. Results are cached in `plan/score.json`
and recomputed when the trajectory, facilities, places or alignment change. Map packs (format 2) carry the floor and score.

## Camera guide (`vision/navigation/locate.py`)

DINOv2 global descriptors shortlist 8 keyframes, then SIFT + ratio test + RANSAC fundamental matrix re-rank them by
geometric inliers. Confidence is "high" only when the best keyframe beats every rival that is somewhere else
(neighbouring frames along the walk do not count). Heading = keyframe viewing direction corrected by the inliers'
horizontal shift. The facility detector adds hints with a direction (left / ahead / right), hazards first.
Held-out video frames are placed on the right keyframe (`tests/test_locate.py`); photos of another scan come back
"low".

## Assistant (Groq)

`POST /assistant/chat` uses Groq's OpenAI-compatible API with tool calling. The model gets the map's real
places and facilities, and must call `plan_route` (the same router) for directions, so it cannot invent rooms
or routes. Replies are plain, short and speakable. Set `GROQ_API_KEY` (and optionally `GROQ_MODEL`) in `.env`;
if the preferred model is unavailable for the key, the best available chat model is chosen automatically.
The key stays on the server.

## Security

* `.env` is loaded at startup (real environment variables win) and is git-ignored.
* `ACCESSMAP_API_TOKEN` (optional) requires a token on every write (`X-AccessMap-Token` or `Bearer`).
  Reads stay open so phones can sync. The web app sends `NEXT_PUBLIC_ACCESSMAP_API_TOKEN`, and the phone has a
  token field in Settings. Set it whenever the API is reachable beyond a trusted LAN, and restrict
  `CORS_ORIGINS`.

## Data honesty

* All numbers are computed from stored results; repeated analyses of one photo count once (latest).
* Every photo carries provenance: **Real photo · licence** (from `samples/*/SOURCES.json`) or **Your upload**.
  The synthetic test walk (`vision/spatial/synthetic_walk.py`) is only used by tests and is not shown.
* Absolute sizes are withheld (`REQUIRES_CALIBRATION`) unless a scale source exists; compliance is
  screening only — the UI never says "compliant".
* The campus structure starts empty: buildings and floors are created by the user.
* Navigation covers only the walked path of a video; unexplored space is unknown, not blocked. A
  reconstruction without a scale reference reports distances descriptively instead of in metres.

## Places, floor plans and directions

* **Places** (`<job>/places.json`) are user-tagged points such as *CC Lab* or *CG Lab*, with a kind, aliases, an
  arrival hint and accessibility notes. Each one joins the graph at its nearest walked-path node. Places of kind
  *Stairs* / *Ramp* also tag the path edges within 2 m (or 3 path steps if unscaled), so a manually marked
  staircase blocks wheelchair routes just like a detected one. *Lift* / *Toilet* / *Entrance* places answer
  "nearest lift" etc.
* **Directions**: ground-plane headings give *turn left/right* and *bear left/right*; tagged edges add stairs,
  ramp and obstacle notes; places and facilities near the path become landmarks ("passing CC Lab on your
  right"). The profile (walking, wheelchair, low vision) adds the help that person needs.
* **Floor plan** (`<job>/plan/`): `generated.png` is a top-down density map of the sparse points plus the splat
  centres on the walked storey (+Y up, so image x = local x and image y = local z). An uploaded plan is placed
  over it with a saved similarity transform (scale, rotation, offset, opacity).
* **3D Explorer modes**: Orbit; Walk (first person at eye height, WASD/arrows, drag to look, height follows
  the walked floor); Replay walk / Walk me there (the camera glides along the path or route).
* Stage demo kit: [demo/README.md](../demo/README.md).

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
