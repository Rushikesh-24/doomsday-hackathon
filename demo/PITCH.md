# AccessMap: pitch script for the judges

A 6-minute stage script plus backup answers. Every number below comes from the demo scan (`demopcc`) or the
test suite in this repository, so you can stand behind all of it.

---

## 0. Before you go on stage (10 min)

1. Start the API and the website on the laptop (see `demo/README.md` → *Setup*). Bind the API to the LAN so the
   phone can reach it: `uvicorn api.app:app --host 0.0.0.0 --port 8000`.
2. Phone on the **same Wi-Fi**. AccessMap → Settings → Server → `http://<laptop-ip>:8000` → **Sync**. Wait for
   "Online · synced just now". The PCC map card shows **79 · B**.
3. Then turn **Wi-Fi off** on the phone (or switch to airplane mode) once, to show the offline moment later, and turn it back on.
4. Open these tabs on the laptop:
   * `http://localhost:3000/explorer?id=demopcc` (opens on the **Score** tab)
   * `http://localhost:3000/studio?tab=video`
   * `http://localhost:3000/navigation?id=demopcc`
5. Print or open `mobile/test_screens/goldens/*.png` as backup slides in case the phone or Wi-Fi fails.

---

## 1. The problem (30 s)

> "Campus maps tell you *where* a room is. They don't tell a wheelchair user whether they can get there without
> stairs, they don't warn a blind student about an obstacle in the corridor, and nobody has a map of most
> buildings' insides at all.
> AccessMap turns **one phone walkthrough video** into an accessible, navigable digital twin. It scores how
> accessible the building is, shows *where* it isn't, and guides people through it on their phone, even offline."

## 2. Video → 3D → map, live job control (60 s): AI Studio, Video tab

* Show the **processing queue**: every video job with thumbnail, stage and progress.
* Press **Pause** on a running job: the COLMAP / Gaussian-splat worker process is genuinely frozen (psutil
  suspend), then **Resume** continues from the same point. **Stop** keeps finished stages, so **Start** resumes from there.
* Talking points:
  * The pipeline does frame selection, **COLMAP** camera poses, gravity alignment and metric scale, a **Gaussian splat**
    (demo scan: 66,877 Gaussians, 26.5 dB held-out PSNR), and an **OWLv2 + YOLO11** facility detector whose
    detections are triangulated into 3D.
  * Don't start a new scan live: it takes about 5 minutes on the RTX 4060.

## 3. The digital twin and its accessibility score (90 s): 3D Explorer

* The Explorer opens on the **Score** tab: **79 / 100, grade B, "Mostly accessible"**.
* Walk the six bars. Each has a weight and a reason in plain English:
  | Category | Points | Why |
  |---|---|---|
  | Step-free reach | 20 / 30 | 2 of 3 places reachable without stairs from the Lobby |
  | Lift / ramp alternative | 5 / 10 | lifts were seen, but the walk doesn't show one linking the two levels ("verify on site") |
  | Stair safety | 15 / 15 | every stair segment has a registered handrail |
  | Clear width | 20 / 20 | the whole measured corridor is ≥ 0.9 m (median 2.52 m) |
  | Obstacle-free path | 10 / 10 | no obstacles near the walking line |
  | Wayfinding aids | 9 / 15 | only 58 % of the path is within 4 m of a sign or tactile paving |
* **"Where it is not accessible"**: click *Upper landing has no step-free route*. The 3D view flies to the red pin
  at the top of the stairs. The same pins appear on the floor plan.
* Point at the **floor plan**: the walls are no longer a blurry point cloud. They are **measured**. For every
  25 cm of the walk we find the nearest wall on each side among the scan's points at body height, which gives the
  corridor outline and its clear width. A test checks the measurement on a synthetic corridor (1.6 m wide with a 0.6 m pinch).
* Honesty line (judges like this): *"Unseen areas are reported as unknown, never as compliant. Lifts we saw but
  couldn't connect are flagged 'verify on site'."*
* Optional: Details tab → **Delete map** (it asks for confirmation; the source video is kept) and **Building location**
  (*Use my location*, which pins the map so phones open it automatically).

## 4. The phone app (2 min): hand the phone to a judge if you can

1. **Home**: "Where are we going today?" The **"You are here"** card uses GPS: the PCC map is pinned to this
   building, so the app opened the right map by itself. The card shows the score **79 · B** and "2 access issues".
2. Open the map: a **dark vector floor plan**, with walls measured from the scan, place labels, red/amber
   problem diamonds, a scale bar and the score ring in the corner.
3. Profile chip → **Wheelchair**. Tap **CG Lab**: "About 1 minute, 12 metres · Wheelchair / step-free", with a lime
   route and chevrons flowing toward the goal. **Start guidance**: big step cards, voice and haptics.
4. **From: Upper landing → CG Lab** as a wheelchair user: an honest **"No step-free route (a route exists only via
   stairs)"** instead of a dangerous one.
5. **Camera guide** (the wow moment): tap the camera, point the phone down the corridor.
   * The photo is matched against the scan's own keyframes: **DINOv2** shortlists, then **SIFT + RANSAC**
     verifies geometry. The result is "Near CC Lab · high" and the phone's heading.
   * The route is re-planned **on the phone** from that spot, and **lime AR chevrons** bend toward where to walk:
     "Turn left / Go straight · 9 m to CG Lab".
   * The detector reads the scene: "Door on your right", "Careful: Stairs ahead". **Describe** speaks a caption
     (Florence-2): *"a long hallway with a black and white checkered floor"*.
   * Numbers: 4 / 4 held-out video frames placed on exactly the right keyframe with high confidence, about 1 s each warm.
     A photo from a *different* building comes back "low", **not a confident wrong answer**.
6. **Ask Guide**: Groq (`openai/gpt-oss-120b`) gets "I'm at the spot the camera located near CC Lab…". It
   can only answer routes by **calling our real router as a tool**, so it cannot invent rooms or directions.
7. **Airplane mode** → everything above except the camera and Groq still works: map, routing, guidance, voice,
   and an on-device assistant. The map pack is **34 KB**; the phone re-downloads only when the content hash changes.

## 4b. Whole building: 4 floors, lift and stairs (60 s): web → Buildings

* One scan per floor, stacked into one building. The lift and stairwell are tagged with the same name on every
  floor, so they link automatically and the floors line up on the vertical lift shaft.
* Route tab: *From* building entrance → *To* a room on the 2nd floor, **Wheelchair** → "Take **Lift 1 up to the
  Second floor (2 floors)**". The 3D view draws the route up the shaft. **Walking** may take the stairs instead.
* Lifts & stairs → **Lift 1 → Out of service**: the wheelchair route becomes "no step-free route (Lift 1 out of
  service)". It's honest, and never sends a wheelchair user to the stairs.
* Phone: the building is one map with **G 1 2 3** floor buttons, floor-by-floor route and guidance that switches
  floor at the lift. It works offline, and the server and phone give word-for-word the same answers (parity test on
  8 cross-floor routes).
* How to record it: [BUILDING.md](BUILDING.md).

## 5. Why it's real (30 s)

* **Same answers offline and online.** The phone's router is a Dart port of the server's, and a parity test checks
  them **word for word** on 10 real requests (`mobile/test/router_parity_test.dart`).
* **Accessible by construction.** Large text (tested at 2× system text on a 360 dp phone with no overflow:
  the profile picker switches to a stacked layout), high-contrast mode, screen-reader labels, ≥ 48 dp targets,
  and lime on near-black (~15:1 contrast).
* **Tested.** 128 Python tests (API, pipeline control, scoring, camera localisation, map packs) and 16 Flutter tests.

## 6. Close (15 s)

> "One walkthrough video in, and out comes an accessibility audit with locations, a digital twin, and a phone
> guide that works without signal and tells you where you are just by looking. That's AccessMap."

---

## Judge Q&A, short answers

| Question | Answer |
|---|---|
| How accurate is the scale? | Gravity-aligned and **approximately metric** (camera height prior). Distances are labelled as estimates. Widths come from the scan's points, not a tape measure. |
| What if the camera guide is wrong? | It is keyframe-level, not 6-DoF. Geometric verification rejects look-alikes ("low" means we say so), and the user can always pick a start place by hand. |
| Indoor GPS? | GPS only picks the **building** (map pinned to lat/lng). Inside, the camera guide locates you, or you tap where you are. |
| Privacy? | Photos for the camera guide go only to your own server and are not stored. The Groq key stays server-side (`.env`, git-ignored). The phone never sees it. |
| Does it need the internet? | No. Maps, routing, guidance, voice and the basic assistant run on the phone. Only the camera guide and Groq need the server. |
| How is the score computed? | Six weighted categories from the scan (`vision/navigation/score.py`). Each comes with its reason, and the issues are located. Unknown counts as unknown, not as a pass. |
| Scaling to a campus? | Each walk becomes a map; buildings and floors are linked in Campus; phones sync only changed packs (ETag, about 34 KB each). |
| Who tags rooms? | Anyone, in the 3D Explorer: click a spot or walk there and press *Tag where I'm standing*. Tags re-sync to phones automatically. |

## Where things are (for technical judges)

| Feature | Code |
|---|---|
| Accessibility score + issues | `vision/navigation/score.py`, `GET /spatial/reconstructions/{id}/accessibility` |
| Measured walls / clear width | `vision/navigation/floor.py` |
| Camera localisation + scene hints | `vision/navigation/locate.py`, `POST /maps/{id}/locate` |
| Map packs, GPS pin, registry | `vision/navigation/mappack.py`, `api/mobile.py` |
| Groq assistant with router tool | `vision/assistant/groq.py` |
| Pause / resume / stop jobs | `vision/spatial/control.py`, `POST /spatial/jobs/{id}/{action}` |
| Delete a map | `DELETE /spatial/jobs/{id}` (Explorer → Details) |
| Multi-floor buildings (stacking, lifts, cross-floor routes) | `vision/navigation/building.py`, `api/building.py`, web `/buildings` |
| Phone app | `mobile/lib/` (map: `ui/widgets/plan_view.dart`, camera: `ui/screens/camera_guide.dart`) |
| Screenshots of every screen | `mobile/test_screens/goldens/` (regenerate: `flutter test test_screens --update-goldens`) |
