# Stage demo kit

Everything needed to demo AccessMap live, with no waiting on processing while you're on stage.

**The full judge-facing script (talk track, numbers, phone demo, Q&A) is in [PITCH.md](PITCH.md).**
**Recording a whole multi-floor building (lift, stairs) step by step: [BUILDING.md](BUILDING.md).**

| Path | What it is |
|---|---|
| `bundles/pcc/` | Finished 3D scan of the PCC block (stair landing → lobby → lab corridor) with tagged places: **Upper landing, Main staircase, Lobby (default start), CC Lab, CG Lab**. Includes the source video. |
| `floorplans/pcc_floor_sketch.png` | Architect-style sketch for the "upload your floor plan" moment. It's deliberately rotated and scaled, so you can show it being aligned. It was drawn from the scan, not taken from an official plan. |
| `load_demo.py` | Installs the bundles as reconstruction `demopcc` (safe to re-run; it never touches your own scans). |
| `export_demo.py` | Saves any finished scan, including its places and plans, as a bundle. Re-run it after re-tagging. |
| `make_sample_plan.py` | Re-draws the floor-plan sketch from a bundle. |
| `../samples/images/` | Ten real CC-licensed photos (stairs, ramps, lifts, tactile paving…) for the AI Studio part. |

> **Before the show:** the five demo places sit at approximate spots along the walk (from what is visible in
> the video). Open the Explorer, move each pin to its real spot (Edit → re-pick, or Walk mode → *Tag where I'm
> standing*), then run `export_demo.py demopcc pcc` so the bundle keeps your corrections.

## Setup (once, about 2 minutes)

```bash
.venv/Scripts/python demo/load_demo.py --reset
```
```bash
.venv/Scripts/uvicorn api.app:app --port 8000
```
```bash
pnpm --dir web dev
```

Optionally, pre-fill the Overview numbers by running the sample photos through the running API:

```bash
.venv/Scripts/python demo/load_demo.py --photos
```

Open these in tabs before you go on stage:

* http://localhost:3000/explorer?id=demopcc
* http://localhost:3000/navigation?id=demopcc
* http://localhost:3000/studio

## The 5-minute script

**1. The problem (20 s).** "Campus maps don't say *how* to get somewhere if you use a wheelchair or can't see
the signs. We turn a phone walkthrough into an accessible, navigable digital twin."

**2. AI Studio (60 s).** Drop `samples/images/stairs_oau_campus.jpg` and `ramp_arts.jpg`. Point at the boxes,
labels and accessibility attributes (handrail present, step count…). Then mention that the same engine runs on
video, and show the pcc video in the Video tab. Don't start a new reconstruction live: it takes about 5 minutes.

**3. 3D Explorer (90 s).** Open `explorer?id=demopcc`.
* **Orbit**, then press **Replay walk**: the camera flies the captured path.
* Switch to **Walk**: use W/A/S/D or the on-screen arrows to walk down the corridor. The mini-map shows where you are.
* Press **Tag a place**, click a doorway and name it *Seminar Hall*. It appears in 3D and on the plan straight away.
* **Floor plan** tab → *Upload plan* → `demo/floorplans/pcc_floor_sketch.png`. Drag it and use the rotate and
  scale sliders to line it up with the AI plan, then *Save alignment*. "The AI drew the left one from the scan;
  the right one is the architect's plan, now lined up."

**4. Wayfinding (90 s).** Open `navigation?id=demopcc`.
* Type **"Take me to the CG Lab"** with the **Wheelchair** profile → step-free route, turn-by-turn directions, the
  arrival hint "Double door - push to open", and the route drawn in 3D and on the plan. Press **Walk me there**.
* Change the start to **Upper landing** and ask again: *No step-free route*. "The only way down from here is the
  staircase, so it tells you, and you'd be pointed to the lift instead."
* Switch to **Low vision** for the same request: the route goes via the stairs with "hold the handrail; count the
  steps", and **Read aloud** speaks it.
* Try **"Where's the nearest lift?"** or the mic button, e.g. "How do I get from the computer centre to the
  graphics lab?" (aliases work).

**5. Close (20 s).** "One video in, an accessible map out: detection, 3D, floor plan and inclusive directions,
all running locally on a laptop GPU."

## If something goes wrong on stage

* **AI engine offline** pill: the API isn't running; start it with the uvicorn command above.
* **Reconstruction missing**: re-run `demo/load_demo.py`.
* **The 3D view is slow on a projector laptop**: switch the Layers → splat off; the walked path, places and route still show.
* **Mic doesn't work**: type the request instead. The free-text parser and the voice input use the same
  endpoint (`/navigation/{id}/ask`).
