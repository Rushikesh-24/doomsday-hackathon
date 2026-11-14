# Recording a whole building (4 floors, lift and stairs)

How to turn your college block (Ground + 3 floors) into one building that people can navigate from the
ground-floor entrance to any room, by lift or stairs, on the web and offline on the phone.

## 1. Record (one video per floor, about 1-2 min each)

Do this on **every floor**, Ground, 1, 2 and 3:

1. **Start at the lift door.** Stand facing the lift for 2-3 s, then turn and walk the corridor slowly with the
   phone upright at chest height. Walking slowly and steadily is what makes a good scan.
2. **Pass the stairwell landing** (where the stairs arrive on this floor). Pause there for a second.
3. Walk the whole corridor past the labs and classrooms, and **end back at the lift** if you can. Coming back to
   the start (a loop) makes the scan much more accurate.

You do **not** need to film the stairs or ride the lift: floors are joined by the lift and stairs you tag in
step 3. You can film a stair flight separately if you want it on screen; it isn't used for routing.

Upload each video in **AI Studio → Walkthrough video** (the *fast* preset is enough; *balanced* looks nicer in 3D).
The four jobs run one after another; you can pause, stop or delete them in the processing queue.

### Stairs videos (what the PCCE building uses)

You can also film **the climb itself** ("ground to first floor steps", "second to third floor stairs"). Add such a
video to the floor it **starts** on, with **ends on** set to the floor it finishes on. AccessMap reads the floor
change from the camera height (flat, then climbing, then flat again). The climb becomes a "Take the stairs up to the
First floor" step that wheelchair routes cannot use. A floor can have several walks (left and right corridors).

Press **Stitch scans** (Floors tab). Frames of each video are matched against the others, and walks that overlap are
joined and lined up: the bottom of a stairs video with the corridor it starts from, the top with the corridor it
reaches, and left / right walks where they cross. Stairwells look identical on every floor, so the matcher never
trusts a match between different floors. A stairs video that starts where the previous one ended is stacked
directly above it (same stairwell).

If a walk shows *not linked*, give one spot the **same place name** in both walks (e.g. `Main corridor junction`),
or place it by hand with *Adjust placement*.

## 2. Make the building (web → Buildings)

1. **Create**: name it (e.g. *Main Block*) and choose **4 floors**. This creates Ground, First, Second and Third
   floor.
2. On each floor card, **choose that floor's scan**. Floor height defaults to 3.5 m (change it if your floors are
   taller).

## 3. Tag rooms, the lift and the stairs (web → 3D Explorer, once per floor)

On each floor's scan (the card's link *Tag rooms, lift and stairs on this floor*):

| What | Kind | Name |
|---|---|---|
| The lift door | **Lift** | the **same name on every floor**, e.g. `Lift 1` |
| Where the stairs arrive | **Stairwell (landing)** | the **same name on every floor**, e.g. `Main stairs` |
| Labs, classrooms, office, toilets… | Lab / Classroom / … | `CG Lab`, `Classroom 204`… |
| The main entrance (Ground only) | Entrance / exit | `Main entrance`, and switch on *Default starting point* |

Use **Stairwell (landing)**, not *Stairs*, for the landing. *Stairs* marks the corridor around it as stairs,
which would block wheelchair routes going past it.

## 4. Link the floors (web → Buildings → Lifts & stairs)

Press **Link lifts & stairs automatically**. Every lift / stairwell name found on 2+ floors becomes one connector:

* **Lift 1**: stops at G · 1 · 2 · 3. If it skips a floor, set that floor to *does not stop here*. Use
  **Out of service** when it is broken: wheelchair routes then honestly say there is no step-free route.
* **Main stairs**: connects every floor it was tagged on.
* A second lift or a back staircase: tag it the same way (e.g. `Lift 2`), or add it by hand on this tab.

The floors are stacked automatically using the lift and stairs landings, because a lift shaft is vertical. Check
the 3D view: if a floor looks rotated or shifted, open its card → *Adjust placement by hand*.

## 5. Try it

* **Web → Buildings → Route**: *From* = building entrance, *To* = `CG Lab · Second floor`, profile **Wheelchair**
  → "Go straight … Take **Lift 1 up to the Second floor (2 floors)** … You have arrived at CG Lab." The 3D view
  draws the route up the lift shaft.
* Switch to **Walking**: short trips may take the stairs instead.
* Set **Lift 1 → Out of service**: the wheelchair route becomes "no step-free route (Lift 1 out of service)".
* **Phone**: pull to sync. The building appears as one map (*4 floors*). Use the floor buttons **G 1 2 3** on the
  left of the map. The route is drawn floor by floor, each step shows its floor (*FLOOR G → 2* for the lift), and
  guidance switches floor after the lift step. It all works offline.
* Typed requests understand floors: "Toilet on the 2nd floor", "Office on the ground floor".

## On stage (30 s)

> "Every floor is one walkthrough video. We tag the lift and stairwell with the same name on each floor, so the
> building stacks itself: the lift shaft is vertical, so the floors line up on it. Now a wheelchair user at the
> ground-floor entrance asks for the CG Lab on the second floor. AccessMap takes them through the lift, never the
> stairs. If the lift breaks, we mark it out of service and the app says so honestly instead of sending them to
> the stairs."
