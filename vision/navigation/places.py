"""User-named places, 2D floor plans and human-friendly wayfinding for a reconstruction.

Places ("CC Lab", "CG Lab", "Main entrance", ...) are points the user tags in the 3D Explorer. They live in
`<job>/places.json` next to the reconstruction (so a demo bundle carries them) and are attached to the
navigation graph exactly like registered facilities: each joins its nearest walked-path node.

Floor plan (`<job>/plan/`):
  generated.png  top-down density map of the sparse COLMAP points in the gravity-aligned local frame
                 (+Y up, so the plan is the x/z plane; image x = local x, image y = local z)
  uploaded.*     an architectural plan the user supplies, aligned onto the generated one by a similarity
                 transform (scale, rotation, offset) the user adjusts and saves
  plan.json      bounds + pixel scale of the generated map, uploaded plan file name + transform

Directions turn the graph route into walking instructions: headings on the ground plane give
left/right turns, edge tags give stairs/ramp/obstacle notes, nearby facilities and places become
landmarks ("pass the elevator on your left"), and the accessibility profile adds the help that person
needs. Distances are metres only when the reconstruction has a metric scale.
"""
from __future__ import annotations

import json
import math
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_lock = threading.RLock()

KINDS = {
    "lab": "Lab", "classroom": "Classroom", "office": "Office", "entrance": "Entrance / exit", "toilet": "Toilet",
    "lift": "Lift", "stairs": "Stairs", "stairwell": "Stairwell (landing)", "ramp": "Ramp", "library": "Library", "canteen": "Canteen",
    "help_desk": "Help desk", "meeting_point": "Meeting point", "other": "Other",
}
# words people say for facility classes (for "where is the nearest lift")
CLASS_WORDS = {
    "elevator": ["lift", "elevator"], "staircase": ["stairs", "staircase", "stairway"], "ramp": ["ramp"],
    "door": ["door"], "entrance_exit": ["entrance", "exit", "way out", "gate"], "toilet_sign": ["toilet", "restroom",
    "washroom", "bathroom"], "handrail": ["handrail", "railing"], "signboard": ["sign", "signboard", "notice"],
    "corridor": ["corridor", "hallway"], "tactile_paving": ["tactile", "tactile paving"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ------------------------------------------------------------------ places
def _places_path(job: Path) -> Path:
    return job / "places.json"


def list_places(job: Path) -> list[dict]:
    p = _places_path(job)
    if not p.exists():
        return []
    return json.loads(p.read_text())["places"]


def _write(job: Path, places: list[dict]) -> None:
    _places_path(job).write_text(json.dumps({"places": places, "updated": _now()}, indent=1))


def _clean(body: dict, partial: bool = False) -> dict:
    out = {}
    if "name" in body or not partial:
        name = str(body.get("name") or "").strip()
        if not name or len(name) > 60:
            raise ValueError("name is required (1-60 characters)")
        out["name"] = name
    if "kind" in body or not partial:
        kind = body.get("kind") or "other"
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {list(KINDS)}")
        out["kind"] = kind
    if "position" in body or not partial:
        pos = body.get("position")
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3 and all(isinstance(v, (int, float)) and
                                                                         math.isfinite(v) for v in pos)):
            raise ValueError("position must be [x, y, z] in the reconstruction's local frame")
        out["position"] = [round(float(v), 4) for v in pos]
    for k, lim in (("description", 400), ("floor", 40)):
        if k in body:
            out[k] = (str(body[k]).strip()[:lim] or None) if body[k] is not None else None
    if "aliases" in body:
        al = body["aliases"] or []
        if isinstance(al, str):
            al = [a.strip() for a in al.split(",")]
        out["aliases"] = [a for a in (str(x).strip()[:40] for x in al) if a][:8]
    if "access" in body:
        a = body["access"] or {}
        out["access"] = {"step_free": a.get("step_free"), "accessible_toilet": a.get("accessible_toilet"),
                         "notes": (str(a.get("notes") or "").strip()[:300] or None)}
    if "is_start" in body:
        out["is_start"] = bool(body["is_start"])
    return out


def add_place(job: Path, body: dict) -> dict:
    p = {"id": "place-" + uuid.uuid4().hex[:8], "description": None, "aliases": [], "floor": None,
         "access": {"step_free": None, "accessible_toilet": None, "notes": None}, "is_start": False,
         **_clean(body), "created": _now()}
    with _lock:
        places = list_places(job)
        if any(_norm(x["name"]) == _norm(p["name"]) for x in places):
            raise ValueError(f"a place named '{p['name']}' already exists")
        if p["is_start"]:
            for x in places:
                x["is_start"] = False
        places.append(p)
        _write(job, places)
    return p


def update_place(job: Path, pid: str, body: dict) -> dict:
    with _lock:
        places = list_places(job)
        cur = next((x for x in places if x["id"] == pid), None)
        if cur is None:
            raise KeyError(f"place {pid} not found")
        ch = _clean(body, partial=True)
        if "name" in ch and any(_norm(x["name"]) == _norm(ch["name"]) and x["id"] != pid for x in places):
            raise ValueError(f"a place named '{ch['name']}' already exists")
        if ch.get("is_start"):
            for x in places:
                x["is_start"] = False
        cur.update(ch, updated=_now())
        _write(job, places)
    return cur


def delete_place(job: Path, pid: str) -> None:
    with _lock:
        places = list_places(job)
        rest = [x for x in places if x["id"] != pid]
        if len(rest) == len(places):
            raise KeyError(f"place {pid} not found")
        _write(job, rest)


# ------------------------------------------------------------------ graph integration
# user-tagged place kinds that behave like detected facilities (routing tags + "nearest X" requests)
# "stairwell" = the landing / door of a staircase that links floors in a building: it does NOT mark the corridor
# around it as stairs (only "stairs" does), so wheelchair routes can still pass it.
KIND_TO_CLASS = {"lift": "elevator", "stairs": "staircase", "stairwell": "staircase", "ramp": "ramp", "entrance": "entrance_exit",
                 "toilet": "toilet_sign"}


def attach_places(graph: dict, job: Path) -> dict:
    """Add place nodes (each joined to its nearest walked-path node) to a graph from navigation.graph.build.
    Places tagged as Stairs / Ramp also tag the walked-path edges within `radius` of them (default 2 m, or
    3 path steps without metric scale), so a manually marked staircase blocks wheelchair routes exactly like a
    detected one. Their nodes carry the matching facility class so 'nearest lift' finds tagged lifts too."""
    cams = [n for n in graph["nodes"] if n["kind"] == "trajectory"]
    P = np.array([n["pos"] for n in cams])
    graph["places"] = []
    med = graph.get("median_step") or 1e-6
    default_r = 2.0 if graph.get("units") == "m" else 3 * med
    cam_pos = {n["id"]: np.array(n["pos"]) for n in cams}
    for pl in list_places(job):
        p = np.array(pl["position"])
        d = np.linalg.norm(P - p, axis=1)
        k = int(np.argmin(d))
        node = {"id": pl["id"], "kind": "place", "name": pl["name"], "place_kind": pl["kind"],
                "pos": pl["position"], "attached_to": cams[k]["id"]}
        if pl["kind"] in KIND_TO_CLASS:
            node["class_name"] = KIND_TO_CLASS[pl["kind"]]
        graph["nodes"].append(node)
        graph["edges"].append({"a": cams[k]["id"], "b": pl["id"], "length": float(d[k]), "tags": ["approach"],
                               "facilities": []})
        if pl["kind"] in ("stairs", "ramp"):
            tag = "stairs" if pl["kind"] == "stairs" else "ramp"
            r = float(pl.get("radius") or default_r)
            for e in graph["edges"]:
                if e["a"] in cam_pos and e["b"] in cam_pos:
                    mid = (cam_pos[e["a"]] + cam_pos[e["b"]]) / 2
                    if np.linalg.norm(mid - p) <= r and tag not in e["tags"]:
                        e["tags"] = sorted({*e["tags"], tag})
                        e["facilities"] = [*e["facilities"], pl["id"]]
        graph["places"].append({**pl, "attached_to": cams[k]["id"], "offset_from_path": round(float(d[k]), 3)})
    return graph


def resolve(graph: dict, ref: str) -> str:
    """Node id, place name/alias (case/space-insensitive) or facility class -> a goal accepted by graph.route."""
    ids = {n["id"] for n in graph["nodes"]}
    if ref in ids:
        return ref
    key = _norm(ref)
    if key in ("start", "videostart", "entrance0", "begin"):
        st = next((p for p in graph.get("places", []) if p.get("is_start")), None)
        return st["id"] if st else "cam0"
    for p in graph.get("places", []):
        if key == _norm(p["name"]) or key in (_norm(a) for a in p.get("aliases", [])):
            return p["id"]
    classes = {n.get("class_name") for n in graph["nodes"] if n.get("class_name")}
    if ref in classes:
        return ref
    for cls, words in CLASS_WORDS.items():
        if cls in classes and key in (_norm(w) for w in words):
            return cls
    raise KeyError(f"no place or facility called '{ref}' in this reconstruction")


def parse_request(graph: dict, text: str) -> dict:
    """Free text ('how do I get from CC lab to the CG lab?') -> {'start': ref|None, 'goal': ref}."""
    t = " " + text.lower() + " "
    t_n = _norm(text)
    names = []   # (normalised needle, ref, display)
    for p in graph.get("places", []):
        for nm in [p["name"], *p.get("aliases", [])]:
            names.append((_norm(nm), p["id"], p["name"]))
    classes = {n.get("class_name") for n in graph["nodes"] if n.get("class_name")}
    for cls, words in CLASS_WORDS.items():
        if cls in classes:
            for w in words:
                names.append((_norm(w), cls, cls.replace("_", " ")))
    hits = []
    for needle, ref, disp in sorted(names, key=lambda x: -len(x[0])):
        if not needle:
            continue
        i = t_n.find(needle)
        if i >= 0 and not any(i < h[0] + len(h[1]) and h[0] < i + len(needle) for h in hits):
            hits.append((i, needle, ref, disp))
    hits.sort()
    if not hits:
        raise KeyError("I could not find a known place in that request. Tagged places: " +
                       (", ".join(p["name"] for p in graph.get("places", [])) or "none yet"))
    # "from A to B" / "A to B": first mention is the start only when the text says so
    from_idx = max(t.find(" from "), t.find(" i am at "), t.find(" i'm at "), t.find(" im at "))
    if len(hits) >= 2 and from_idx >= 0:
        return {"start": hits[0][2], "goal": hits[-1][2], "start_name": hits[0][3], "goal_name": hits[-1][3]}
    return {"start": None, "goal": hits[-1][2], "start_name": None, "goal_name": hits[-1][3]}


# ------------------------------------------------------------------ directions
def _ground(v) -> np.ndarray:
    return np.array([v[0], v[2]], dtype=float)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.atan2(_cross(a, b), float(np.dot(a, b))))


def _dist_text(d: float, unit: str, total: float) -> str:
    if unit == "m":
        if d < 1.5:
            return "a few steps"
        return f"about {round(d)} m" if d >= 3 else f"about {d:.1f} m"
    frac = d / total if total else 0
    return "a short way" if frac < 0.15 else ("part of the way" if frac < 0.45 else "most of the way")


HELP = {
    "wheelchair": {
        "intro": "Step-free route: stairs are excluded.",
        "ramp": "Ramp ahead - take it slowly; its gradient has not been verified.",
        "obstacle": "Obstacle reported near the path - keep to the clear side.",
        "door": "Door ahead - it may be heavy; ask for help if it has no automatic opener.",
        "arrive": "If the entrance has a step, use the bell or ask staff for the portable ramp.",
    },
    "low_vision": {
        "intro": "Route prefers handrails and avoids reported obstacles. Directions can be read aloud.",
        "stairs": "Stairs ahead - hold the handrail; count the steps.",
        "stairs_no_handrail": "Stairs ahead with no handrail found - go slowly or ask for assistance.",
        "obstacle": "Obstacle reported near the path - slow down and use your cane or guide.",
        "ramp": "Slope ahead.",
        "arrive": "Listen for the room - ask nearby staff to confirm the door.",
    },
    "standard": {"intro": "Shortest walked route.", "arrive": ""},
}
NEAR_CLASSES = {"elevator", "staircase", "ramp", "door", "entrance_exit", "toilet_sign", "signboard",
                "tactile_paving", "handrail"}


def directions(graph: dict, rt: dict) -> dict:
    """Add `directions` (turn-by-turn walking steps with landmarks and profile help) to a feasible route."""
    if not rt.get("feasible"):
        return rt
    nodes = {n["id"]: n for n in graph["nodes"]}
    ids = rt["nodes"]
    unit = rt["units"]
    total = rt["length"] or 1e-9
    edges = {}
    for e in graph["edges"]:
        edges[(e["a"], e["b"])] = e
        edges[(e["b"], e["a"])] = e
    prof = HELP.get(rt["profile"], HELP["standard"])
    pts = [np.array(nodes[i]["pos"], float) for i in ids]
    med = graph.get("median_step") or 1e-6
    landmarks = [n for n in nodes.values() if n["id"] not in (ids[0], ids[-1]) and (
        (n["kind"] == "facility" and n.get("class_name") in NEAR_CLASSES) or n["kind"] == "place")]
    used_landmarks: set[str] = set()
    out = []
    start_name = nodes[ids[0]].get("name") or ("the start of the walk" if ids[0] == "cam0" else "your position")
    goal = nodes[ids[-1]]
    goal_name = goal.get("name") or "the " + (goal.get("class_name") or "destination").replace("_", " ")
    out.append({"kind": "start", "text": f"Start at {start_name}. {prof['intro']}".strip(), "at": ids[0]})
    # walk legs: split at turns sharper than 35 deg (ground plane), skipping zero-length hops
    seg_dir, run_len, run_start, run_tags = None, 0.0, 0, set()
    legs = []
    for k in range(len(ids) - 1):
        a, b = pts[k], pts[k + 1]
        d = _ground(b - a)
        L = float(np.linalg.norm(b - a))
        e = edges.get((ids[k], ids[k + 1]), {"tags": []})
        # the hop between a place/facility and its nearest path point says nothing about walking direction
        if np.linalg.norm(d) < 1e-9 or "approach" in e["tags"]:
            run_len += L
            run_tags |= set(e["tags"])
            continue
        d = d / np.linalg.norm(d)
        if seg_dir is not None and abs(_angle(seg_dir, d)) > 35 and run_len > 0.3 * med:
            legs.append({"from": run_start, "to": k, "dir": seg_dir, "length": run_len, "tags": run_tags,
                         "turn": _angle(seg_dir, d)})
            run_len, run_start, run_tags = 0.0, k, set()
            seg_dir = d
        elif seg_dir is None:
            seg_dir = d
        else:   # smooth the running heading
            seg_dir = seg_dir * run_len + d * L
            seg_dir = seg_dir / (np.linalg.norm(seg_dir) or 1)
        run_len += L
        run_tags |= set(e["tags"])
    legs.append({"from": run_start, "to": len(ids) - 1, "dir": seg_dir if seg_dir is not None else np.array([1.0, 0]),
                 "length": run_len, "tags": run_tags, "turn": None})
    for li, leg in enumerate(legs):
        txt = f"Go straight for {_dist_text(leg['length'], unit, total)}"
        notes = []
        for tag in ("stairs_no_handrail", "stairs", "ramp", "obstacle"):
            if tag in leg["tags"]:
                if tag == "stairs" and "stairs_no_handrail" in leg["tags"]:
                    continue
                notes.append(prof.get(tag) or {"stairs": "Stairs on this stretch.", "ramp": "Ramp on this stretch.",
                                               "obstacle": "Obstacle reported near the path.",
                                               "stairs_no_handrail": "Stairs with no handrail found."}[tag])
        # landmarks within ~1.5 path steps of this leg, reported with their side
        seen = []
        for k in range(leg["from"], leg["to"] + 1):
            for lm in landmarks:
                if lm["id"] in used_landmarks:
                    continue
                v = np.array(lm["pos"]) - pts[k]
                if np.linalg.norm(v) < 1.5 * med:
                    side = _cross(leg["dir"], _ground(v))
                    where = " on your right" if side > 0.15 * med else (" on your left" if side < -0.15 * med else "")
                    nm = lm.get("name") or "the " + lm.get("class_name", "").replace("_", " ")
                    seen.append(f"{nm}{where}")
                    used_landmarks.add(lm["id"])
                    if lm.get("class_name") == "door" and prof.get("door"):
                        notes.append(prof["door"])
        if seen:
            txt += ", passing " + ", ".join(seen[:3])
        out.append({"kind": "walk", "text": txt + ".", "length": round(leg["length"], 3), "notes": notes,
                    "at": ids[leg["from"]], "to": ids[leg["to"]]})
        if leg["turn"] is not None:
            a = leg["turn"]
            word = ("Turn around" if abs(a) > 140 else ("Turn right" if a > 0 else "Turn left") if abs(a) > 60
                    else ("Bear right" if a > 0 else "Bear left"))
            out.append({"kind": "turn", "text": word + ".", "angle": round(a), "at": ids[leg["to"]]})
    arrive = f"You have arrived at {goal_name}."
    gp = next((p for p in graph.get("places", []) if p["id"] == goal["id"]), None)
    extra = []
    if gp:
        acc = gp.get("access") or {}
        if gp.get("description"):
            extra.append(gp["description"])
        if acc.get("step_free") is False and rt["profile"] == "wheelchair":
            extra.append("Note: this place is marked as NOT step-free.")
        if acc.get("notes"):
            extra.append(acc["notes"])
    if prof.get("arrive"):
        extra.append(prof["arrive"])
    extra = [x if x.rstrip().endswith((".", "!", "?")) else x.rstrip() + "." for x in extra if x and x.strip()]
    out.append({"kind": "arrive", "text": " ".join([arrive, *extra]).strip(), "at": ids[-1]})
    rt["directions"] = out
    rt["start_name"], rt["goal_name"] = start_name, goal_name
    rt["spoken"] = " ".join(s["text"] + (" " + " ".join(s.get("notes", [])) if s.get("notes") else "") for s in out)
    return rt


# ------------------------------------------------------------------ floor plan
def _plan_dir(job: Path) -> Path:
    d = job / "plan"
    d.mkdir(exist_ok=True)
    return d


def plan_info(job: Path) -> dict:
    p = _plan_dir(job) / "plan.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_plan(job: Path, info: dict) -> None:
    (_plan_dir(job) / "plan.json").write_text(json.dumps(info, indent=1))


def _splat_centres(path: Path) -> np.ndarray | None:
    """x, y, z of opaque Gaussians from a binary little-endian 3DGS PLY (all-float properties)."""
    if not path.is_file():
        return None
    raw = path.read_bytes()
    marker = b"end_header" + bytes([10])
    end = raw.find(marker)
    if end < 0:
        return None
    head = raw[:end].decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in head:
        return None
    n = next(int(l.split()[2]) for l in head if l.startswith("element vertex"))
    props = [l.split()[2] for l in head if l.startswith("property float")]
    if len([l for l in head if l.startswith("property")]) != len(props):
        return None
    a = np.frombuffer(raw, dtype="<f4", count=n * len(props), offset=end + len(marker)).reshape(n, len(props))
    xyz = a[:, [props.index("x"), props.index("y"), props.index("z")]]
    if "opacity" in props:
        xyz = xyz[1 / (1 + np.exp(-a[:, props.index("opacity")])) > 0.3]
    return xyz[np.isfinite(xyz).all(1)].astype(np.float64)


def generate_plan(job: Path, max_px: int = 1400) -> dict:
    """Top-down density map of the sparse points (local frame) - walls and furniture show as dark strokes."""
    from PIL import Image, ImageFilter

    traj = json.loads((job / "trajectory.json").read_text())
    al = json.loads((job / "alignment.json").read_text())
    T = np.array(al["T_local_from_sfm"], float)
    cams = np.array([c["center_local"] for c in traj["cameras"] if c.get("center_local")], float)
    xyz = np.load(job / "sfm/models/0/points.npz")["xyz"]
    P = (np.c_[xyz, np.ones(len(xyz))] @ T.T)[:, :3]
    sp = _splat_centres(job / "splat/splat_local.ply")      # already in the local frame; much denser
    if sp is not None and len(sp):
        P = np.vstack([P, sp])
    # keep the storey the walk happened on: points within a band around camera height
    cy = float(np.median(cams[:, 1]))
    step = np.linalg.norm(np.diff(cams, axis=0), axis=1)
    span = max(float(np.ptp(cams[:, [0, 2]], axis=0).max()), 1e-3)
    band = max(span * 0.35, 6 * float(np.median(step)) if len(step) else span * 0.35)
    keep = np.abs(P[:, 1] - cy) < band
    P = P[keep]
    lo = np.percentile(P[:, [0, 2]], 1, axis=0) if len(P) > 50 else P[:, [0, 2]].min(0)
    hi = np.percentile(P[:, [0, 2]], 99, axis=0) if len(P) > 50 else P[:, [0, 2]].max(0)
    lo = np.minimum(lo, cams[:, [0, 2]].min(0))
    hi = np.maximum(hi, cams[:, [0, 2]].max(0))
    pad = 0.08 * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    ppu = max_px / float((hi - lo).max())
    W, H = int(math.ceil((hi[0] - lo[0]) * ppu)), int(math.ceil((hi[1] - lo[1]) * ppu))
    ix = ((P[:, 0] - lo[0]) * ppu).astype(int)
    iz = ((P[:, 2] - lo[1]) * ppu).astype(int)
    ok = (ix >= 0) & (ix < W) & (iz >= 0) & (iz < H)
    den = np.zeros((H, W), np.float32)
    np.add.at(den, (iz[ok], ix[ok]), 1.0)
    img = Image.fromarray(np.clip(den / max(np.percentile(den[den > 0], 90), 1) * 255, 0, 255).astype(np.uint8))
    r = max(1, int(round(max(W, H) / 260)))
    img = img.filter(ImageFilter.MaxFilter(2 * r + 1)).filter(ImageFilter.GaussianBlur(r))
    a = np.asarray(img, np.float32) / 255.0
    ink = np.array([30, 41, 59], np.float32)        # slate-800 strokes on a paper background
    paper = np.array([248, 250, 252], np.float32)
    rgb = paper[None, None] * (1 - a[..., None] * 0.85) + ink[None, None] * (a[..., None] * 0.85)
    Image.fromarray(rgb.astype(np.uint8)).save(_plan_dir(job) / "generated.png")
    info = plan_info(job)
    info["generated"] = {"file": "generated.png", "width": W, "height": H, "px_per_unit": ppu,
                         "origin": [float(lo[0]), float(lo[1])], "axes": "image x = local x, image y = local z",
                         "points_used": int(ok.sum()), "units": traj["units"], "created": _now()}
    info.setdefault("uploaded", None)
    _save_plan(job, info)
    return info


def ensure_plan(job: Path) -> dict:
    info = plan_info(job)
    if not info.get("generated") or not (_plan_dir(job) / "generated.png").exists():
        info = generate_plan(job)
    return info


def save_uploaded_plan(job: Path, data: bytes, filename: str) -> dict:
    from PIL import Image, ImageOps
    import io
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as e:
        raise ValueError(f"not a readable image: {e}")
    im.thumbnail((3000, 3000))
    d = _plan_dir(job)
    for old in d.glob("uploaded.*"):
        old.unlink()
    im.save(d / "uploaded.png")
    info = ensure_plan(job)
    g = info["generated"]
    s = min(g["width"] / im.width, g["height"] / im.height)
    info["uploaded"] = {"file": "uploaded.png", "name": filename, "width": im.width, "height": im.height,
                        "transform": {"scale": s, "rotation": 0.0, "x": 0.0, "y": 0.0, "opacity": 0.6},
                        "created": _now()}
    _save_plan(job, info)
    return info


def set_plan_transform(job: Path, tr: dict) -> dict:
    info = ensure_plan(job)
    if not info.get("uploaded"):
        raise ValueError("upload a floor plan image first")
    cur = info["uploaded"]["transform"]
    for k in ("scale", "rotation", "x", "y", "opacity"):
        if k in tr:
            v = float(tr[k])
            if not math.isfinite(v):
                raise ValueError(f"{k} must be a number")
            cur[k] = v
    cur["scale"] = min(max(cur["scale"], 1e-3), 1e3)
    cur["opacity"] = min(max(cur["opacity"], 0.0), 1.0)
    _save_plan(job, info)
    return info


def remove_uploaded_plan(job: Path) -> dict:
    info = ensure_plan(job)
    for old in _plan_dir(job).glob("uploaded.*"):
        old.unlink()
    info["uploaded"] = None
    _save_plan(job, info)
    return info
