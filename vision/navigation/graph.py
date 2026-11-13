"""Accessibility-aware navigation over a real reconstruction.

Graph: the walked camera trajectory (registered keyframes, in order) is the only space known to be
traversable; consecutive keyframes are joined, revisits (non-consecutive keyframes closer than 1.5x the
median step) add loop edges, and each registered facility is attached to its nearest trajectory node.
Edges whose midpoint lies inside an (expanded) staircase / ramp box, or near an obstacle, are tagged.
Profiles set per-tag costs; `inf` means forbidden. Units follow the reconstruction's scale status
(metres only if the reconstruction is metric/approximate; otherwise relative SfM units).
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

PROFILES = {
    "standard": {"stairs": 1.0, "ramp": 1.0, "obstacle": 1.2, "label": "Standard walking"},
    "wheelchair": {"stairs": math.inf, "ramp": 1.0, "obstacle": 4.0, "label": "Wheelchair / step-free"},
    "low_vision": {"stairs": 1.5, "stairs_no_handrail": 4.0, "ramp": 1.0, "obstacle": 6.0,
                   "label": "Low vision (prefer handrails, avoid obstacles)"},
}


def build(job_dir: Path) -> dict:
    traj = json.loads((job_dir / "trajectory.json").read_text())
    fac = json.loads((job_dir / "facilities.json").read_text())["facilities"]
    cams = [c for c in traj["cameras"] if c.get("center_local")]
    if len(cams) < 2:
        raise ValueError("reconstruction has fewer than 2 registered cameras")
    P = np.array([c["center_local"] for c in cams])
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    med = float(np.median(step)) or 1e-6
    nodes = [{"id": f"cam{i}", "kind": "trajectory", "pos": P[i].round(4).tolist(), "frame": c["frame"], "t": c["t"]}
             for i, c in enumerate(cams)]
    edges = []
    for i in range(len(cams) - 1):
        if step[i] < 8 * med:                       # do not bridge tracking gaps
            edges.append((i, i + 1))
    for i in range(len(cams)):
        d = np.linalg.norm(P - P[i], axis=1)
        for j in np.nonzero((d < 1.5 * med))[0]:
            if j > i + 2:
                edges.append((i, int(j)))
    reg = [f for f in fac if f.get("position_local")]
    boxes = {"stairs": [], "ramp": []}
    rails = [(np.array(f["bbox_local"][0]), np.array(f["bbox_local"][1])) for f in reg
             if f["class_name"] == "handrail" and f.get("bbox_local")]
    obstacles = [np.array(f["position_local"]) for f in reg if f["class_name"] == "obstacle"]
    for f in reg:
        key = {"staircase": "stairs", "ramp": "ramp"}.get(f["class_name"])
        if key and f.get("bbox_local"):
            lo, hi = np.array(f["bbox_local"][0]), np.array(f["bbox_local"][1])
            pad = 0.5 * med
            boxes[key].append((lo - pad, hi + pad, f["facility_id"]))
    E = []
    for i, j in edges:
        mid = (P[i] + P[j]) / 2
        tags, refs, stair_boxes = [], [], []
        for key, bl in boxes.items():
            for lo, hi, fid in bl:
                if np.all(mid >= lo) and np.all(mid <= hi):
                    tags.append(key)
                    refs.append(fid)
                    if key == "stairs":
                        stair_boxes.append((lo, hi))
        # a staircase "has a handrail" if a registered handrail's 3D box overlaps the staircase box
        if stair_boxes and not any(np.all(np.minimum(hi, rh) >= np.maximum(lo, rl))
                                   for lo, hi in stair_boxes for rl, rh in rails):
            tags.append("stairs_no_handrail")
        if any(np.linalg.norm(o - mid) < 1.5 * med for o in obstacles):
            tags.append("obstacle")
        E.append({"a": f"cam{i}", "b": f"cam{j}", "length": float(np.linalg.norm(P[i] - P[j])),
                  "tags": sorted(set(tags)), "facilities": refs})
    for f in reg:
        p = np.array(f["position_local"])
        d = np.linalg.norm(P - p, axis=1)
        k = int(np.argmin(d))
        nodes.append({"id": f["facility_id"], "kind": "facility", "class_name": f["class_name"],
                      "pos": p.round(4).tolist(), "confidence": f.get("registration_confidence")})
        E.append({"a": f"cam{k}", "b": f["facility_id"], "length": float(d[k]), "tags": ["approach"],
                  "facilities": [f["facility_id"]]})
    return {"nodes": nodes, "edges": E, "units": traj["units"], "scale_status": traj["scale_status"],
            "median_step": med, "facilities_unregistered": len(fac) - len(reg),
            "note": "graph covers only the walked path of this video; unexplored space is unknown, not blocked"}


def _cost(e: dict, prof: dict) -> float:
    m = 1.0
    for t in e["tags"]:
        if t in prof:
            m = max(m, prof[t]) if math.isfinite(prof[t]) else math.inf
    return e["length"] * m


def route(graph: dict, start: str, goal: str, profile: str = "standard") -> dict:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {list(PROFILES)}")
    prof = PROFILES[profile]
    ids = {n["id"]: n for n in graph["nodes"]}
    if goal not in ids:   # class name: nearest facility of that class (resolved by cost below)
        targets = {n["id"] for n in graph["nodes"] if n.get("class_name") == goal}
        if not targets:
            raise KeyError(f"no registered '{goal}' in this reconstruction")
    else:
        targets = {goal}
    if start not in ids:
        raise KeyError(f"unknown start node {start}")
    adj: dict[str, list] = {}
    for e in graph["edges"]:
        c = _cost(e, prof)
        adj.setdefault(e["a"], []).append((e["b"], c, e))
        adj.setdefault(e["b"], []).append((e["a"], c, e))
    dist, prev = {start: 0.0}, {}
    pq = [(0.0, start)]
    reached = None
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        if u in targets:
            reached = u
            break
        for v, c, e in adj.get(u, []):
            if not math.isfinite(c):
                continue
            nd = d + c
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, (u, e)
                heapq.heappush(pq, (nd, v))
    if reached is None:
        # explain: is it reachable when restrictions are lifted?
        alt = route(graph, start, goal, "standard") if profile != "standard" else None
        return {"feasible": False, "profile": profile, "reason": (
            f"no route satisfying '{prof['label']}' within this reconstruction" +
            (" (a route exists only via stairs)" if alt and alt.get("feasible") else "")), "path": [], "steps": []}
    path, segs = [reached], []
    while path[-1] != start:
        u, e = prev[path[-1]]
        segs.append(e)
        path.append(u)
    path.reverse()
    segs.reverse()
    length = sum(e["length"] for e in segs)
    steps, cur = [], None
    for e in segs:
        key = ",".join(t for t in e["tags"] if t != "approach") or "path"
        if cur and cur["kind"] == key:
            cur["length"] += e["length"]
        else:
            cur = {"kind": key, "length": e["length"]}
            steps.append(cur)
    unit = "m" if graph["scale_status"] != "relative" else "units"
    words = {"path": "Continue along the walked path", "stairs": "Stairs (handrail registered alongside)",
             "stairs,stairs_no_handrail": "Stairs - no handrail registered in 3D (verify on site)", "ramp": "Ramp",
             "obstacle": "Obstacle reported near the path"}
    for s in steps:
        s["instruction"] = f"{words.get(s['kind'], s['kind'])} for {s['length']:.1f} {unit}"
        s["length"] = round(s["length"], 3)
    return {"feasible": True, "profile": profile, "profile_label": prof["label"], "goal": reached,
            "path": [ids[n]["pos"] for n in path], "nodes": path, "length": round(length, 3), "units": unit,
            "scale_status": graph["scale_status"], "steps": steps,
            "warnings": sorted({t for e in segs for t in e["tags"] if t in ("stairs", "stairs_no_handrail", "obstacle")})}
