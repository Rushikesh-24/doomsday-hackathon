"""Accessibility score of one scanned map: an explainable 0-100 number plus the places where it falls short.

Every category is computed from evidence already in the scan - the navigation graph (walked path, stairs,
ramps, obstacles), the registered facilities and the corridor widths measured by `floor.py` - and carries
its weight, its own 0-1 score and a plain-language reason, so the number can be defended line by line.
Each issue has a position in the local frame, so the web viewer, the floor plan and the phone app can point
at it. What the scan did not see is reported as "unknown", never as a pass.

Categories (weights sum to 100):
  step_free     30  share of destinations (tagged places, else the walked path) reachable without stairs
  vertical      10  stairs have a step-free alternative (lift / ramp) in the scan
  stair_safety  15  staircases with a registered handrail
  clear_width   20  share of the measured corridor at least 0.9 m wide (wheelchair minimum)
  obstacles     10  registered obstacles near the walking line
  wayfinding    15  share of the route within 4 m of signage, tactile paving or an accessibility symbol
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WEIGHTS = {"step_free": 30, "vertical": 10, "stair_safety": 15, "clear_width": 20, "obstacles": 10, "wayfinding": 15}
LABELS = {"step_free": "Step-free reach", "vertical": "Lift / ramp alternative", "stair_safety": "Stair safety",
          "clear_width": "Clear width", "obstacles": "Obstacle-free path", "wayfinding": "Wayfinding aids"}
WHEELCHAIR_MIN_M = 0.9
AID_CLASSES = {"signboard", "tactile_paving", "accessibility_symbol"}


def _grade(s: float) -> str:
    return "A" if s >= 85 else "B" if s >= 70 else "C" if s >= 55 else "D" if s >= 40 else "E"


def _components(nodes: list[dict], edges: list[dict], blocked: set[str]) -> dict[str, int]:
    parent = {n["id"]: n["id"] for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in edges:
        if blocked & set(e["tags"]):
            continue
        a, b = find(e["a"]), find(e["b"])
        if a != b:
            parent[a] = b
    return {n: find(n) for n in parent}


def compute(job: Path) -> dict:
    from vision.navigation.floor import ensure_floor, narrow_stretches
    from vision.navigation.graph import build
    from vision.navigation.places import attach_places

    g = attach_places(build(job), job)
    fac = json.loads((job / "facilities.json").read_text())["facilities"]
    reg = [f for f in fac if f.get("position_local")]
    nodes, edges = g["nodes"], g["edges"]
    pos = {n["id"]: np.array(n["pos"], float) for n in nodes}
    traj_ids = [n["id"] for n in nodes if n["kind"] == "trajectory"]
    places = g.get("places", [])
    start = next((p["id"] for p in places if p.get("is_start")), traj_ids[0])
    if start not in pos:
        start = traj_ids[0]
    issues: list[dict] = []
    cats: dict[str, dict] = {}

    def issue(kind, severity, title, detail, at, **kw):
        issues.append({"id": f"{kind}-{len(issues)}", "kind": kind, "severity": severity, "title": title,
                       "detail": detail, "position": [round(float(v), 3) for v in at], **kw})

    # ---------------------------------------------------------------- step-free reach
    comp = _components(nodes, edges, {"stairs"})
    home = comp[start]
    stair_edges = [e for e in edges if "stairs" in e["tags"]]
    if places:
        dest = [p for p in places if p["id"] != start and p.get("kind") not in ("stairs",)]
        ok = [p for p in dest if comp.get(p["id"]) == home]
        frac = len(ok) / len(dest) if dest else 1.0
        for p in dest:
            if comp.get(p["id"]) != home:
                issue("no_step_free", "high", f"{p['name']} has no step-free route",
                      f"From the start, {p['name']} can only be reached using stairs.", p["position"],
                      place=p["name"])
        reason = (f"{len(ok)} of {len(dest)} places reachable without stairs from "
                  f"{next((p['name'] for p in places if p['id'] == start), 'the start')}.")
    else:
        seg = defaultdict(float)
        for e in edges:
            if e["a"] in traj_ids and e["b"] in traj_ids and "stairs" not in e["tags"]:
                seg[comp[e["a"]]] += e["length"]
        total = sum(e["length"] for e in edges if e["a"] in traj_ids and e["b"] in traj_ids) or 1.0
        frac = seg.get(home, 0.0) / total
        reason = f"{frac:.0%} of the walked path is reachable from the start without stairs."
    cats["step_free"] = {"score": frac, "reason": reason}

    # ---------------------------------------------------------------- vertical alternative / stair safety
    lifts = [f for f in reg if f["class_name"] in ("elevator", "ramp")]
    if stair_edges:
        mid = np.mean([(pos[e["a"]] + pos[e["b"]]) / 2 for e in stair_edges], axis=0)
        stair_only = any(i["kind"] == "no_step_free" for i in issues) or cats["step_free"]["score"] < 1
        if not stair_only:
            cats["vertical"] = {"score": 1.0, "reason": "Stairs present, but every destination also has a step-free route."}
        elif lifts:
            cats["vertical"] = {"score": 0.5, "reason": (
                f"{len(lifts)} lift/ramp seen in the scan, but the walk does not show it connecting the levels "
                "the stairs join (verify on site).")}
            issue("unlinked_lift", "medium", "Lift not linked to the stairs' levels",
                  "A lift or ramp was seen, but this scan does not show a step-free link between the levels.",
                  lifts[0]["position_local"])
        else:
            cats["vertical"] = {"score": 0.0, "reason": "Stairs present and no lift or ramp was seen in the scan."}
            issue("no_alternative", "high", "Stairs without a lift or ramp",
                  "No lift or ramp was found near these stairs. Wheelchair users cannot change level here.", mid)
        bare = [e for e in stair_edges if "stairs_no_handrail" in e["tags"]]
        cats["stair_safety"] = {"score": 1 - len(bare) / len(stair_edges),
                                "reason": f"{len(stair_edges) - len(bare)} of {len(stair_edges)} stair segments have a "
                                          "registered handrail."}
        if bare:
            m = np.mean([(pos[e["a"]] + pos[e["b"]]) / 2 for e in bare], axis=0)
            issue("no_handrail", "high", "Stairs without a handrail",
                  "No handrail was registered alongside these stairs. Risky for low-vision and older visitors.", m)
    else:
        cats["vertical"] = {"score": 1.0, "reason": "No stairs on the walked path."}
        cats["stair_safety"] = {"score": 1.0, "reason": "No stairs on the walked path."}

    # ---------------------------------------------------------------- clear width
    floor = ensure_floor(job)
    w = [x for x in floor["width_m"] if x is not None]
    if w:
        good = sum(1 for x in w if x >= WHEELCHAIR_MIN_M) / len(w)
        cats["clear_width"] = {"score": good, "reason": (
            f"{good:.0%} of the measured corridor is at least {WHEELCHAIR_MIN_M} m wide "
            f"(median {floor['summary']['median_width_m']} m; {floor['summary']['measured_fraction']:.0%} of the path "
            "had walls on both sides to measure).")}
        u = floor["unit_per_m"]
        for n in narrow_stretches(floor, WHEELCHAIR_MIN_M):
            issue("narrow", "high" if n["min_width_m"] < 0.8 else "medium",
                  f"Narrow passage ({n['min_width_m']:.2f} m)",
                  f"Clear width drops below {WHEELCHAIR_MIN_M} m for about {n['to_m'] - n['from_m']:.1f} m.",
                  [n["at"][0], float(np.median([p[1] for p in pos.values()])), n["at"][1]],
                  path2d=n["points"], unit_per_m=u)
    else:
        cats["clear_width"] = {"score": None, "reason": "Not enough wall evidence in the scan to measure widths."}

    # ---------------------------------------------------------------- obstacles near the walking line
    P = np.array([pos[i] for i in traj_ids])
    med = g["median_step"]
    near_obs = []
    for f in reg:
        if f["class_name"] != "obstacle":
            continue
        p = np.array(f["position_local"])
        if float(np.min(np.linalg.norm(P - p, axis=1))) < 6 * med:
            near_obs.append(f)
            issue("obstacle", "medium", "Obstacle near the path",
                  "An obstacle was detected close to the walking line. Check it does not narrow the route.", p,
                  facility_id=f["facility_id"])
    length = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) or 1.0
    per10 = len(near_obs) / max(length / (10 if g["scale_status"] != "relative" else 10 * med / 0.6), 1e-6)
    cats["obstacles"] = {"score": max(0.0, 1 - per10 / 2),
                         "reason": f"{len(near_obs)} obstacle(s) near {length:.0f} {g['units']} of path."}

    # ---------------------------------------------------------------- wayfinding aids
    aids = np.array([f["position_local"] for f in reg if f["class_name"] in AID_CLASSES])
    reach = 4.0 if g["scale_status"] != "relative" else 4.0 * med / 0.6
    if len(aids):
        d = np.min(np.linalg.norm(P[:, None, :] - aids[None, :, :], axis=2), axis=1)
        cov = float(np.mean(d < reach))
    else:
        d, cov = np.full(len(P), np.inf), 0.0
    cats["wayfinding"] = {"score": cov, "reason": (
        f"{cov:.0%} of the path is within 4 m of signage, tactile paving or an accessibility symbol "
        f"({len(aids)} registered).")}
    gap_start = None
    for i in range(len(P) + 1):
        bad = i < len(P) and d[i] >= reach
        if bad and gap_start is None:
            gap_start = i
        elif not bad and gap_start is not None:
            seg_len = float(np.sum(np.linalg.norm(np.diff(P[gap_start:i], axis=0), axis=1))) if i - gap_start > 1 else 0
            if seg_len >= 2 * reach:
                issue("no_signage", "low", "Long stretch without signage",
                      f"About {seg_len:.0f} {g['units']} with no sign, tactile paving or symbol in view.",
                      P[(gap_start + i - 1) // 2])
            gap_start = None

    # ---------------------------------------------------------------- total
    known = {k: v for k, v in cats.items() if v["score"] is not None}
    wsum = sum(WEIGHTS[k] for k in known) or 1
    total = sum(WEIGHTS[k] * known[k]["score"] for k in known) / wsum * 100
    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: order[x["severity"]])
    return {
        "score": round(total), "grade": _grade(total),
        "categories": [{"key": k, "label": LABELS[k], "weight": WEIGHTS[k],
                        "score": None if cats[k]["score"] is None else round(cats[k]["score"], 3),
                        "points": None if cats[k]["score"] is None else round(WEIGHTS[k] * cats[k]["score"], 1),
                        "reason": cats[k]["reason"]} for k in WEIGHTS],
        "issues": issues,
        "counts": {s: sum(1 for i in issues if i["severity"] == s) for s in ("high", "medium", "low")},
        "floor": {"median_width_m": floor["summary"]["median_width_m"], "min_width_m": floor["summary"]["min_width_m"],
                  "measured_fraction": floor["summary"]["measured_fraction"]},
        "units": g["units"], "scale_status": g["scale_status"],
        "note": ("Scored from what this scan saw along the walked path. Widths and positions are estimates "
                 "(approximate scale); unseen areas are unknown, not compliant or non-compliant."),
    }


def ensure_score(job: Path) -> dict:
    """Cached alongside the floor analysis; recomputed when any input file changes."""
    out = job / "plan" / "score.json"
    src = [job / n for n in ("trajectory.json", "alignment.json", "facilities.json", "places.json")]
    newest = max(p.stat().st_mtime for p in src if p.exists())
    if out.exists() and out.stat().st_mtime >= newest:
        try:
            return json.loads(out.read_text())
        except Exception:
            pass
    s = compute(job)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s))
    return s


__all__ = ["compute", "ensure_score", "WEIGHTS"]
