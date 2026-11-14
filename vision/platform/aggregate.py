"""Read-only aggregation over the existing stores (vision index, measurement runs, spatial jobs).
All numbers are computed from stored results; repeated analyses of the same image count once (latest)."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

from vision.config import REPO_ROOT, get_settings  # noqa: F401 (get_settings re-exported)
from vision.platform import store as PS

FACILITY_CLASSES = {"staircase", "ramp", "elevator", "door", "entrance_exit", "handrail", "tactile_paving",
                    "corridor", "walkway", "accessibility_symbol", "signboard", "obstacle", "wheelchair_user"}


def _mdb():
    return sqlite3.connect(get_settings().output_dir / "measurements.db", timeout=30)


@lru_cache(maxsize=1)
def sample_provenance() -> dict[str, dict]:
    out = {}
    for f in (REPO_ROOT / "samples/images/SOURCES.json", REPO_ROOT / "samples/videos/SOURCES.json"):
        if f.exists():
            for r in json.loads(f.read_text()):
                out[r["file"]] = {"kind": "real_sample", "license": r.get("license"), "author": r.get("author"),
                                  "source": r.get("page"), "title": r.get("title")}
    return out


def provenance(name: str) -> dict:
    return sample_provenance().get(name, {"kind": "user_upload"})


def public_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(get_settings().output_dir)
    except ValueError:
        return None
    return f"/vision/outputs/{rel.as_posix()}" if Path(path).exists() else None


def library(latest_only: bool = True) -> list[dict]:
    try:
        with _mdb() as c:
            rows = c.execute("SELECT request_id, json_path, image_name, width, height, created FROM vision_results "
                             "ORDER BY created DESC").fetchall()
    except sqlite3.OperationalError:
        return []
    seen, out = set(), []
    for rid, jp, name, w, h, created in rows:
        if latest_only and name in seen:
            continue
        p = Path(jp)
        if not p.exists():
            continue
        if (get_settings().output_dir / "validation") in p.resolve().parents:
            continue            # validation-dataset artefacts (e.g. DIODE) are not campus content
        seen.add(name)
        try:
            res = json.loads(p.read_text())
        except Exception:
            continue
        stem = p.with_suffix("")
        out.append({"request_id": rid, "image_name": name, "width": w, "height": h, "created": created,
                    "counts": res.get("counts_by_class", {}), "detections": len(res.get("detections", [])),
                    "findings": [f["code"] for f in res.get("findings", [])],
                    "model": res.get("model", {}).get("key"),
                    "annotated_url": public_url(str(stem) + "_annotated.jpg"),
                    "source_url": public_url(str(stem) + "_source.jpg"),
                    "has_source": Path(str(stem) + "_source.jpg").exists(),
                    "json_url": public_url(p), "provenance": provenance(name)})
    return out


def _one(sql: str, args: tuple):
    try:
        with _mdb() as c:
            return c.execute(sql, args).fetchone()
    except sqlite3.OperationalError:          # store not created yet
        return None


def load_result(request_id: str) -> dict | None:
    r = _one("SELECT json_path FROM vision_results WHERE request_id=?", (request_id,))
    if not r or not Path(r[0]).exists():
        return None
    return json.loads(Path(r[0]).read_text())


def latest_measurement_for(request_id: str) -> dict | None:
    r = _one("SELECT json_path FROM measurement_runs WHERE vision_request_id=? ORDER BY created DESC LIMIT 1",
             (request_id,))
    if not r or not r[0] or not Path(r[0]).exists():
        return None
    rep = json.loads(Path(r[0]).read_text())
    rep["annotated_url"] = public_url(rep.get("annotated_image"))
    return rep


def spatial_jobs() -> list[dict]:
    from vision.spatial import jobs as J
    try:
        return J.list_jobs()
    except Exception:
        return []


def spatial_facilities() -> list[dict]:
    from vision.spatial import jobs as J
    out = []
    for j in spatial_jobs():
        p = J.job_dir(j["id"]) / "facilities.json"
        if not p.exists():
            continue
        vid = J.get_video(j["video_id"]) or {}
        for f in json.loads(p.read_text())["facilities"]:
            out.append({**f, "reconstruction_id": j["id"], "video_name": vid.get("name")})
    return out


def facilities(q: str | None = None, cls: str | None = None, status: str | None = None) -> list[dict]:
    reviews = PS.latest_reviews()
    out = []
    for item in library():
        res = load_result(item["request_id"]) or {}
        meas = latest_measurement_for(item["request_id"])
        mby = {}
        for f in (meas or {}).get("facilities", []):
            mby[f["detection_id"]] = f["measurements"]
        for d in res.get("detections", []):
            if d["class_name"] not in FACILITY_CLASSES:
                continue
            ms = mby.get(d["id"], [])
            out.append({"id": d["id"], "kind": "image_detection", "class_name": d["class_name"],
                        "confidence": d["confidence"], "source_name": item["image_name"],
                        "source_request_id": item["request_id"], "bbox": d["bbox"],
                        "annotated_url": item["annotated_url"], "source_url": item["source_url"],
                        "provenance": item["provenance"],
                        "attributes": {k: v.get("value") for k, v in d.get("attributes", {}).items()
                                       if v.get("status") in ("detected", "estimated")},
                        "measurements": [{"name": m["name"], "value": m["value"], "unit": m.get("unit"),
                                          "uncertainty": m.get("uncertainty"), "status": m["status"],
                                          "provisional_value": m.get("provisional_value")} for m in ms],
                        "review": reviews.get(d["id"])})
    for f in spatial_facilities():
        # facility_id comes from the source video, so two reconstructions of one video share ids: scope it.
        uid = f"{f['reconstruction_id']}:{f['facility_id']}"
        out.append({"id": uid, "facility_id": f["facility_id"], "kind": "reconstruction_facility",
                    "class_name": f["class_name"],
                    "confidence": f.get("registration_confidence"), "source_name": f.get("video_name"),
                    "reconstruction_id": f["reconstruction_id"], "position_local": f.get("position_local"),
                    "position_units": f.get("position_units"), "observations": len(f.get("observations", [])),
                    "registered": f.get("position_sfm") is not None, "reason": f.get("reason"),
                    "provenance": provenance(f.get("video_name") or ""),
                    "measurements": f.get("measurements", []), "review": reviews.get(uid)})
    if q:
        ql = q.lower()
        out = [x for x in out if ql in x["class_name"] or ql in (x.get("source_name") or "").lower() or ql in x["id"]]
    if cls:
        out = [x for x in out if x["class_name"] == cls]
    if status:
        out = [x for x in out if (x.get("review") or {}).get("status", "unreviewed") == status]
    return out


def compliance(profile: str | None = None) -> dict:
    reviews = PS.latest_reviews()
    issues, outcomes, rules = [], Counter(), Counter()
    gaps = []
    for item in library():
        meas = latest_measurement_for(item["request_id"])
        if meas:
            for a in meas.get("assessment", []):
                outcomes[a["outcome"]] += 1
                if a["outcome"] in ("POTENTIAL_ISSUE", "POSSIBLE_ISSUE"):
                    rules[a["rule_id"]] += 1
                    issues.append({**a, "image_name": item["image_name"], "request_id": item["request_id"],
                                   "measurement_run": meas["run_id"], "profile": meas["assessment_summary"]["profile"],
                                   "review": reviews.get(a["detection_id"]), "annotated_url": meas.get("annotated_url")})
        res = load_result(item["request_id"]) or {}
        for f in res.get("findings", []):
            if f["severity"] == "warning":
                gaps.append({"code": f["code"], "message": f["message"], "image_name": item["image_name"],
                             "request_id": item["request_id"], "detection_ids": f["detection_ids"]})
    reviewed = sum(1 for i in issues if i.get("review"))
    fixed = sum(1 for i in issues if (i.get("review") or {}).get("status") == "fixed")
    return {"issues": sorted(issues, key=lambda i: (i["outcome"] != "POTENTIAL_ISSUE", i["rule_id"])),
            "gaps": gaps, "outcome_counts": dict(outcomes), "issues_by_rule": dict(rules.most_common()),
            "gap_counts": dict(Counter(g["code"] for g in gaps)),
            "progress": {"issues": len(issues), "reviewed": reviewed, "fixed": fixed},
            "statement": "Screening results only - no compliance determination. POSSIBLE issues are within "
                         "measurement uncertainty or based on undetected (not absent) features."}


def overview() -> dict:
    lib = library()
    det = Counter()
    for x in lib:
        det.update(x["counts"])
    try:
        with _mdb() as c:
            n_analyses = c.execute("SELECT COUNT(*) FROM vision_results").fetchone()[0]
            n_runs = c.execute("SELECT COUNT(*) FROM measurement_runs").fetchone()[0]
            recent_runs = c.execute("SELECT run_id, image_name, scale_status, created FROM measurement_runs "
                                    "ORDER BY created DESC LIMIT 8").fetchall()
    except sqlite3.OperationalError:          # fresh install: measurement store not created yet
        n_analyses, n_runs, recent_runs = 0, 0, []
    mstat = Counter()
    for x in lib:
        m = latest_measurement_for(x["request_id"])
        for f in (m or {}).get("facilities", []):
            for mm in f["measurements"]:
                mstat[mm["status"]] += 1
    jobs = spatial_jobs()
    sf = spatial_facilities()
    comp = compliance()
    activity = [{"kind": "image_analysis", "title": x["image_name"], "time": x["created"], "ref": x["request_id"]}
                for x in lib[:8]]
    activity += [{"kind": "measurement", "title": r[1], "time": r[3], "ref": r[0], "detail": r[2]} for r in recent_runs]
    for j in jobs[:8]:
        activity.append({"kind": "spatial_job", "title": j["id"], "time": j["updated"], "ref": j["id"],
                         "detail": j["status"]})
    activity.sort(key=lambda a: a["time"] or "", reverse=True)
    return {"images": {"distinct": len(lib), "analyses_total": n_analyses,
                       "real_samples": sum(1 for x in lib if x["provenance"]["kind"] == "real_sample")},
            "detections_by_class": dict(det.most_common()),
            "measurements": {"runs_total": n_runs, "latest_by_status": dict(mstat)},
            "spatial": {"jobs": len(jobs), "by_status": dict(Counter(j["status"] for j in jobs)),
                        "facilities_3d": len(sf), "registered_3d": sum(1 for f in sf if f.get("position_sfm"))},
            "compliance": {"outcomes": comp["outcome_counts"], "gaps": comp["gap_counts"],
                           "issues": comp["progress"]["issues"]},
            "reviews": dict(Counter(r["status"] for r in PS.latest_reviews().values())),
            "activity": activity[:15]}
