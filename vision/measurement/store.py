"""SQLite persistence: vision results index (to reuse detections) and measurement history."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from vision.config import get_settings

_lock = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS vision_results (request_id TEXT PRIMARY KEY, json_path TEXT, image_name TEXT,
    width INT, height INT, created TEXT);
CREATE TABLE IF NOT EXISTS measurement_runs (run_id TEXT PRIMARY KEY, created TEXT, image_name TEXT,
    vision_request_id TEXT, json_path TEXT, scale_status TEXT, rules_profile TEXT);
CREATE TABLE IF NOT EXISTS measurements (measurement_id TEXT PRIMARY KEY, run_id TEXT, detection_id TEXT,
    facility_class TEXT, name TEXT, status TEXT, value TEXT, created TEXT, json TEXT);
CREATE INDEX IF NOT EXISTS idx_meas_det ON measurements(detection_id, created);
CREATE TABLE IF NOT EXISTS assessments (run_id TEXT, detection_id TEXT, json TEXT);
CREATE INDEX IF NOT EXISTS idx_ass_det ON assessments(detection_id);
"""


def _db_path() -> Path:
    return get_settings().output_dir / "measurements.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_db_path(), timeout=30)
    c.executescript(_SCHEMA)
    return c


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def register_vision_result(request_id: str, json_path: str, image_name: str, width: int, height: int) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO vision_results VALUES (?,?,?,?,?,?)",
                  (request_id, json_path, image_name, width, height, now()))


def get_vision_result_path(request_id: str) -> str | None:
    with _lock, _conn() as c:
        r = c.execute("SELECT json_path FROM vision_results WHERE request_id=?", (request_id,)).fetchone()
    return r[0] if r else None


def save_report(report) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO measurement_runs VALUES (?,?,?,?,?,?,?)",
                  (report.run_id, report.processed_at, report.image.name, report.vision_request_id, report.json_path,
                   report.scale.status.value, report.assessment_summary.profile))
        for f in report.facilities:
            for m in f.measurements:
                c.execute("INSERT OR REPLACE INTO measurements VALUES (?,?,?,?,?,?,?,?,?)",
                          (m.id, report.run_id, m.detection_id, m.facility_class, m.name, m.status.value,
                           json.dumps(m.value), report.processed_at, m.model_dump_json()))
        for a in report.assessment:
            c.execute("INSERT INTO assessments VALUES (?,?,?)", (report.run_id, a.detection_id, a.model_dump_json()))


def measurements_for(detection_id: str, latest_only: bool) -> list[dict]:
    with _lock, _conn() as c:
        runs = c.execute("SELECT DISTINCT m.run_id, r.created, r.image_name, r.scale_status, r.rules_profile, "
                         "r.json_path FROM measurements m JOIN measurement_runs r ON m.run_id=r.run_id "
                         "WHERE m.detection_id=? ORDER BY r.created DESC", (detection_id,)).fetchall()
        if latest_only:
            runs = runs[:1]
        out = []
        for run_id, created, img, sst, prof, jp in runs:
            ms = [json.loads(j) for (j,) in c.execute(
                "SELECT json FROM measurements WHERE run_id=? AND detection_id=? ORDER BY measurement_id",
                (run_id, detection_id))]
            ass = [json.loads(j) for (j,) in c.execute(
                "SELECT json FROM assessments WHERE run_id=? AND detection_id=?", (run_id, detection_id))]
            out.append({"run_id": run_id, "created": created, "image_name": img, "scale_status": sst,
                        "rules_profile": prof, "report_json": jp, "measurements": ms, "assessment": ass})
    return out
