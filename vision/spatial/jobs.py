"""Video/job store (SQLite) and background job runner for the spatial pipeline."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from vision.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.RLock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY, name TEXT, path TEXT, meta TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, video_id TEXT, status TEXT, progress TEXT, options TEXT,
    created TEXT, updated TEXT, error TEXT);
"""


def root() -> Path:
    r = get_settings().output_dir / "spatial"
    r.mkdir(parents=True, exist_ok=True)
    return r


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(root() / "spatial.db", timeout=30)
    c.executescript(_SCHEMA)
    return c


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_video(src: Path, name: str, meta: dict) -> str:
    vid = uuid.uuid4().hex[:10]
    d = root() / "videos" / vid
    d.mkdir(parents=True, exist_ok=True)
    dest = d / ("source" + Path(name).suffix.lower())
    shutil.move(str(src), dest)
    with _lock, _db() as c:
        c.execute("INSERT INTO videos VALUES (?,?,?,?,?)", (vid, name, str(dest), json.dumps(meta), now()))
    return vid


def get_video(vid: str) -> dict | None:
    with _lock, _db() as c:
        r = c.execute("SELECT id,name,path,meta,created FROM videos WHERE id=?", (vid,)).fetchone()
    return None if r is None else {"id": r[0], "name": r[1], "path": r[2], "meta": json.loads(r[3]), "created": r[4]}


def job_dir(job_id: str) -> Path:
    return root() / "jobs" / job_id


def create_job(vid: str, options: dict) -> str:
    jid = uuid.uuid4().hex[:12]
    with _lock, _db() as c:
        c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", (jid, vid, "queued", json.dumps({}), json.dumps(options),
                                                                 now(), now(), None))
    return jid


def update_job(jid: str, **kw) -> None:
    sets, vals = [], []
    for k, v in kw.items():
        sets.append(f"{k}=?")
        vals.append(json.dumps(v) if k in ("progress", "options") else v)
    sets.append("updated=?")
    vals.append(now())
    with _lock, _db() as c:
        c.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", (*vals, jid))


def get_job(jid: str) -> dict | None:
    with _lock, _db() as c:
        r = c.execute("SELECT id,video_id,status,progress,options,created,updated,error FROM jobs WHERE id=?",
                      (jid,)).fetchone()
    if r is None:
        return None
    j = {"id": r[0], "video_id": r[1], "status": r[2], "progress": json.loads(r[3] or "{}"),
         "options": json.loads(r[4] or "{}"), "created": r[5], "updated": r[6], "error": r[7]}
    sp = job_dir(jid) / "state.json"
    if sp.exists():
        st = json.loads(sp.read_text())
        j["stages"] = {k: {kk: v.get(kk) for kk in ("status", "seconds", "error", "summary", "message")}
                       for k, v in st["stages"].items()}
    return j


def list_jobs(video_id: str | None = None) -> list[dict]:
    with _lock, _db() as c:
        q = "SELECT id FROM jobs" + (" WHERE video_id=?" if video_id else "") + " ORDER BY created DESC"
        ids = [r[0] for r in c.execute(q, (video_id,) if video_id else ())]
    return [get_job(i) for i in ids]


def _alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def mark_interrupted() -> int:
    """At startup: jobs left 'running'/'queued' whose owning process is gone are marked resumable
    ('interrupted'). Jobs still running in another live process (e.g. the CLI) are left alone."""
    with _lock, _db() as c:
        rows = c.execute("SELECT id FROM jobs WHERE status IN ('running','queued')").fetchall()
    n = 0
    for (jid,) in rows:
        lock = job_dir(jid) / "RUNNING"
        if lock.exists() and lock.read_text().strip().isdigit() and _alive(int(lock.read_text().strip())):
            continue
        update_job(jid, status="interrupted")
        n += 1
    return n


class Runner:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spatial-job")
        self.active: dict[str, object] = {}

    def submit(self, jid: str, force: list[str] | None = None) -> None:
        update_job(jid, status="queued", error=None)
        self.pool.submit(self._run, jid, set(force or ()))

    def _run(self, jid: str, force: set[str]) -> None:
        from vision.spatial.pipeline import SpatialPipeline
        j = get_job(jid)
        v = get_video(j["video_id"])
        update_job(jid, status="running")
        try:
            pipe = SpatialPipeline(job_dir(jid), Path(v["path"]), j["options"], video_id=v["id"],
                                   on_progress=lambda p: update_job(jid, progress=p))
            st = pipe.run(force=force)
            update_job(jid, status=st["status"], progress=pipe.progress())
        except Exception as e:
            log.exception("spatial job %s crashed", jid)
            update_job(jid, status="failed", error=f"{e.__class__.__name__}: {e}")


_runner: Runner | None = None


def get_runner() -> Runner:
    global _runner
    if _runner is None:
        _runner = Runner()
    return _runner
