"""Pause / resume / stop of spatial jobs (vision/spatial/control.py, Runner controls, /spatial/jobs API)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

COUNTER = "import sys, time\nn = 0\nwhile True:\n    n += 1\n    open(sys.argv[1], 'w').write(str(n))\n    time.sleep(0.05)\n"


def _count(p: Path) -> int:
    try:
        return int(p.read_text() or 0)
    except (OSError, ValueError):
        return 0


def _until(cond, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_control_suspends_resumes_and_kills_a_real_worker(tmp_path):
    from vision.spatial.control import JobControl, JobStopped
    out = tmp_path / "n.txt"
    proc = subprocess.Popen([sys.executable, "-c", COUNTER, str(out)])
    c = JobControl("t1")
    c.register(proc)
    try:
        assert _until(lambda: _count(out) > 3)            # worker started (slow under load)
        c.pause()
        time.sleep(0.3)
        frozen = _count(out)
        time.sleep(0.6)
        assert _count(out) == frozen and proc.poll() is None        # suspended, not dead
        c.resume()
        assert _until(lambda: _count(out) > frozen)
        c.stop()
        proc.wait(timeout=10)
        with pytest.raises(JobStopped):
            c.checkpoint()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_checkpoint_blocks_while_paused():
    import threading
    from vision.spatial.control import JobControl
    c = JobControl("t2")
    c.pause()
    passed = threading.Event()
    threading.Thread(target=lambda: (c.checkpoint(), passed.set()), daemon=True).start()
    time.sleep(0.3)
    assert not passed.is_set()
    c.resume()
    assert passed.wait(2) and c.paused_seconds > 0.2


class FakePipeline:
    """Stands in for SpatialPipeline: reports progress forever until stopped (no models / COLMAP)."""

    def __init__(self, job_dir, video, options, on_progress=None, video_id="v", control=None):
        self.control, self.on_progress, self.n = control, on_progress, 0

    def progress(self):
        return {"stage": "sfm", "overall": min(0.99, self.n / 1000), "stages": {"sfm": {"status": "running",
                "message": f"tick {self.n}"}}}

    def run(self, force=None):
        while True:
            self.n += 1
            self.on_progress(self.progress())
            self.control.checkpoint()
            time.sleep(0.02)


@pytest.fixture()
def client(monkeypatch):
    import vision.spatial.pipeline as P
    monkeypatch.setattr(P, "SpatialPipeline", FakePipeline)
    from api.app import app
    with TestClient(app) as c:
        yield c


def _wait(client, jid, want, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/spatial/jobs/{jid}/progress").json()["status"]
        if s == want:
            return s
        time.sleep(0.1)
    raise AssertionError(f"job {jid} never reached {want} (is {s})")


def test_jobs_api_pause_resume_stop_start(client, tmp_path):
    from vision.spatial import jobs as J
    src = tmp_path / "walk.mp4"
    src.write_bytes(b"\x00" * 16)
    vid = J.add_video(src, "walk.mp4", {"duration_s": 12.0})
    jid = J.create_job(vid, {"preset": "fast"})
    J.get_runner().submit(jid)
    _wait(client, jid, "running")
    row = next(j for j in client.get("/spatial/jobs").json()["jobs"] if j["id"] == jid)
    assert row["video_name"] == "walk.mp4" and row["preset"] == "fast" and row["duration_s"] == 12.0

    assert client.post(f"/spatial/jobs/{jid}/resume").status_code == 409          # not paused
    assert client.post(f"/spatial/jobs/{jid}/pause").json()["status"] == "paused"
    time.sleep(0.3)
    t1 = client.get(f"/spatial/jobs/{jid}/progress").json()["stages"]["sfm"]["message"]
    time.sleep(0.4)
    assert client.get(f"/spatial/jobs/{jid}/progress").json()["stages"]["sfm"]["message"] == t1   # frozen
    assert client.post(f"/spatial/jobs/{jid}/resume").json()["status"] == "running"
    time.sleep(0.3)
    assert client.get(f"/spatial/jobs/{jid}/progress").json()["stages"]["sfm"]["message"] != t1

    assert client.post(f"/spatial/jobs/{jid}/start").status_code == 409           # already running
    assert client.post(f"/spatial/jobs/{jid}/stop").json()["status"] == "stopping"
    _wait(client, jid, "stopped")
    assert client.post(f"/spatial/jobs/{jid}/stop").status_code == 409
    assert client.post(f"/spatial/jobs/{jid}/start", json={"preset": "ultra"}).status_code == 409
    assert client.post(f"/spatial/jobs/{jid}/start", json={"preset": "balanced"}).json()["status"] == "queued"
    _wait(client, jid, "running")
    assert J.get_job(jid)["options"]["preset"] == "balanced"
    client.post(f"/spatial/jobs/{jid}/stop")
    _wait(client, jid, "stopped")
    assert client.post(f"/spatial/jobs/{jid}/explode").status_code == 404
    assert client.post("/spatial/jobs/nope/pause").status_code == 404


def test_delete_running_paused_queued_and_interrupted_jobs(client, tmp_path):
    """Delete works in every state: active jobs are stopped first, their folder and row disappear."""
    from vision.spatial import control as C
    from vision.spatial import jobs as J
    src = tmp_path / "walk2.mp4"
    src.write_bytes(b"\x00" * 16)
    vid = J.add_video(src, "walk2.mp4", {"duration_s": 5.0})

    def fresh():
        jid = J.create_job(vid, {"preset": "fast"})
        J.job_dir(jid).mkdir(parents=True, exist_ok=True)
        (J.job_dir(jid) / "state.json").write_text('{"stages": {}}')
        return jid

    # running
    a = fresh()
    J.get_runner().submit(a)
    _wait(client, a, "running")
    r = client.delete(f"/spatial/jobs/{a}")
    assert r.status_code == 200 and r.json()["was"] == "running" and r.json()["files_removed"]
    assert J.get_job(a) is None and not J.job_dir(a).exists() and C.get(a) is None

    # paused (the worker is blocked in a checkpoint)
    b = fresh()
    J.get_runner().submit(b)
    _wait(client, b, "running")
    client.post(f"/spatial/jobs/{b}/pause")
    _wait(client, b, "paused")
    assert client.delete(f"/spatial/jobs/{b}").json()["was"] == "paused"
    assert J.get_job(b) is None and not J.job_dir(b).exists() and C.get(b) is None

    # queued behind a running job (single worker): never starts
    c1, c2 = fresh(), fresh()
    J.get_runner().submit(c1)
    J.get_runner().submit(c2)
    _wait(client, c1, "running")
    assert client.delete(f"/spatial/jobs/{c2}").json()["was"] == "queued"
    client.delete(f"/spatial/jobs/{c1}")
    time.sleep(0.5)
    assert J.get_job(c2) is None and not J.job_dir(c2).exists()

    # half done / interrupted (no live worker)
    d = fresh()
    J.update_job(d, status="interrupted")
    assert client.delete(f"/spatial/jobs/{d}").json()["was"] == "interrupted"
    assert J.get_job(d) is None and not J.job_dir(d).exists()
    assert J.get_video(vid) is not None                       # the source video is kept
    assert client.delete(f"/spatial/jobs/{d}").status_code == 404
