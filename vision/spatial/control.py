"""Pause / resume / stop for running spatial jobs.

Each running job has a JobControl. Pipeline stages call `checkpoint()` from their progress callbacks and
worker poll loops, and register the worker subprocesses they launch (COLMAP, splat training):

* pause  - worker process trees are suspended in place (psutil; NtSuspendProcess on Windows, SIGSTOP on
           POSIX) and in-process stages block at their next checkpoint. Nothing is lost.
* resume - suspended processes continue exactly where they were.
* stop   - worker trees are killed and the next checkpoint raises JobStopped. The interrupted stage goes
           back to 'pending', so starting the job again resumes from the last finished stage.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class JobStopped(Exception):
    """Raised inside a pipeline when the user stops the job."""


def _tree(pid: int):
    import psutil
    try:
        p = psutil.Process(pid)
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return []


class JobControl:
    def __init__(self, jid: str):
        self.jid = jid
        self._lock = threading.RLock()
        self._run = threading.Event()
        self._run.set()                       # set = not paused
        self.stop_requested = False
        self.procs: list = []                 # subprocess.Popen objects of live workers
        self.paused_seconds = 0.0

    # ---------------------------------------------------------------- workers
    def register(self, proc) -> None:
        with self._lock:
            self.procs.append(proc)
            if self.stop_requested:
                self._kill(proc)
            elif not self._run.is_set():
                self._signal(proc, "suspend")

    def unregister(self, proc) -> None:
        with self._lock:
            if proc in self.procs:
                self.procs.remove(proc)

    def _signal(self, proc, what: str) -> None:
        import psutil
        for p in _tree(proc.pid):
            try:
                getattr(p, what)()
            except psutil.Error:
                pass

    def _kill(self, proc) -> None:
        self._signal(proc, "resume")          # a suspended process cannot exit cleanly on some platforms
        self._signal(proc, "kill")

    # ---------------------------------------------------------------- commands
    @property
    def paused(self) -> bool:
        return not self._run.is_set()

    def pause(self) -> None:
        with self._lock:
            if self.stop_requested or self.paused:
                return
            self._run.clear()
            for p in list(self.procs):
                self._signal(p, "suspend")
        log.info("job %s paused", self.jid)

    def resume(self) -> None:
        with self._lock:
            for p in list(self.procs):
                self._signal(p, "resume")
            self._run.set()
        log.info("job %s resumed", self.jid)

    def stop(self) -> None:
        with self._lock:
            self.stop_requested = True
            for p in list(self.procs):
                self._kill(p)
            self._run.set()                   # wake a paused checkpoint so it can raise
        log.info("job %s stop requested", self.jid)

    def checkpoint(self) -> None:
        """Block while paused; raise JobStopped once a stop was requested."""
        if self.stop_requested:
            raise JobStopped(self.jid)
        if not self._run.is_set():
            t0 = time.time()
            self._run.wait()
            self.paused_seconds += time.time() - t0
        if self.stop_requested:
            raise JobStopped(self.jid)


_controls: dict[str, JobControl] = {}
_reg_lock = threading.Lock()


def get(jid: str) -> JobControl | None:
    return _controls.get(jid)


def create(jid: str) -> JobControl:
    with _reg_lock:
        c = _controls[jid] = JobControl(jid)
        return c


def release(jid: str) -> None:
    with _reg_lock:
        _controls.pop(jid, None)
