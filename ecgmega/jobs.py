"""Tiny in-process job manager for long-running work (dataset builds, training)."""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .errors import Cancelled

MAX_LOG_LINES = 1000


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"           # queued | running | succeeded | failed | cancelled | interrupted
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: float = 0.0
    message: str = ""
    logs: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False)   # set after status is final AND persisted

    def to_dict(self, logs: bool = True, tail: int = 200) -> dict:
        d = {k: getattr(self, k) for k in ("id", "kind", "params", "status", "created", "started", "finished",
                                           "progress", "message", "result", "error")}
        if logs:
            d["logs"] = self.logs[-tail:]
        return d


class JobContext:
    def __init__(self, job: Job):
        self.job = job
        self.cancel = job.cancel_event

    def log(self, msg: str) -> None:
        self.job.logs.append(f"{time.strftime('%H:%M:%S')} {msg}")
        if len(self.job.logs) > MAX_LOG_LINES:
            del self.job.logs[: len(self.job.logs) - MAX_LOG_LINES]

    def progress(self, frac: float, msg: str = "") -> None:
        self.job.progress = float(max(0.0, min(1.0, frac)))
        if msg:
            self.job.message = msg


class JobManager:
    def __init__(self, persist_dir: str | Path | None = None, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ecg-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._dir = Path(persist_dir) if persist_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(self._dir.glob("*.json")):          # history from earlier server runs
                try:
                    d = json.loads(f.read_text())
                    j = Job(id=d["id"], kind=d["kind"], params=d.get("params", {}), status=d["status"],
                            created=d["created"], started=d.get("started"), finished=d.get("finished"),
                            progress=d.get("progress", 0.0), message=d.get("message", ""),
                            logs=d.get("logs", []), result=d.get("result"), error=d.get("error"))
                    if j.status in {"queued", "running"}:
                        j.status, j.error = "interrupted", "server restarted while the job was running"
                    self._jobs[j.id] = j
                except Exception:
                    continue

    def submit(self, kind: str, fn: Callable[[JobContext], dict], params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[JobContext], dict]) -> None:
        ctx = JobContext(job)
        if job.cancel_event.is_set():
            job.status, job.finished = "cancelled", time.time()
            self._persist(job)
            job.done.set()
            return
        job.status, job.started = "running", time.time()
        try:
            job.result = fn(ctx)
            outcome = "succeeded"
        except Cancelled:
            outcome = "cancelled"
        except Exception as e:                                   # noqa: BLE001
            outcome, job.error = "failed", f"{type(e).__name__}: {e}"
            ctx.log(traceback.format_exc(limit=6))
        job.finished = time.time()
        if outcome == "succeeded":
            job.progress = 1.0
        job.status = outcome                 # status becomes final only once everything else is set
        self._persist(job)
        job.done.set()

    def _persist(self, job: Job) -> None:
        if self._dir:
            try:
                (self._dir / f"{job.id}.json").write_text(json.dumps(job.to_dict(tail=MAX_LOG_LINES), default=str))
            except Exception:
                pass

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> bool:
        j = self._jobs.get(job_id)
        if j is None or j.status not in {"queued", "running"}:
            return False
        j.cancel_event.set()
        return True

    def shutdown(self) -> None:
        for j in self._jobs.values():
            j.cancel_event.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
