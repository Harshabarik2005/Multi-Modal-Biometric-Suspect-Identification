"""Background jobs for enrollment and scanning.

Both take minutes -- a scan runs detection, tracking and three embedding models
over every frame -- so neither can happen inside an HTTP request. The browser
gets a job id immediately and polls for progress.

Deliberately in-process, not Celery or RQ. Those need a broker and a separate
worker process, which is a lot of moving parts for a single-machine review
console. The trade-off is stated rather than hidden: **jobs do not survive a
restart**, and there is no retry. For this system that is acceptable -- a lost
enrollment is re-uploaded, and a lost scan is re-run -- but it is exactly the
thing to replace first if this ever runs somewhere that matters.

Concurrency is capped at one worker by default. The models are the bottleneck
and they contend for the same 4GB of VRAM; running two scans at once makes both
slower and risks running out of memory mid-run.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from app.core.logging import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    #: 0.0 to 1.0. Best-effort; some stages cannot report progress usefully.
    progress: float = 0.0
    message: str = "queued"
    result: Any = None
    error: str = ""
    created_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobRunner:
    """Runs jobs on a small thread pool and keeps their state in memory."""

    def __init__(self, max_workers: int = 1, history: int = 200) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="frs-job"
        )
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._futures: dict[str, Future] = {}
        #: Per-job cleanup, run once whether or not the job ever ran.
        self._cleanups: dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self._history = history

    def submit(
        self,
        kind: str,
        work: Callable[[Job], Any],
        cleanup: Callable[[], None] | None = None,
    ) -> Job:
        """Queue `work`, which receives its own `Job` so it can report progress.

        `cleanup` runs exactly once, whether or not `work` ever does. Uploaded
        footage used to be deleted in the work function's own `finally`, which
        never runs for a job that is cancelled at shutdown or is still queued
        when the process stops -- leaving a temp directory of someone's
        biometric footage on disk with nothing tracking it (SEC-11).
        """
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._prune_locked()

        def run() -> None:
            job.status = JobStatus.RUNNING
            job.message = "starting"
            try:
                job.result = work(job)
                job.status = JobStatus.SUCCEEDED
                job.progress = 1.0
                if job.message in ("starting", "queued"):
                    job.message = "done"
            except Exception as exc:  # noqa: BLE001 - a failed job must not
                # take the server down with it; the browser needs to be told.
                job.status = JobStatus.FAILED
                job.error = str(exc) or exc.__class__.__name__
                job.message = "failed"
                logger.error(
                    "Job %s (%s) failed: %s\n%s",
                    job.id, job.kind, exc, traceback.format_exc(),
                )
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                self._clean(job.id)

        if cleanup is not None:
            with self._lock:
                self._cleanups[job.id] = cleanup

        self._futures[job.id] = self._executor.submit(run)
        return job

    def _clean(self, job_id: str) -> None:
        """Run a job's cleanup, at most once, whoever gets there first."""
        with self._lock:
            cleanup = self._cleanups.pop(job_id, None)
        if cleanup is None:
            return
        try:
            cleanup()
        except Exception:  # noqa: BLE001 - cleanup failing must not propagate
            logger.exception("Cleanup for job %s failed", job_id)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 25, kind: str | None = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        return list(reversed(jobs))[:limit]

    def _prune_locked(self) -> None:
        """Drop the oldest finished jobs once history is exceeded.

        Only finished ones: evicting a running job would lose the handle the
        browser is polling.
        """
        while len(self._jobs) > self._history:
            for job_id, job in list(self._jobs.items()):
                if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                    self._jobs.pop(job_id, None)
                    self._futures.pop(job_id, None)
                    self._cleanups.pop(job_id, None)
                    break
            else:
                return  # nothing finished to evict

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        # Cancelled and never-started jobs never reach their own `finally`, so
        # their uploads would be left behind (SEC-11).
        with self._lock:
            pending = list(self._cleanups)
        for job_id in pending:
            self._clean(job_id)


@dataclass
class ProgressReporter:
    """Small helper so job bodies can report progress without ceremony."""

    job: Job
    stages: list[str] = field(default_factory=list)
    _index: int = 0

    def stage(self, message: str) -> None:
        self.job.message = message
        if self.stages:
            self._index = min(self._index + 1, len(self.stages))
            self.job.progress = self._index / (len(self.stages) + 1)

    def fraction(self, value: float, message: str | None = None) -> None:
        self.job.progress = max(0.0, min(1.0, value))
        if message:
            self.job.message = message
