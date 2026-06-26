"""
Background Job Queue — Day 4
Runs diagnosis pipelines asynchronously so callers don't wait 12–60s.

Flow:
  POST /api/v1/jobs/diagnose  →  returns {job_id, status: "queued"}
  GET  /api/v1/jobs/{job_id}  →  returns {status: "running|done|failed", result?}

Jobs run in a FastAPI BackgroundTask, stored in-memory (or Postgres if enabled).
In production: replace with Celery + Redis or ARQ for persistent queues.

Job lifecycle:
  queued → running → done | failed
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel

from app.core.logger import get_logger
from app.core.models import DiagnosisResult

logger = get_logger(__name__)


class JobStatus(str, Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    scenario_id: str | None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    result: DiagnosisResult | None = None
    error: str | None = None


# In-memory store — {job_id: Job}
_jobs: dict[str, Job] = {}


def create_job(scenario_id: str | None = None) -> Job:
    """Create a new job in QUEUED state and register it."""
    job = Job(
        job_id=str(uuid.uuid4())[:12],
        status=JobStatus.QUEUED,
        scenario_id=scenario_id,
        created_at=datetime.now(timezone.utc),
    )
    _jobs[job.job_id] = job
    logger.info(f"Job {job.job_id} created for scenario='{scenario_id}'")
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def list_jobs(limit: int = 20) -> list[Job]:
    """Return most recent jobs first."""
    jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


async def run_job(
    job_id: str,
    scenario_id: str,
    metrics: Any,
    alert_summary: str,
    raw_logs: list[str],
    k8s_patch_template: str = "",
) -> None:
    """
    Execute a diagnosis job in the background.
    Updates job state transitions: queued → running → done | failed.
    """
    job = _jobs.get(job_id)
    if not job:
        logger.error(f"Job {job_id} not found — cannot run")
        return

    import time
    started = time.time()

    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)
    logger.info(f"Job {job_id} started")

    try:
        from app.agent.graph import run_diagnosis
        result = await run_diagnosis(
            scenario_id=scenario_id,
            metrics=metrics,
            alert_summary=alert_summary,
            raw_logs=raw_logs,
            k8s_patch_template=k8s_patch_template,
        )
        job.status = JobStatus.DONE
        job.result = result
        job.completed_at = datetime.now(timezone.utc)
        job.duration_seconds = round(time.time() - started, 2)
        logger.info(
            f"Job {job_id} done in {job.duration_seconds}s — "
            f"severity={result.severity}, fix={result.fix_category}"
        )

    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-500:]}"
        job.completed_at = datetime.now(timezone.utc)
        job.duration_seconds = round(time.time() - started, 2)
        logger.error(f"Job {job_id} failed: {exc}")


def purge_old_jobs(keep: int = 100) -> int:
    """Remove oldest jobs beyond the keep limit. Returns count removed."""
    if len(_jobs) <= keep:
        return 0
    sorted_ids = sorted(_jobs, key=lambda jid: _jobs[jid].created_at)
    to_remove = sorted_ids[:len(_jobs) - keep]
    for jid in to_remove:
        del _jobs[jid]
    return len(to_remove)
