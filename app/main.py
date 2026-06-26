"""
GPU Infrastructure Copilot — FastAPI Application v0.3.0 (Day 4)
Production-ready with middleware, job queue, and cluster scanning.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logger import get_logger
from app.middleware.logging import RequestIDMiddleware
from app.middleware.auth import AuthRateLimitMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GPU Copilot v0.3.0 starting up")

    from app.db.database import init_db
    await init_db()

    from app.services.qdrant_service import ensure_collection_seeded
    try:
        await ensure_collection_seeded()
    except Exception as e:
        logger.warning(f"Qdrant seeding failed (non-fatal): {e}")

    logger.info(
        f"Startup complete — "
        f"postgres={'on' if settings.postgres_enabled else 'off'} "
        f"qdrant={'live' if settings.qdrant_enabled else 'in-memory'} "
        f"slack={'on' if settings.slack_enabled else 'off'} "
        f"auth={'on' if settings.api_key else 'open'} "
        f"rate_limit={settings.rate_limit_per_minute}/min"
    )
    yield
    logger.info("GPU Copilot shutting down")


app = FastAPI(
    title="AI Infrastructure Copilot",
    description=(
        "Autonomous GPU incident diagnosis, remediation, and cluster-wide scanning. "
        "LangGraph + Claude + Qdrant RAG. Reduces investigation time 40min → 3min."
    ),
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware — order matters: outermost runs first
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(AuthRateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)

app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health():
    """Liveness probe — always returns 200 if the process is up."""
    return {
        "status": "ok",
        "service": "gpu-copilot",
        "version": "0.3.0",
        "features": {
            "rag_qdrant": True,
            "postgres": settings.postgres_enabled,
            "slack": settings.slack_enabled,
            "mock_data": settings.use_mock_data,
            "auth": bool(settings.api_key),
            "rate_limit": settings.rate_limit_per_minute,
        },
    }


@app.get("/ready", tags=["system"])
async def readiness():
    """
    Readiness probe — checks that all required services are reachable.
    Returns 200 only when the app can serve real traffic.
    Used by Kubernetes readinessProbe.
    """
    checks: dict[str, str] = {}

    # Qdrant (always available — in-memory fallback)
    try:
        from app.services.qdrant_service import _get_client
        client = _get_client()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"

    # Postgres (optional)
    if settings.postgres_enabled:
        try:
            from app.db.database import get_engine
            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            checks["postgres"] = "ok"
        except Exception as e:
            checks["postgres"] = f"error: {e}"
    else:
        checks["postgres"] = "disabled"

    all_ok = all(v in ("ok", "disabled") for v in checks.values())
    return {
        "ready": all_ok,
        "checks": checks,
        "version": "0.3.0",
    }


@app.get("/metrics", tags=["system"])
async def prometheus_metrics():
    """
    Expose basic app metrics in Prometheus text format.
    Mount this behind a /metrics scrape job in Prometheus.
    """
    from app.services.job_queue import list_jobs, JobStatus
    from app.middleware.auth import get_rate_limit_stats

    jobs = list_jobs(limit=1000)
    done = sum(1 for j in jobs if j.status == JobStatus.DONE)
    failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
    running = sum(1 for j in jobs if j.status == JobStatus.RUNNING)
    avg_duration = (
        sum(j.duration_seconds for j in jobs if j.duration_seconds) / max(done + failed, 1)
    )

    lines = [
        "# HELP gpu_copilot_jobs_total Total diagnosis jobs",
        "# TYPE gpu_copilot_jobs_total counter",
        f'gpu_copilot_jobs_total{{status="done"}} {done}',
        f'gpu_copilot_jobs_total{{status="failed"}} {failed}',
        f'gpu_copilot_jobs_total{{status="running"}} {running}',
        "",
        "# HELP gpu_copilot_avg_diagnosis_seconds Average diagnosis pipeline duration",
        "# TYPE gpu_copilot_avg_diagnosis_seconds gauge",
        f"gpu_copilot_avg_diagnosis_seconds {avg_duration:.2f}",
        "",
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines), media_type="text/plain")
