"""
GPU Infrastructure Copilot — FastAPI Application v0.2.0 (Day 2)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GPU Copilot v0.2.0 starting up")

    # Initialize PostgreSQL schema
    from app.db.database import init_db
    await init_db()

    # Seed Qdrant with historical incidents (idempotent)
    from app.services.qdrant_service import ensure_collection_seeded
    try:
        await ensure_collection_seeded()
    except Exception as e:
        logger.warning(f"Qdrant seeding failed (non-fatal, will retry on first request): {e}")

    logger.info(
        f"Startup complete — "
        f"postgres={'enabled' if settings.postgres_enabled else 'disabled'} "
        f"qdrant={'enabled' if settings.qdrant_enabled else 'in-memory'} "
        f"slack={'enabled' if settings.slack_enabled else 'disabled'} "
        f"mock_data={settings.use_mock_data}"
    )
    yield
    logger.info("GPU Copilot shutting down")


app = FastAPI(
    title="AI Infrastructure Copilot",
    description=(
        "Autonomous GPU incident diagnosis and remediation using LLM agents, "
        "Qdrant RAG, and Kubernetes patch generation. "
        "Reduces investigation time from 40 minutes to under 3 minutes."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "service": "gpu-copilot",
        "version": "0.2.0",
        "features": {
            "rag_qdrant": settings.qdrant_enabled or True,   # in-memory always available
            "postgres": settings.postgres_enabled,
            "slack": settings.slack_enabled,
            "mock_data": settings.use_mock_data,
        },
    }
