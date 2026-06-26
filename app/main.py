"""
GPU Infrastructure Copilot — FastAPI Application
Diagnoses GPU failures using LLM agents and infrastructure telemetry.
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
    logger.info("GPU Copilot starting up — loading fixtures and config")
    yield
    logger.info("GPU Copilot shutting down")


app = FastAPI(
    title="GPU Infrastructure Copilot",
    description="AI-powered SRE agent for GPU cluster incident diagnosis and remediation",
    version="0.1.0",
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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "gpu-copilot"}
