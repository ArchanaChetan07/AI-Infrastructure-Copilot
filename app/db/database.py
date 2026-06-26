"""
PostgreSQL Database Layer
Async SQLAlchemy models and CRUD for persistent incident storage.

Tables:
  - incidents        : one row per DiagnosisResult
  - remediation_steps: normalized steps per incident (FK to incidents)

Disabled by default (postgres_enabled=False) — when disabled, all
write operations are no-ops and reads return empty results, so the
app works fully without a running Postgres instance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── SQLAlchemy setup ──────────────────────────────────────────────────────────

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None and settings.postgres_enabled:
        _engine = create_async_engine(
            settings.postgres_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None and settings.postgres_enabled:
        engine = get_engine()
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


# ── ORM Models ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class IncidentRecord(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(String(64), unique=True, nullable=False, index=True)
    scenario_id = Column(String(128), nullable=True)
    node = Column(String(128), nullable=False)
    affected_gpu = Column(Integer, nullable=True)
    pod = Column(String(256), nullable=True)
    namespace = Column(String(128), nullable=True)

    severity = Column(String(32), nullable=False)
    fix_category = Column(String(64), nullable=False)
    root_cause = Column(Text, nullable=False)
    contributing_factors = Column(JSON, nullable=True)  # list[str]
    confidence = Column(Float, nullable=False, default=0.0)
    k8s_patch_yaml = Column(Text, nullable=True)

    investigation_duration_seconds = Column(Float, nullable=False)
    diagnosed_at = Column(DateTime(timezone=True), nullable=False)
    slack_notified = Column(Boolean, default=False)

    gpu_metrics_summary = Column(JSON, nullable=True)
    alert_summary = Column(Text, nullable=True)
    agent_trace = Column(JSON, nullable=True)  # list[str]

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    steps = relationship(
        "RemediationStepRecord",
        back_populates="incident",
        cascade="all, delete-orphan",
    )


class RemediationStepRecord(Base):
    __tablename__ = "remediation_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_fk = Column(Integer, ForeignKey("incidents.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    action = Column(String(256), nullable=False)
    command = Column(Text, nullable=True)
    description = Column(Text, nullable=False)

    incident = relationship("IncidentRecord", back_populates="steps")


# ── Schema init ───────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create tables if they don't exist. Called at app startup."""
    if not settings.postgres_enabled:
        logger.info("PostgreSQL disabled — skipping schema init")
        return
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("PostgreSQL schema initialized")


# ── CRUD ──────────────────────────────────────────────────────────────────────

async def save_incident(result: "DiagnosisResult") -> int | None:  # noqa: F821
    """
    Persist a DiagnosisResult to PostgreSQL.
    Returns the DB row ID, or None if Postgres is disabled.
    """
    if not settings.postgres_enabled:
        logger.info(f"PostgreSQL disabled — incident {result.incident_id} not persisted")
        return None

    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            record = IncidentRecord(
                incident_id=result.incident_id,
                scenario_id=result.scenario_id,
                node=result.node,
                affected_gpu=result.affected_gpu,
                pod=result.pod,
                namespace=result.namespace,
                severity=result.severity.value,
                fix_category=result.fix_category.value,
                root_cause=result.root_cause,
                contributing_factors=result.contributing_factors,
                confidence=result.confidence,
                k8s_patch_yaml=result.k8s_patch_yaml,
                investigation_duration_seconds=result.investigation_duration_seconds,
                diagnosed_at=result.diagnosed_at,
                slack_notified=result.slack_notified,
                gpu_metrics_summary=result.gpu_metrics_summary,
                alert_summary=result.alert_summary,
                agent_trace=result.agent_trace,
            )
            session.add(record)
            await session.flush()

            for step in result.remediation_steps:
                session.add(RemediationStepRecord(
                    incident_fk=record.id,
                    step=step.step,
                    action=step.action,
                    command=step.command,
                    description=step.description,
                ))

        logger.info(f"Saved incident {result.incident_id} to PostgreSQL (id={record.id})")
        return record.id


async def get_incident(incident_id: str) -> dict[str, Any] | None:
    """Fetch a single incident by ID."""
    if not settings.postgres_enabled:
        return None

    factory = get_session_factory()
    async with factory() as session:
        stmt = select(IncidentRecord).where(IncidentRecord.incident_id == incident_id)
        row = (await session.execute(stmt)).scalar_one_or_none()
        if not row:
            return None
        return _record_to_dict(row)


async def list_incidents(
    limit: int = 20,
    severity: str | None = None,
    node: str | None = None,
) -> list[dict[str, Any]]:
    """List recent incidents with optional filters."""
    if not settings.postgres_enabled:
        return []

    factory = get_session_factory()
    async with factory() as session:
        stmt = select(IncidentRecord).order_by(IncidentRecord.diagnosed_at.desc()).limit(limit)
        if severity:
            stmt = stmt.where(IncidentRecord.severity == severity)
        if node:
            stmt = stmt.where(IncidentRecord.node == node)

        rows = (await session.execute(stmt)).scalars().all()
        return [_record_to_dict(r) for r in rows]


async def get_mttr_stats() -> dict[str, Any]:
    """
    Return mean time to resolution stats across severity levels.
    The 'resolution' here is investigation_duration_seconds — a proxy for MTTR.
    """
    if not settings.postgres_enabled:
        return {"postgres_enabled": False}

    from sqlalchemy import func

    factory = get_session_factory()
    async with factory() as session:
        stmt = select(
            IncidentRecord.severity,
            func.count(IncidentRecord.id).label("count"),
            func.avg(IncidentRecord.investigation_duration_seconds).label("avg_seconds"),
            func.min(IncidentRecord.investigation_duration_seconds).label("min_seconds"),
            func.max(IncidentRecord.investigation_duration_seconds).label("max_seconds"),
        ).group_by(IncidentRecord.severity)

        rows = (await session.execute(stmt)).all()

        return {
            "by_severity": [
                {
                    "severity": r.severity,
                    "count": r.count,
                    "avg_investigation_seconds": round(r.avg_seconds or 0, 2),
                    "min_seconds": round(r.min_seconds or 0, 2),
                    "max_seconds": round(r.max_seconds or 0, 2),
                }
                for r in rows
            ]
        }


def _record_to_dict(row: IncidentRecord) -> dict[str, Any]:
    return {
        "incident_id": row.incident_id,
        "scenario_id": row.scenario_id,
        "node": row.node,
        "affected_gpu": row.affected_gpu,
        "pod": row.pod,
        "namespace": row.namespace,
        "severity": row.severity,
        "fix_category": row.fix_category,
        "root_cause": row.root_cause,
        "contributing_factors": row.contributing_factors,
        "confidence": row.confidence,
        "investigation_duration_seconds": row.investigation_duration_seconds,
        "diagnosed_at": row.diagnosed_at.isoformat() if row.diagnosed_at else None,
        "slack_notified": row.slack_notified,
        "agent_trace": row.agent_trace,
    }
