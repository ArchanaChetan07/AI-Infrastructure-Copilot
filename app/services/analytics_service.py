"""
Analytics Service — Day 3
Computes incident trends, MTTR tracking, failure heatmaps, and recurrence
detection over the Postgres incident history.

When Postgres is disabled, returns computed stats from the fixture
historical_incidents.json so dashboards always show something useful.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Live Postgres analytics ───────────────────────────────────────────────────

async def get_mttr_trend(days: int = 30) -> dict[str, Any]:
    """
    Return daily average investigation duration over the past N days.
    Trend shows the '40 min → 3 min' improvement over time.
    """
    if not settings.postgres_enabled:
        return _mock_mttr_trend(days)

    from sqlalchemy import func, text
    from app.db.database import get_session_factory, IncidentRecord

    factory = get_session_factory()
    async with factory() as session:
        from sqlalchemy import select, cast, Date
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        stmt = (
            select(
                func.date(IncidentRecord.diagnosed_at).label("day"),
                func.count(IncidentRecord.id).label("count"),
                func.avg(IncidentRecord.investigation_duration_seconds).label("avg_seconds"),
            )
            .where(IncidentRecord.diagnosed_at >= cutoff)
            .group_by(func.date(IncidentRecord.diagnosed_at))
            .order_by(func.date(IncidentRecord.diagnosed_at))
        )
        rows = (await session.execute(stmt)).all()
        return {
            "days": days,
            "trend": [
                {
                    "date": str(r.day),
                    "incident_count": r.count,
                    "avg_investigation_seconds": round(r.avg_seconds or 0, 1),
                }
                for r in rows
            ],
        }


async def get_failure_heatmap() -> dict[str, Any]:
    """
    Return incident counts grouped by node × fix_category.
    Reveals which nodes have recurring GPU issues.
    """
    if not settings.postgres_enabled:
        return _mock_failure_heatmap()

    from app.db.database import get_session_factory, IncidentRecord
    from sqlalchemy import func, select

    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(
                IncidentRecord.node,
                IncidentRecord.fix_category,
                IncidentRecord.severity,
                func.count(IncidentRecord.id).label("count"),
            )
            .group_by(IncidentRecord.node, IncidentRecord.fix_category, IncidentRecord.severity)
            .order_by(func.count(IncidentRecord.id).desc())
        )
        rows = (await session.execute(stmt)).all()
        return {
            "heatmap": [
                {"node": r.node, "fix_category": r.fix_category, "severity": r.severity, "count": r.count}
                for r in rows
            ]
        }


async def get_recurrence_report() -> dict[str, Any]:
    """
    Identify nodes or GPUs with recurring incidents of the same type.
    A recurrence is 2+ incidents with same fix_category on same node within 7 days.
    """
    if not settings.postgres_enabled:
        return _mock_recurrence_report()

    from app.db.database import get_session_factory, IncidentRecord
    from sqlalchemy import func, select

    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(
                IncidentRecord.node,
                IncidentRecord.fix_category,
                func.count(IncidentRecord.id).label("count"),
                func.min(IncidentRecord.diagnosed_at).label("first_seen"),
                func.max(IncidentRecord.diagnosed_at).label("last_seen"),
            )
            .group_by(IncidentRecord.node, IncidentRecord.fix_category)
            .having(func.count(IncidentRecord.id) >= 2)
            .order_by(func.count(IncidentRecord.id).desc())
        )
        rows = (await session.execute(stmt)).all()
        recurrences = []
        for r in rows:
            days_span = (r.last_seen - r.first_seen).days if r.last_seen and r.first_seen else 0
            recurrences.append({
                "node": r.node,
                "fix_category": r.fix_category,
                "occurrences": r.count,
                "first_seen": r.first_seen.isoformat() if r.first_seen else None,
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                "span_days": days_span,
                "recommendation": _recurrence_recommendation(r.fix_category, r.count, days_span),
            })
        return {"recurrences": recurrences, "total_recurring_nodes": len({r["node"] for r in recurrences})}


async def get_dashboard_summary() -> dict[str, Any]:
    """
    Single endpoint for the ops dashboard — combines all analytics.
    Returns total incidents, severity breakdown, top failing nodes, MTTR.
    """
    if not settings.postgres_enabled:
        return _mock_dashboard_summary()

    from app.db.database import get_session_factory, IncidentRecord
    from sqlalchemy import func, select

    factory = get_session_factory()
    async with factory() as session:
        # Total and severity breakdown
        severity_stmt = (
            select(IncidentRecord.severity, func.count(IncidentRecord.id).label("count"))
            .group_by(IncidentRecord.severity)
        )
        severity_rows = (await session.execute(severity_stmt)).all()
        severity_breakdown = {r.severity: r.count for r in severity_rows}
        total = sum(severity_breakdown.values())

        # Overall MTTR
        mttr_stmt = select(func.avg(IncidentRecord.investigation_duration_seconds))
        avg_seconds = (await session.execute(mttr_stmt)).scalar() or 0

        # Top 5 most problematic nodes
        node_stmt = (
            select(IncidentRecord.node, func.count(IncidentRecord.id).label("count"))
            .group_by(IncidentRecord.node)
            .order_by(func.count(IncidentRecord.id).desc())
            .limit(5)
        )
        node_rows = (await session.execute(node_stmt)).all()

        # Most common fix
        fix_stmt = (
            select(IncidentRecord.fix_category, func.count(IncidentRecord.id).label("count"))
            .group_by(IncidentRecord.fix_category)
            .order_by(func.count(IncidentRecord.id).desc())
            .limit(1)
        )
        fix_row = (await session.execute(fix_stmt)).first()

        return {
            "total_incidents": total,
            "severity_breakdown": severity_breakdown,
            "avg_investigation_seconds": round(avg_seconds, 1),
            "avg_investigation_minutes": round(avg_seconds / 60, 1),
            "top_failing_nodes": [{"node": r.node, "count": r.count} for r in node_rows],
            "most_common_fix": fix_row.fix_category if fix_row else None,
            "postgres_enabled": True,
        }


# ── Mock analytics (when Postgres is disabled) ───────────────────────────────
# Computed from historical_incidents.json so dashboards always look real

def _load_historical() -> list[dict]:
    p = Path(settings.fixtures_dir) / "incidents" / "historical_incidents.json"
    return json.loads(p.read_text())


def _mock_mttr_trend(days: int) -> dict[str, Any]:
    """Simulated MTTR trend showing improvement over time."""
    from datetime import date
    today = date.today()
    # Simulate decreasing MTTR: starts at 2400s (40 min), ends at 180s (3 min)
    trend = []
    for i in range(min(days, 30)):
        d = today - timedelta(days=days - i)
        # Exponential decay from 2400 to 180 seconds
        progress = i / max(days - 1, 1)
        avg_s = 2400 * (1 - progress) + 180 * progress + (((i % 3) - 1) * 30)
        trend.append({
            "date": d.isoformat(),
            "incident_count": [2, 1, 3, 1, 2, 0, 1][i % 7],
            "avg_investigation_seconds": round(max(avg_s, 150), 1),
        })
    return {"days": days, "trend": trend, "note": "simulated — enable Postgres for real data"}


def _mock_failure_heatmap() -> dict[str, Any]:
    incidents = _load_historical()
    counts: dict[tuple, int] = defaultdict(int)
    for inc in incidents:
        counts[(inc["node"], inc["fix_category"], inc["severity"])] += 1
    return {
        "heatmap": [
            {"node": k[0], "fix_category": k[1], "severity": k[2], "count": v}
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        ],
        "note": "from fixture historical_incidents — enable Postgres for live data",
    }


def _mock_recurrence_report() -> dict[str, Any]:
    incidents = _load_historical()
    node_fix: dict[tuple, list] = defaultdict(list)
    for inc in incidents:
        node_fix[(inc["node"], inc["fix_category"])].append(inc["date"])
    recurrences = []
    for (node, fix), dates in node_fix.items():
        if len(dates) >= 2:
            recurrences.append({
                "node": node,
                "fix_category": fix,
                "occurrences": len(dates),
                "first_seen": min(dates),
                "last_seen": max(dates),
                "recommendation": _recurrence_recommendation(fix, len(dates), 30),
            })
    return {
        "recurrences": recurrences,
        "total_recurring_nodes": len({r["node"] for r in recurrences}),
        "note": "from fixture data — enable Postgres for live recurrence detection",
    }


def _mock_dashboard_summary() -> dict[str, Any]:
    incidents = _load_historical()
    severity_counts = Counter(inc["severity"] for inc in incidents)
    avg_resolution = sum(inc["resolution_minutes"] * 60 for inc in incidents) / max(len(incidents), 1)
    node_counts = Counter(inc["node"] for inc in incidents)
    fix_counts = Counter(inc["fix_category"] for inc in incidents)
    return {
        "total_incidents": len(incidents),
        "severity_breakdown": dict(severity_counts),
        "avg_investigation_seconds": round(avg_resolution, 1),
        "avg_investigation_minutes": round(avg_resolution / 60, 1),
        "top_failing_nodes": [{"node": n, "count": c} for n, c in node_counts.most_common(5)],
        "most_common_fix": fix_counts.most_common(1)[0][0] if fix_counts else None,
        "postgres_enabled": False,
        "note": "from fixture data — enable Postgres for live stats",
    }


def _recurrence_recommendation(fix_category: str, count: int, span_days: int) -> str:
    recs = {
        "gpu_drain_and_reset": f"GPU has required {count} resets. Investigate hardware — likely failing VRAM or cooling system degradation.",
        "config_patch": f"Config-related incidents repeat ({count}×). Implement admission controller to enforce resource limits.",
        "nvlink_reset": f"NVLink instability recurring ({count}×). Schedule PCIe/NVLink cable inspection. Consider driver upgrade.",
        "pod_restart": f"Pod crashes repeating ({count}×). Review memory limits and vLLM KV cache config.",
        "node_drain": f"Node required draining {count} times. Check for hardware issues or kernel updates.",
        "manual_intervention": f"Manual intervention needed {count} times. Document runbook and automate recovery steps.",
    }
    return recs.get(fix_category, f"Recurring issue ({count}×). Investigate root cause.")
