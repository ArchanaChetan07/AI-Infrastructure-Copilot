"""
API routes — all endpoints for GPU Copilot v0.2.0 (Day 2).

New endpoints:
  GET  /api/v1/incidents              — list persisted incidents (Postgres)
  GET  /api/v1/incidents/{id}         — get single incident
  GET  /api/v1/incidents/stats/mttr   — MTTR stats by severity
  POST /api/v1/rag/search             — search Qdrant directly
  GET  /api/v1/rag/incidents          — list all seeded historical incidents
  GET  /api/v1/slack/preview          — preview Slack message for a scenario
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.graph import run_diagnosis
from app.core.config import settings
from app.core.fixtures import load_scenario_bundle
from app.core.logger import get_logger
from app.core.models import DiagnoseRequest, DiagnosisResult

logger = get_logger(__name__)
router = APIRouter()


# ── Diagnosis ─────────────────────────────────────────────────────────────────

@router.post(
    "/diagnose",
    response_model=DiagnosisResult,
    summary="Run AI diagnosis on a GPU incident",
    tags=["diagnosis"],
)
async def diagnose(request: DiagnoseRequest) -> DiagnosisResult:
    """
    Trigger the 5-node LangGraph diagnosis agent (Day 2: includes RAG retrieval).

    - Pass `scenario_id` to use fixture data (mock mode).
    - Pass `alert_payload` for production Alertmanager webhooks.

    Post-pipeline: saves to Postgres, notifies Slack, upserts new incident into Qdrant.
    """
    if settings.use_mock_data:
        scenario_id = request.scenario_id or "gpu_thermal_throttle_ecc"
        try:
            bundle = load_scenario_bundle(scenario_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        alert = bundle["alert"]
        alert_summary = (
            f"{alert.commonAnnotations.get('summary', '')} "
            f"Node: {alert.commonLabels.get('node', 'unknown')}. "
            f"Pod: {alert.alerts[0].labels.pod if alert.alerts else 'unknown'}. "
            f"Namespace: {alert.alerts[0].labels.namespace if alert.alerts else 'unknown'}."
        )

        return await run_diagnosis(
            scenario_id=scenario_id,
            metrics=bundle["metrics"],
            alert_summary=alert_summary,
            raw_logs=bundle["logs"],
            k8s_patch_template=bundle["k8s_patch"],
        )

    else:
        if not request.alert_payload:
            raise HTTPException(
                status_code=400,
                detail="alert_payload required when use_mock_data=False",
            )
        alert = request.alert_payload
        alert_summary = (
            f"{alert.commonAnnotations.get('summary', '')} "
            f"Node: {alert.commonLabels.get('node', 'unknown')}."
        )
        raise HTTPException(
            status_code=501,
            detail="Live mode not yet implemented. Set USE_MOCK_DATA=true.",
        )


# ── Scenarios & Fixtures ──────────────────────────────────────────────────────

@router.get("/scenarios", summary="List available demo scenarios", tags=["fixtures"])
async def list_scenarios():
    data = json.loads(
        (Path(settings.fixtures_dir) / "scenarios" / "scenarios.json").read_text()
    )
    return {
        "scenarios": [
            {"id": s["id"], "name": s["name"], "severity": s["severity"], "description": s["description"]}
            for s in data["scenarios"]
        ]
    }


@router.get("/metrics/mock", summary="Raw mock GPU metrics", tags=["fixtures"])
async def mock_metrics():
    from app.core.fixtures import load_gpu_metrics
    return load_gpu_metrics()


@router.get("/alert/mock", summary="Raw mock Alertmanager payload", tags=["fixtures"])
async def mock_alert():
    from app.core.fixtures import load_alert_payload
    return load_alert_payload()


# ── Incident History (Postgres) ───────────────────────────────────────────────

@router.get("/incidents", summary="List persisted incidents", tags=["incidents"])
async def list_incidents(
    limit: int = 20,
    severity: str | None = None,
    node: str | None = None,
):
    """
    List incidents persisted to PostgreSQL (requires POSTGRES_ENABLED=true).
    Returns empty list when Postgres is disabled.
    """
    from app.db.database import list_incidents as db_list
    return {"incidents": await db_list(limit=limit, severity=severity, node=node)}


@router.get("/incidents/stats/mttr", summary="MTTR stats by severity", tags=["incidents"])
async def mttr_stats():
    """
    Return mean investigation time grouped by severity.
    Useful for tracking the '40 min → 3 min' improvement over time.
    """
    from app.db.database import get_mttr_stats
    return await get_mttr_stats()


@router.get("/incidents/{incident_id}", summary="Get a single incident", tags=["incidents"])
async def get_incident(incident_id: str):
    from app.db.database import get_incident as db_get
    result = await db_get(incident_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Incident {incident_id} not found (or Postgres is disabled)",
        )
    return result


# ── RAG / Qdrant ─────────────────────────────────────────────────────────────

class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/rag/search", summary="Search Qdrant for similar incidents", tags=["rag"])
async def rag_search(request: RAGSearchRequest):
    """
    Directly query the Qdrant vector store.
    Useful for exploring what historical incidents exist and testing retrieval quality.
    """
    from app.services.qdrant_service import retrieve_similar_incidents
    results = await retrieve_similar_incidents(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "top_k": request.top_k,
        "results": results,
    }


@router.get("/rag/incidents", summary="List all seeded historical incidents", tags=["rag"])
async def list_historical_incidents():
    """Return the full historical incident corpus used to seed Qdrant."""
    data = json.loads(
        (Path(settings.fixtures_dir) / "incidents" / "historical_incidents.json").read_text()
    )
    return {"count": len(data), "incidents": data}


@router.post("/rag/seed", summary="Re-seed Qdrant with historical incidents", tags=["rag"])
async def reseed_qdrant():
    """Force re-seed Qdrant from fixtures. Useful after adding new historical incidents."""
    from app.services.qdrant_service import _get_client, ensure_collection_seeded
    client = _get_client()
    try:
        client.delete_collection(settings.qdrant_collection)
    except Exception:
        pass
    await ensure_collection_seeded()
    return {"status": "seeded", "collection": settings.qdrant_collection}


# ── Slack ─────────────────────────────────────────────────────────────────────

@router.get(
    "/slack/preview/{scenario_id}",
    summary="Preview Slack message for a scenario",
    tags=["slack"],
)
async def slack_preview(scenario_id: str):
    """
    Generate and return the Slack Block Kit message that would be posted
    for a given scenario — without actually sending it.
    Useful for testing Slack formatting without needing a real webhook.
    """
    from unittest.mock import AsyncMock, patch
    from app.services.slack_service import build_slack_message_preview
    from app.core.models import DiagnosisResult, Severity, FixCategory, RemediationStep
    from datetime import datetime, timezone
    import uuid

    try:
        bundle = load_scenario_bundle(scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    scenario = bundle["scenario"]

    # Build a representative mock DiagnosisResult for preview
    mock_result = DiagnosisResult(
        incident_id=f"INC-PREVIEW-{str(uuid.uuid4())[:6].upper()}",
        scenario_id=scenario_id,
        node=scenario["node"],
        affected_gpu=scenario["affected_gpu"],
        pod=scenario["pod"],
        namespace=scenario["namespace"],
        diagnosed_at=datetime.now(timezone.utc),
        investigation_duration_seconds=11.4,
        severity=Severity(scenario["severity"]),
        root_cause=scenario["expected_root_cause"],
        contributing_factors=["Sustained high GPU load", "ECC error accumulation", "Insufficient cooling headroom"],
        fix_category=FixCategory(scenario["expected_fix_category"]),
        remediation_steps=[
            RemediationStep(step=1, action="Cordon node", command=f"kubectl cordon {scenario['node']}", description="Prevent new pod scheduling"),
            RemediationStep(step=2, action="Drain workloads", command=f"kubectl drain {scenario['node']} --ignore-daemonsets", description="Evict running pods"),
            RemediationStep(step=3, action="Reset GPU", command=f"nvidia-smi --id={scenario['affected_gpu']} --gpu-reset", description="Clear ECC errors"),
            RemediationStep(step=4, action="Uncordon node", command=f"kubectl uncordon {scenario['node']}", description="Re-enable scheduling"),
        ],
        k8s_patch_yaml=bundle.get("k8s_patch", "# K8s patch YAML would appear here"),
        gpu_metrics_summary={},
        log_snippets=[],
        alert_summary=f"GPU incident on {scenario['node']}",
        agent_trace=[
            "fetch_context: 1 unhealthy GPU, 3 log snippets",
            f"rag_retrieve: 2 similar incidents found",
            f"analyze_signals: primary anomaly detected (RAG-augmented)",
            f"root_cause: {scenario['severity'].upper()} — {scenario['expected_fix_category']}",
            "recommend_fix: 4 remediation steps generated",
        ],
        confidence=0.93,
        similar_incidents=[],
    )

    preview = build_slack_message_preview(mock_result)
    return {
        "scenario_id": scenario_id,
        "slack_enabled": settings.slack_enabled,
        "webhook_configured": bool(settings.slack_webhook_url),
        "preview": preview,
    }
