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

from fastapi import APIRouter, BackgroundTasks, HTTPException
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


# ── Day 3: Alertmanager / Grafana Webhook ─────────────────────────────────────

@router.post(
    "/alert/webhook",
    summary="Receive Alertmanager or Grafana alert webhook",
    tags=["webhooks"],
)
async def alert_webhook(payload: dict):
    """
    Direct webhook receiver for Alertmanager and Grafana alerts.

    Configure in Alertmanager as:
      receivers:
        - name: gpu-copilot
          webhook_configs:
            - url: http://gpu-copilot:8000/api/v1/alert/webhook

    Automatically:
      1. Parses the alert payload
      2. Fetches live GPU metrics (Prometheus) or fixtures
      3. Fetches live pod logs (Kubernetes) or fixtures
      4. Runs the 5-node LangGraph diagnosis pipeline
      5. Posts result to Slack
      6. Persists to Postgres
    """
    from app.core.models import AlertPayload

    # Normalize Grafana vs Alertmanager format
    if "alerts" not in payload and "ruleName" in payload:
        # Grafana unified alerting format
        payload = _grafana_to_alertmanager(payload)

    try:
        alert = AlertPayload(**payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid alert payload: {e}")

    node = alert.commonLabels.get("node", "gpu-node-03")
    logger.info(f"Webhook received: {alert.commonLabels.get('alertname')} on {node}")

    # Fetch metrics and logs (live or fixture)
    from app.integrations.prometheus import get_metrics_for_alert
    from app.integrations.kubernetes import fetch_all_logs_for_alert
    from app.core.fixtures import load_k8s_patch

    metrics = await get_metrics_for_alert(alert)
    raw_logs = await fetch_all_logs_for_alert(alert)
    k8s_patch_template = load_k8s_patch("fixtures/expected/k8s_patch_gpu_drain.yaml")

    alert_summary = (
        f"{alert.commonAnnotations.get('summary', '')} "
        f"Node: {node}. "
        + (f"Pod: {alert.alerts[0].labels.pod}." if alert.alerts and alert.alerts[0].labels.pod else "")
    )

    result = await run_diagnosis(
        scenario_id="live_webhook",
        metrics=metrics,
        alert_summary=alert_summary,
        raw_logs=raw_logs,
        k8s_patch_template=k8s_patch_template,
    )
    return result


def _grafana_to_alertmanager(grafana: dict) -> dict:
    """Convert Grafana unified alerting webhook to Alertmanager format."""
    return {
        "version": "4",
        "status": "firing" if grafana.get("state") == "alerting" else "resolved",
        "receiver": "gpu-copilot",
        "groupLabels": {"alertname": grafana.get("ruleName", "GrafanaAlert")},
        "commonLabels": {
            "alertname": grafana.get("ruleName", "GrafanaAlert"),
            "severity": grafana.get("evalMatches", [{}])[0].get("tags", {}).get("severity", "critical"),
            "node": grafana.get("evalMatches", [{}])[0].get("tags", {}).get("instance", "unknown"),
        },
        "commonAnnotations": {
            "summary": grafana.get("message", "Grafana alert fired"),
            "description": grafana.get("ruleUrl", ""),
        },
        "alerts": [{
            "status": "firing",
            "labels": {
                "alertname": grafana.get("ruleName", "GrafanaAlert"),
                "severity": "critical",
            },
            "annotations": {"summary": grafana.get("message", "")},
            "startsAt": "2024-01-15T14:28:00Z",
            "fingerprint": "grafana-webhook",
        }],
    }


# ── Day 3: Auto-Remediation ───────────────────────────────────────────────────

@router.post(
    "/remediate/{incident_id}",
    summary="Execute remediation for a diagnosed incident",
    tags=["remediation"],
)
async def execute_remediation(
    incident_id: str,
    mode: str = "dry_run",
    patch_yaml: str | None = None,
):
    """
    Execute the AI-generated K8s patch for a diagnosed incident.

    Modes:
    - `dry_run`  (default) — kubectl apply --dry-run=server, nothing changes
    - `confirm`  — queues the patch, waits for POST /remediate/{id}/confirm
    - `auto`     — applies immediately (CRITICAL/HIGH only)

    Always safe to call with dry_run=true — shows exactly what would happen.
    """
    from app.services.remediation_service import (
        RemediationMode, execute_remediation as do_remediate,
    )
    from app.db.database import get_incident

    try:
        rem_mode = RemediationMode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Use: dry_run, confirm, auto")

    incident_data = await get_incident(incident_id)
    if not incident_data and not settings.use_mock_data:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    # Build a minimal DiagnosisResult from stored data for remediation
    from app.core.models import DiagnosisResult, Severity, FixCategory, RemediationStep
    from datetime import datetime, timezone

    if incident_data:
        mock_result = DiagnosisResult(
            incident_id=incident_id,
            scenario_id=incident_data.get("scenario_id"),
            node=incident_data["node"],
            affected_gpu=incident_data.get("affected_gpu"),
            pod=incident_data.get("pod"),
            namespace=incident_data.get("namespace"),
            diagnosed_at=datetime.now(timezone.utc),
            investigation_duration_seconds=0,
            severity=Severity(incident_data["severity"]),
            root_cause=incident_data["root_cause"],
            contributing_factors=incident_data.get("contributing_factors", []),
            fix_category=FixCategory(incident_data["fix_category"]),
            remediation_steps=[],
            k8s_patch_yaml=patch_yaml or "",
            gpu_metrics_summary={},
            log_snippets=[],
            alert_summary="",
            agent_trace=[],
            confidence=incident_data.get("confidence", 0.9),
        )
    else:
        # Demo mode — build from request params
        mock_result = DiagnosisResult(
            incident_id=incident_id,
            scenario_id="demo",
            node="gpu-node-03",
            affected_gpu=2,
            pod=None, namespace=None,
            diagnosed_at=datetime.now(timezone.utc),
            investigation_duration_seconds=0,
            severity=Severity.CRITICAL,
            root_cause="Demo remediation",
            contributing_factors=[],
            fix_category=FixCategory.GPU_DRAIN_AND_RESET,
            remediation_steps=[],
            k8s_patch_yaml=patch_yaml or "",
            gpu_metrics_summary={},
            log_snippets=[],
            alert_summary="",
            agent_trace=[],
            confidence=0.9,
        )

    return await do_remediate(mock_result, rem_mode, patch_yaml)


@router.post(
    "/remediate/{incident_id}/confirm",
    summary="Confirm and execute a queued remediation",
    tags=["remediation"],
)
async def confirm_remediation(incident_id: str):
    """
    Confirm and execute a remediation previously queued with mode=confirm.
    This is the human-in-the-loop approval step.
    """
    from app.services.remediation_service import confirm_remediation as do_confirm
    return await do_confirm(incident_id)


@router.get(
    "/remediate/pending",
    summary="List remediations waiting for confirmation",
    tags=["remediation"],
)
async def list_pending():
    from app.services.remediation_service import list_pending_confirmations
    return {"pending": await list_pending_confirmations()}


# ── Day 3: Dashboard Analytics ────────────────────────────────────────────────

@router.get(
    "/dashboard/summary",
    summary="Ops dashboard summary — incident counts, MTTR, top nodes",
    tags=["analytics"],
)
async def dashboard_summary():
    """
    Summary stats for the ops dashboard.
    Returns total incidents, severity breakdown, top failing nodes, avg MTTR.
    Uses fixture data when Postgres is disabled — always returns useful numbers.
    """
    from app.services.analytics_service import get_dashboard_summary
    return await get_dashboard_summary()


@router.get(
    "/dashboard/mttr-trend",
    summary="MTTR trend over time",
    tags=["analytics"],
)
async def mttr_trend(days: int = 30):
    """
    Daily average investigation time over the past N days.
    Shows the '40 min → 3 min' improvement curve.
    """
    from app.services.analytics_service import get_mttr_trend
    return await get_mttr_trend(days)


@router.get(
    "/dashboard/heatmap",
    summary="Node × fix_category failure heatmap",
    tags=["analytics"],
)
async def failure_heatmap():
    """Incident count by node and fix category — reveals problem hardware."""
    from app.services.analytics_service import get_failure_heatmap
    return await get_failure_heatmap()


@router.get(
    "/dashboard/recurrences",
    summary="Recurring incidents — same node, same failure type",
    tags=["analytics"],
)
async def recurrence_report():
    """
    Identify nodes with recurring failures of the same type.
    Includes actionable recommendations per recurrence pattern.
    """
    from app.services.analytics_service import get_recurrence_report
    return await get_recurrence_report()


# ── Day 3: Live Infrastructure ────────────────────────────────────────────────

@router.get(
    "/live/metrics/{node}",
    summary="Fetch live GPU metrics from Prometheus",
    tags=["live"],
)
async def live_metrics(node: str):
    """
    Query real GPU metrics from Prometheus/DCGM for a given node.
    Requires PROMETHEUS_ENABLED=true and a running Prometheus instance.
    Falls back to fixture data when disabled.
    """
    from app.integrations.prometheus import fetch_live_gpu_metrics
    try:
        metrics = await fetch_live_gpu_metrics(node)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Prometheus unavailable: {e}")


@router.get(
    "/live/metrics/{node}/gpu/{gpu_id}/trends",
    summary="GPU metric trends (temperature, utilization, ECC) over time",
    tags=["live"],
)
async def gpu_trends(node: str, gpu_id: int, duration: str = "1h"):
    """
    Time-series trend data for a single GPU.
    Useful for sparklines in the ops dashboard.
    duration: 15m, 1h, 6h, 24h
    """
    from app.integrations.prometheus import fetch_gpu_trends
    return await fetch_gpu_trends(node, gpu_id, duration)


@router.get(
    "/live/node/{node}/status",
    summary="Live Kubernetes node GPU status",
    tags=["live"],
)
async def node_status(node: str):
    """
    Check allocatable vs capacity GPU count on a Kubernetes node.
    Requires K8S_ENABLED=true.
    """
    from app.integrations.kubernetes import get_node_gpu_status
    return await get_node_gpu_status(node)


# ── Day 4: Async Job Queue ─────────────────────────────────────────────────────

@router.post(
    "/jobs/diagnose",
    summary="Submit async diagnosis job — returns immediately",
    tags=["jobs"],
)
async def submit_diagnosis_job(
    request: DiagnoseRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a diagnosis job that runs in the background.
    Returns a job_id immediately — poll GET /jobs/{job_id} for status/result.

    Use this instead of POST /diagnose when you want non-blocking behaviour
    (i.e. your HTTP client has a 30s timeout but diagnosis takes 45s).
    """
    from fastapi import BackgroundTasks
    from app.services.job_queue import create_job, run_job
    from app.core.fixtures import load_scenario_bundle

    scenario_id = request.scenario_id or "gpu_thermal_throttle_ecc"
    bundle = load_scenario_bundle(scenario_id)
    alert = bundle["alert"]
    alert_summary = (
        f"{alert.commonAnnotations.get('summary', '')} "
        f"Node: {alert.commonLabels.get('node', 'unknown')}."
    )

    job = create_job(scenario_id=scenario_id)

    background_tasks.add_task(
        run_job,
        job_id=job.job_id,
        scenario_id=scenario_id,
        metrics=bundle["metrics"],
        alert_summary=alert_summary,
        raw_logs=bundle["logs"],
        k8s_patch_template=bundle["k8s_patch"],
    )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "poll_url": f"/api/v1/jobs/{job.job_id}",
        "message": f"Job queued. Poll {'/api/v1/jobs/' + job.job_id} for status.",
    }


@router.get("/jobs/{job_id}", summary="Get job status and result", tags=["jobs"])
async def get_job(job_id: str):
    """Poll a background diagnosis job by ID."""
    from app.services.job_queue import get_job as _get_job
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.get("/jobs", summary="List recent jobs", tags=["jobs"])
async def list_jobs(limit: int = 20):
    """List the most recent diagnosis jobs with their statuses."""
    from app.services.job_queue import list_jobs as _list_jobs
    return {"jobs": _list_jobs(limit=limit)}


# ── Day 4: Runbook Export ──────────────────────────────────────────────────────

@router.get(
    "/runbook/{scenario_id}",
    summary="Generate Markdown runbook for a scenario",
    tags=["runbook"],
)
async def generate_runbook(scenario_id: str):
    """
    Generate a full Markdown incident runbook for a demo scenario.
    In production: call POST /runbook with an incident_id from Postgres.
    Returns Markdown text ready to paste into Confluence/Notion/GitHub Wiki.
    """
    from app.services.runbook_service import generate_runbook as _gen, generate_runbook_filename
    from app.core.models import DiagnosisResult, Severity, FixCategory, RemediationStep
    from datetime import datetime, timezone
    import uuid

    try:
        bundle = load_scenario_bundle(scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    scenario = bundle["scenario"]
    metrics = bundle["metrics"]

    # Build a representative DiagnosisResult for preview
    gpus = metrics.gpus
    gpu_summary = [
        {
            "gpu_id": g.gpu_id, "temp_c": g.temperature_celsius,
            "util_pct": g.utilization_gpu_percent,
            "mem_used_pct": round(g.memory_used_mb / g.memory_total_mb * 100, 1),
            "ecc_dbe": g.ecc_errors_double_bit, "nvlink_errors": g.nvlink_errors,
            "health": g.health.value,
        }
        for g in gpus
    ]

    result = DiagnosisResult(
        incident_id=f"INC-RB-{str(uuid.uuid4())[:6].upper()}",
        scenario_id=scenario_id,
        node=scenario["node"],
        affected_gpu=scenario["affected_gpu"],
        pod=scenario["pod"],
        namespace=scenario["namespace"],
        diagnosed_at=datetime.now(timezone.utc),
        investigation_duration_seconds=11.4,
        severity=Severity(scenario["severity"]),
        root_cause=scenario["expected_root_cause"],
        contributing_factors=["Fan at 100% capacity", "ECC double-bit errors require GPU reset", "NVLink errors preceded thermal event"],
        fix_category=FixCategory(scenario["expected_fix_category"]),
        remediation_steps=[
            RemediationStep(step=1, action="Cordon node", command=f"kubectl cordon {scenario['node']}", description="Prevent new pod scheduling on affected node"),
            RemediationStep(step=2, action="Drain workloads", command=f"kubectl drain {scenario['node']} --ignore-daemonsets", description="Gracefully evict all running pods"),
            RemediationStep(step=3, action="Reset GPU", command=f"nvidia-smi --id={scenario['affected_gpu']} --gpu-reset", description="Clear uncorrectable ECC errors"),
            RemediationStep(step=4, action="Uncordon node", command=f"kubectl uncordon {scenario['node']}", description="Re-enable node for scheduling after GPU verified healthy"),
        ],
        k8s_patch_yaml=bundle.get("k8s_patch", ""),
        gpu_metrics_summary={"cluster": metrics.cluster, "node": metrics.node, "gpus": gpu_summary, "unhealthy_gpu_count": sum(1 for g in gpus if g.health.value != "OK")},
        log_snippets=["CRITICAL GPU 2: ECC DBE detected — cudaErrorECCUncorrectable"],
        alert_summary=f"GPU incident on {scenario['node']}",
        agent_trace=["fetch_context: 1 unhealthy GPU", "rag_retrieve: 2 similar incidents", "analyze_signals: ECC+thermal anomaly", f"root_cause: {scenario['severity'].upper()}", "recommend_fix: 4 steps"],
        confidence=0.93,
        similar_incidents=[],
    )

    markdown = _gen(result)
    filename = generate_runbook_filename(result)

    from fastapi.responses import Response
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Day 4: Cluster Scan ────────────────────────────────────────────────────────

@router.post(
    "/cluster/scan",
    summary="Scan all GPU nodes and diagnose unhealthy ones",
    tags=["cluster"],
)
async def cluster_scan(cluster: str = "gpu-cluster-prod-01"):
    """
    Scan all GPU nodes in the cluster in parallel.
    Automatically diagnoses any node with unhealthy GPUs.

    In mock mode: simulates a 3-node cluster (node-01 healthy, node-02 warning, node-03 critical).
    In live mode: discovers nodes via kubectl label selector nvidia.com/gpu.

    Returns per-node health status + full DiagnosisResult for each unhealthy node.
    This is the single most impressive demo endpoint.
    """
    from app.services.cluster_scanner import scan_cluster
    return await scan_cluster(cluster)


@router.get(
    "/cluster/nodes",
    summary="List all discovered GPU nodes",
    tags=["cluster"],
)
async def list_cluster_nodes():
    """Discover and list all GPU-capable nodes in the cluster."""
    from app.services.cluster_scanner import _get_cluster_nodes
    nodes = await _get_cluster_nodes()
    return {"nodes": nodes, "count": len(nodes)}
