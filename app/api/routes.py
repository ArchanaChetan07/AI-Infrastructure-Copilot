"""
API routes — all endpoints for the GPU Copilot service.
"""

from fastapi import APIRouter, HTTPException

from app.agent.graph import run_diagnosis
from app.core.config import settings
from app.core.fixtures import load_scenario_bundle
from app.core.logger import get_logger
from app.core.models import DiagnoseRequest, DiagnosisResult

logger = get_logger(__name__)
router = APIRouter()


@router.post("/diagnose", response_model=DiagnosisResult, summary="Diagnose a GPU incident")
async def diagnose(request: DiagnoseRequest) -> DiagnosisResult:
    """
    Trigger the AI diagnosis agent for a GPU incident.

    - In **mock mode** (default), pass a `scenario_id` to use fixture data.
    - In **live mode**, pass a full `alert_payload` from Alertmanager.

    Returns a full diagnosis with root cause, remediation steps, and K8s patch YAML.
    """
    if settings.use_mock_data:
        scenario_id = request.scenario_id or "gpu_thermal_throttle_ecc"
        logger.info(f"Mock mode: loading scenario '{scenario_id}'")

        try:
            bundle = load_scenario_bundle(scenario_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        scenario = bundle["scenario"]
        metrics = bundle["metrics"]
        alert = bundle["alert"]
        raw_logs = bundle["logs"]
        k8s_patch_template = bundle["k8s_patch"]

        alert_summary = (
            f"{alert.commonAnnotations.get('summary', '')} "
            f"Node: {alert.commonLabels.get('node', 'unknown')}. "
            f"Pod: {alert.alerts[0].labels.pod if alert.alerts else 'unknown'}. "
            f"Namespace: {alert.alerts[0].labels.namespace if alert.alerts else 'unknown'}."
        )

    else:
        # Live mode — use the incoming alert to drive fixture selection
        # In a real deployment, this would call Prometheus/DCGM/k8s APIs
        if not request.alert_payload:
            raise HTTPException(
                status_code=400,
                detail="alert_payload required when use_mock_data=False"
            )
        alert = request.alert_payload
        scenario_id = "live"
        alert_summary = (
            f"{alert.commonAnnotations.get('summary', '')} "
            f"Node: {alert.commonLabels.get('node', 'unknown')}."
        )
        # TODO Day 2: fetch live metrics from Prometheus + K8s
        raise HTTPException(
            status_code=501,
            detail="Live mode not yet implemented. Set use_mock_data=True and pass a scenario_id."
        )

    result = await run_diagnosis(
        scenario_id=scenario_id,
        metrics=metrics,
        alert_summary=alert_summary,
        raw_logs=raw_logs,
        k8s_patch_template=k8s_patch_template,
    )
    return result


@router.get("/scenarios", summary="List available demo scenarios")
async def list_scenarios():
    """List all available mock scenarios for demo/testing."""
    import json
    from pathlib import Path
    data = json.loads((Path(settings.fixtures_dir) / "scenarios" / "scenarios.json").read_text())
    return {
        "scenarios": [
            {
                "id": s["id"],
                "name": s["name"],
                "severity": s["severity"],
                "description": s["description"],
            }
            for s in data["scenarios"]
        ]
    }


@router.get("/metrics/mock", summary="Return raw mock GPU metrics")
async def mock_metrics():
    """Return the mock GPU metrics payload (useful for frontend development)."""
    from app.core.fixtures import load_gpu_metrics
    metrics = load_gpu_metrics()
    return metrics


@router.get("/alert/mock", summary="Return raw mock alert payload")
async def mock_alert():
    """Return the mock Alertmanager webhook payload."""
    from app.core.fixtures import load_alert_payload
    alert = load_alert_payload()
    return alert
