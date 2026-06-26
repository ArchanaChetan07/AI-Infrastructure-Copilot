"""
Day 1 tests — fixture loading, models, and API endpoints (no LLM calls).
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.core.fixtures import load_gpu_metrics, load_alert_payload, load_scenario
from app.core.models import ClusterMetrics, AlertPayload, GPUHealth


client = TestClient(app)


# ── Fixture tests ─────────────────────────────────────────────────────────────

def test_load_gpu_metrics():
    metrics = load_gpu_metrics()
    assert isinstance(metrics, ClusterMetrics)
    assert metrics.cluster == "gpu-cluster-prod-01"
    assert len(metrics.gpus) == 4
    # GPU 2 should be critical
    gpu2 = next(g for g in metrics.gpus if g.gpu_id == 2)
    assert gpu2.health == GPUHealth.CRITICAL
    assert gpu2.temperature_celsius > gpu2.temperature_threshold_slowdown
    assert gpu2.ecc_errors_double_bit > 0


def test_load_alert_payload():
    alert = load_alert_payload()
    assert isinstance(alert, AlertPayload)
    assert alert.status == "firing"
    assert len(alert.alerts) == 1
    assert alert.alerts[0].labels.gpu_id == "2"
    assert alert.alerts[0].labels.severity == "critical"


def test_load_scenario_valid():
    scenario = load_scenario("gpu_thermal_throttle_ecc")
    assert scenario["id"] == "gpu_thermal_throttle_ecc"
    assert scenario["severity"] == "critical"
    assert scenario["affected_gpu"] == 2


def test_load_scenario_invalid():
    with pytest.raises(ValueError, match="not found"):
        load_scenario("nonexistent_scenario_id")


def test_gpu_metrics_unhealthy_gpus():
    metrics = load_gpu_metrics()
    unhealthy = [g for g in metrics.gpus if g.health != GPUHealth.OK]
    assert len(unhealthy) == 2  # GPU 0 (WARNING) and GPU 2 (CRITICAL)


# ── API endpoint tests ────────────────────────────────────────────────────────

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "gpu-copilot"


def test_list_scenarios():
    response = client.get("/api/v1/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "scenarios" in data
    assert len(data["scenarios"]) == 3
    ids = [s["id"] for s in data["scenarios"]]
    assert "gpu_thermal_throttle_ecc" in ids
    assert "gpu_memory_oom" in ids
    assert "nvlink_degraded" in ids


def test_mock_metrics_endpoint():
    response = client.get("/api/v1/metrics/mock")
    assert response.status_code == 200
    data = response.json()
    assert data["cluster"] == "gpu-cluster-prod-01"
    assert len(data["gpus"]) == 4


def test_mock_alert_endpoint():
    response = client.get("/api/v1/alert/mock")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "firing"


def test_diagnose_returns_404_for_unknown_scenario():
    response = client.post(
        "/api/v1/diagnose",
        json={"scenario_id": "this_does_not_exist"},
    )
    assert response.status_code == 404


def test_diagnose_with_mock_llm():
    """Test the diagnose endpoint with a mocked LLM — no real API call."""
    mock_result = {
        "incident_id": "INC-20240115-TESTABCD",
        "scenario_id": "gpu_thermal_throttle_ecc",
        "node": "gpu-node-03",
        "affected_gpu": 2,
        "pod": "vllm-inference-7d9f8b-xkp2q",
        "namespace": "ml-serving",
        "diagnosed_at": "2024-01-15T14:30:00Z",
        "investigation_duration_seconds": 8.42,
        "severity": "critical",
        "root_cause": "GPU 2 thermal throttle triggered uncorrectable ECC double-bit error.",
        "contributing_factors": ["High ambient temperature", "Sustained 98% GPU utilization", "Insufficient cooling headroom"],
        "fix_category": "gpu_drain_and_reset",
        "remediation_steps": [
            {"step": 1, "action": "Cordon node", "command": "kubectl cordon gpu-node-03", "description": "Prevent new pod scheduling"},
        ],
        "k8s_patch_yaml": "# patch yaml",
        "gpu_metrics_summary": {},
        "log_snippets": [],
        "alert_summary": "GPU temperature critical",
        "agent_trace": ["fetch_context", "analyze_signals", "root_cause", "recommend_fix"],
        "confidence": 0.92,
        "slack_notified": False,
    }

    with patch("app.api.routes.run_diagnosis", new_callable=AsyncMock) as mock_run:
        from app.core.models import DiagnosisResult, Severity, FixCategory, RemediationStep
        from datetime import datetime, timezone
        mock_run.return_value = DiagnosisResult(**mock_result)

        response = client.post(
            "/api/v1/diagnose",
            json={"scenario_id": "gpu_thermal_throttle_ecc"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["severity"] == "critical"
        assert data["fix_category"] == "gpu_drain_and_reset"
        assert data["confidence"] == 0.92
        assert len(data["agent_trace"]) == 4
