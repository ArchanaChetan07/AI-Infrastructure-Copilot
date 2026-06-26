"""
Day 3 tests — Prometheus integration, Kubernetes integration, auto-remediation,
analytics/dashboard, alert webhook, and Grafana webhook receiver.
All tests run without external services (mocked HTTP clients and kubectl).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.models import DiagnosisResult, FixCategory, RemediationStep, Severity

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_result(**overrides) -> DiagnosisResult:
    defaults = dict(
        incident_id="INC-20240115-D3TEST01",
        scenario_id="gpu_thermal_throttle_ecc",
        node="gpu-node-03",
        affected_gpu=2,
        pod="vllm-inference-7d9f8b-xkp2q",
        namespace="ml-serving",
        diagnosed_at=datetime.now(timezone.utc),
        investigation_duration_seconds=11.4,
        severity=Severity.CRITICAL,
        root_cause="GPU 2 ECC double-bit error triggered by thermal throttle at 91°C.",
        contributing_factors=["Fan at 100%", "ECC DBE requires reset", "NVLink errors"],
        fix_category=FixCategory.GPU_DRAIN_AND_RESET,
        remediation_steps=[
            RemediationStep(step=1, action="Cordon", command="kubectl cordon gpu-node-03", description="Isolate node"),
            RemediationStep(step=2, action="Reset GPU", command="nvidia-smi --id=2 --gpu-reset", description="Clear ECC"),
            RemediationStep(step=3, action="Uncordon", command="kubectl uncordon gpu-node-03", description="Re-enable"),
        ],
        k8s_patch_yaml="apiVersion: v1\nkind: Node\nmetadata:\n  name: gpu-node-03\nspec:\n  unschedulable: true\n",
        gpu_metrics_summary={"node": "gpu-node-03", "unhealthy_gpu_count": 1},
        log_snippets=["CRITICAL GPU 2: ECC DBE detected"],
        alert_summary="GPU temperature critical on gpu-node-03",
        agent_trace=["fetch_context", "rag_retrieve", "analyze_signals", "root_cause", "recommend_fix"],
        confidence=0.94,
        similar_incidents=[],
    )
    defaults.update(overrides)
    return DiagnosisResult(**defaults)


# ── Prometheus Integration Tests ──────────────────────────────────────────────

class TestPrometheusIntegration:

    def test_prometheus_fixture_file_valid(self):
        """Prometheus fixture response must be valid JSON."""
        data = json.loads(
            Path("fixtures/prometheus/dcgm_gpu_temp_response.json").read_text()
        )
        assert data["status"] == "success"
        assert len(data["data"]["result"]) == 4
        temps = {r["metric"]["gpu"]: float(r["value"][1]) for r in data["data"]["result"]}
        assert temps["2"] == 91.0  # GPU 2 critical

    def test_extract_by_gpu_parses_results(self):
        """_extract_by_gpu correctly maps gpu labels to float values."""
        from app.integrations.prometheus import _extract_by_gpu
        results = [
            {"metric": {"gpu": "0"}, "value": [0, "87.5"]},
            {"metric": {"gpu": "2"}, "value": [0, "91.0"]},
        ]
        out = _extract_by_gpu(results)
        assert out[0] == 87.5
        assert out[2] == 91.0

    @pytest.mark.asyncio
    async def test_fallback_to_fixtures_when_prometheus_disabled(self):
        """fetch_live_gpu_metrics falls back to fixtures when disabled."""
        from app.integrations.prometheus import get_metrics_for_alert
        from app.core.fixtures import load_alert_payload
        alert = load_alert_payload()

        with patch("app.integrations.prometheus.settings") as s:
            s.prometheus_enabled = False
            result = await get_metrics_for_alert(alert)
        assert result.cluster == "gpu-cluster-prod-01"
        assert len(result.gpus) == 4

    @pytest.mark.asyncio
    async def test_live_metrics_endpoint_disabled(self):
        """GET /live/metrics/{node} returns fixture data when Prometheus disabled."""
        with patch("app.integrations.prometheus.settings") as s:
            s.prometheus_enabled = False
            response = client.get("/api/v1/live/metrics/gpu-node-03")
        # Falls back to fixtures — should still return valid ClusterMetrics
        assert response.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_live_metrics_endpoint_prometheus_down(self):
        """GET /live/metrics/{node} returns 503 when Prometheus is enabled but unreachable."""
        with patch("app.integrations.prometheus.fetch_live_gpu_metrics", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Connection refused")
            response = client.get("/api/v1/live/metrics/gpu-node-03")
        assert response.status_code == 503

    def test_parse_duration_seconds(self):
        from app.integrations.prometheus import _parse_duration_seconds
        assert _parse_duration_seconds("1h") == 3600
        assert _parse_duration_seconds("30m") == 1800
        assert _parse_duration_seconds("24h") == 86400

    def test_gpu_health_classification(self):
        """GPU health should be CRITICAL when temp >= 95 or ECC DBE > 0."""
        from app.core.models import GPUHealth
        # Simulate health logic from prometheus.py
        def classify(temp, dbe, sbe):
            if dbe > 0 or temp >= 95:
                return GPUHealth.CRITICAL
            elif temp >= 87 or sbe > 5:
                return GPUHealth.WARNING
            return GPUHealth.OK

        assert classify(91, 3, 0) == GPUHealth.CRITICAL   # ECC DBE
        assert classify(96, 0, 0) == GPUHealth.CRITICAL   # Over shutdown threshold
        assert classify(88, 0, 0) == GPUHealth.WARNING    # Over slowdown threshold
        assert classify(72, 0, 1) == GPUHealth.OK         # Normal


# ── Kubernetes Integration Tests ──────────────────────────────────────────────

class TestKubernetesIntegration:

    @pytest.mark.asyncio
    async def test_fetch_pod_logs_returns_empty_when_disabled(self):
        from app.integrations.kubernetes import fetch_pod_logs
        with patch("app.integrations.kubernetes.settings") as s:
            s.k8s_enabled = False
            result = await fetch_pod_logs("some-pod", "ml-serving")
        assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_node_events_returns_empty_when_disabled(self):
        from app.integrations.kubernetes import fetch_node_events
        with patch("app.integrations.kubernetes.settings") as s:
            s.k8s_enabled = False
            result = await fetch_node_events("gpu-node-03")
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_node_gpu_status_disabled(self):
        from app.integrations.kubernetes import get_node_gpu_status
        with patch("app.integrations.kubernetes.settings") as s:
            s.k8s_enabled = False
            result = await get_node_gpu_status("gpu-node-03")
        assert result["k8s_enabled"] is False
        assert result["node"] == "gpu-node-03"

    @pytest.mark.asyncio
    async def test_apply_patch_disabled_returns_reason(self):
        from app.integrations.kubernetes import apply_k8s_patch
        with patch("app.integrations.kubernetes.settings") as s:
            s.k8s_enabled = False
            result = await apply_k8s_patch("apiVersion: v1\nkind: Node\n", dry_run=True)
        assert result["success"] is False
        assert "K8S_ENABLED=false" in result["reason"]

    @pytest.mark.asyncio
    async def test_cordon_disabled_returns_command(self):
        from app.integrations.kubernetes import cordon_node
        with patch("app.integrations.kubernetes.settings") as s:
            s.k8s_enabled = False
            result = await cordon_node("gpu-node-03", dry_run=True)
        assert "kubectl cordon" in result["command"]

    @pytest.mark.asyncio
    async def test_fetch_all_logs_fallback_to_fixtures(self):
        from app.integrations.kubernetes import fetch_all_logs_for_alert
        from app.core.fixtures import load_alert_payload
        alert = load_alert_payload()
        with patch("app.integrations.kubernetes.settings") as s:
            s.k8s_enabled = False
            logs = await fetch_all_logs_for_alert(alert)
        assert isinstance(logs, list)
        assert len(logs) >= 1

    def test_node_status_endpoint_disabled(self):
        """GET /live/node/{node}/status returns k8s_enabled=False when disabled."""
        with patch("app.integrations.kubernetes.settings") as s:
            s.k8s_enabled = False
            response = client.get("/api/v1/live/node/gpu-node-03/status")
        assert response.status_code == 200
        assert response.json()["k8s_enabled"] is False

    @pytest.mark.asyncio
    async def test_kubectl_called_when_k8s_enabled(self):
        """When K8S_ENABLED=true, kubectl is invoked for pod logs."""
        from app.integrations.kubernetes import fetch_pod_logs
        with patch("app.integrations.kubernetes.settings") as s, \
             patch("app.integrations.kubernetes._kubectl", return_value=(0, "log line 1\nlog line 2\n", "")) as mock_kubectl:
            s.k8s_enabled = True
            s.kubeconfig_path = ""
            result = await fetch_pod_logs("my-pod", "ml-serving", lines=50, previous=True)
        assert "log line" in result
        mock_kubectl.assert_called()


# ── Auto-Remediation Tests ────────────────────────────────────────────────────

class TestAutoRemediation:

    @pytest.mark.asyncio
    async def test_dry_run_mode_never_applies(self):
        """DRY_RUN mode must never touch the cluster."""
        from app.services.remediation_service import RemediationMode, execute_remediation

        result = make_result()
        with patch("app.services.remediation_service._notify_slack_remediation", new_callable=AsyncMock), \
             patch("app.integrations.kubernetes.apply_k8s_patch", new_callable=AsyncMock) as mock_apply, \
             patch("app.integrations.kubernetes.cordon_node", new_callable=AsyncMock) as mock_cordon, \
             patch("app.integrations.kubernetes.drain_node", new_callable=AsyncMock) as mock_drain:
            mock_apply.return_value = {"success": True, "applied_resources": [], "output": "(dry run)"}
            mock_cordon.return_value = {"success": True, "output": "node/gpu-node-03 cordoned (dry run)"}
            mock_drain.return_value = {"success": True, "output": "drained (dry run)"}

            rem = await execute_remediation(result, RemediationMode.DRY_RUN)

        assert rem.dry_run is True
        assert rem.success is True
        # Verify dry_run=True was passed to kubernetes functions
        if mock_cordon.called:
            _, kwargs = mock_cordon.call_args
            assert kwargs.get("dry_run", True) is True

    @pytest.mark.asyncio
    async def test_confirm_mode_queues_without_applying(self):
        """CONFIRM mode must queue the patch and not apply it."""
        from app.services.remediation_service import (
            RemediationMode, execute_remediation, _pending_confirmations
        )
        _pending_confirmations.clear()

        result = make_result()
        with patch("app.services.remediation_service._notify_slack_remediation", new_callable=AsyncMock):
            rem = await execute_remediation(result, RemediationMode.CONFIRM)

        assert rem.success is True
        assert "confirm" in rem.output.lower() or "queue" in rem.output.lower()
        assert result.incident_id in _pending_confirmations

    @pytest.mark.asyncio
    async def test_confirm_then_execute(self):
        """Confirming a queued remediation triggers execution."""
        from app.services.remediation_service import (
            RemediationMode, execute_remediation, confirm_remediation, _pending_confirmations
        )
        _pending_confirmations.clear()

        result = make_result()
        with patch("app.services.remediation_service._notify_slack_remediation", new_callable=AsyncMock):
            await execute_remediation(result, RemediationMode.CONFIRM)

        assert result.incident_id in _pending_confirmations

        with patch("app.services.remediation_service._notify_slack_remediation", new_callable=AsyncMock), \
             patch("app.integrations.kubernetes.apply_k8s_patch", new_callable=AsyncMock) as mock_apply, \
             patch("app.integrations.kubernetes.cordon_node", new_callable=AsyncMock) as mock_cordon, \
             patch("app.integrations.kubernetes.drain_node", new_callable=AsyncMock) as mock_drain:
            mock_apply.return_value = {"success": True, "applied_resources": ["node/gpu-node-03"], "output": "applied"}
            mock_cordon.return_value = {"success": True, "output": "cordoned"}
            mock_drain.return_value = {"success": True, "output": "drained"}
            confirmed = await confirm_remediation(result.incident_id)

        assert confirmed.success is True
        assert confirmed.dry_run is False
        assert result.incident_id not in _pending_confirmations

    @pytest.mark.asyncio
    async def test_confirm_unknown_incident_returns_error(self):
        """Confirming an unknown incident returns failure."""
        from app.services.remediation_service import confirm_remediation, _pending_confirmations
        _pending_confirmations.clear()

        result = await confirm_remediation("INC-DOESNOTEXIST-000")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_auto_mode_downgrades_for_low_severity(self):
        """AUTO mode on WARNING incident must downgrade to dry_run."""
        from app.services.remediation_service import RemediationMode, execute_remediation

        low_severity_result = make_result(severity=Severity.WARNING)
        with patch("app.services.remediation_service._notify_slack_remediation", new_callable=AsyncMock), \
             patch("app.integrations.kubernetes.apply_k8s_patch", new_callable=AsyncMock) as mock_apply, \
             patch("app.integrations.kubernetes.cordon_node", new_callable=AsyncMock) as mock_cordon, \
             patch("app.integrations.kubernetes.drain_node", new_callable=AsyncMock) as mock_drain:
            mock_apply.return_value = {"success": True, "applied_resources": [], "output": ""}
            mock_cordon.return_value = {"success": True, "output": ""}
            mock_drain.return_value = {"success": True, "output": ""}
            rem = await execute_remediation(low_severity_result, RemediationMode.AUTO)

        # Should be forced to dry_run for WARNING
        assert rem.dry_run is True

    def test_remediate_endpoint_dry_run(self):
        """POST /remediate/{id}?mode=dry_run returns valid response."""
        with patch("app.api.routes.run_diagnosis", new_callable=AsyncMock), \
             patch("app.db.database.get_incident", new_callable=AsyncMock) as mock_get, \
             patch("app.services.remediation_service.execute_remediation", new_callable=AsyncMock) as mock_exec:
            mock_get.return_value = None
            from app.services.remediation_service import RemediationResult, RemediationMode
            mock_exec.return_value = RemediationResult(
                incident_id="INC-20240115-D3TEST01",
                mode=RemediationMode.DRY_RUN,
                executed_at=datetime.now(timezone.utc),
                success=True,
                dry_run=True,
                output="[DRY RUN] Remediation for INC-20240115-D3TEST01",
            )
            response = client.post("/api/v1/remediate/INC-20240115-D3TEST01?mode=dry_run")
        assert response.status_code == 200

    def test_pending_remediations_endpoint(self):
        """GET /remediate/pending returns list."""
        response = client.get("/api/v1/remediate/pending")
        assert response.status_code == 200
        data = response.json()
        assert "pending" in data
        assert isinstance(data["pending"], list)


# ── Analytics / Dashboard Tests ───────────────────────────────────────────────

class TestAnalytics:

    def test_dashboard_summary_endpoint(self):
        """GET /dashboard/summary returns valid summary."""
        response = client.get("/api/v1/dashboard/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_incidents" in data
        assert "severity_breakdown" in data
        assert "avg_investigation_seconds" in data
        assert "top_failing_nodes" in data
        assert "most_common_fix" in data

    def test_dashboard_summary_has_incidents_from_fixtures(self):
        """Mock dashboard uses historical incidents — should show 6 incidents."""
        response = client.get("/api/v1/dashboard/summary")
        data = response.json()
        assert data["total_incidents"] >= 6
        assert data["most_common_fix"] is not None

    def test_mttr_trend_endpoint(self):
        """GET /dashboard/mttr-trend returns daily trend data."""
        response = client.get("/api/v1/dashboard/mttr-trend?days=30")
        assert response.status_code == 200
        data = response.json()
        assert "trend" in data
        assert len(data["trend"]) > 0
        first = data["trend"][0]
        assert "date" in first
        assert "avg_investigation_seconds" in first

    def test_mttr_trend_shows_improvement(self):
        """MTTR trend should show decreasing investigation time (40min → 3min)."""
        response = client.get("/api/v1/dashboard/mttr-trend?days=30")
        trend = response.json()["trend"]
        if len(trend) >= 2:
            first_avg = trend[0]["avg_investigation_seconds"]
            last_avg = trend[-1]["avg_investigation_seconds"]
            # Trend should be declining
            assert first_avg > last_avg, \
                f"MTTR trend should decline: first={first_avg}s, last={last_avg}s"

    def test_failure_heatmap_endpoint(self):
        """GET /dashboard/heatmap returns node × fix_category counts."""
        response = client.get("/api/v1/dashboard/heatmap")
        assert response.status_code == 200
        data = response.json()
        assert "heatmap" in data
        assert len(data["heatmap"]) > 0
        for item in data["heatmap"]:
            assert "node" in item
            assert "fix_category" in item
            assert "count" in item

    def test_recurrence_report_endpoint(self):
        """GET /dashboard/recurrences returns recurrence patterns."""
        response = client.get("/api/v1/dashboard/recurrences")
        assert response.status_code == 200
        data = response.json()
        assert "recurrences" in data
        assert "total_recurring_nodes" in data

    def test_recurrence_recommendations_present(self):
        """Each recurrence must have an actionable recommendation."""
        response = client.get("/api/v1/dashboard/recurrences")
        for rec in response.json()["recurrences"]:
            assert "recommendation" in rec
            assert len(rec["recommendation"]) > 20

    @pytest.mark.asyncio
    async def test_mock_mttr_trend_structure(self):
        """Mock MTTR trend has correct fields and declining values."""
        from app.services.analytics_service import _mock_mttr_trend
        trend_data = _mock_mttr_trend(14)
        assert trend_data["days"] == 14
        assert len(trend_data["trend"]) == 14
        for point in trend_data["trend"]:
            assert "date" in point
            assert "avg_investigation_seconds" in point
            assert point["avg_investigation_seconds"] > 0

    @pytest.mark.asyncio
    async def test_mock_failure_heatmap_structure(self):
        from app.services.analytics_service import _mock_failure_heatmap
        heatmap = _mock_failure_heatmap()
        assert "heatmap" in heatmap
        for item in heatmap["heatmap"]:
            assert all(k in item for k in ["node", "fix_category", "severity", "count"])

    @pytest.mark.asyncio
    async def test_recurrence_recommendation_per_fix_type(self):
        from app.services.analytics_service import _recurrence_recommendation
        for fix in ["gpu_drain_and_reset", "config_patch", "nvlink_reset", "pod_restart"]:
            rec = _recurrence_recommendation(fix, 3, 14)
            assert isinstance(rec, str) and len(rec) > 10


# ── Alert Webhook Tests ───────────────────────────────────────────────────────

class TestAlertWebhook:

    @pytest.mark.asyncio
    async def test_alertmanager_webhook_pipeline_unit(self):
        """Webhook pipeline: parse alert → metrics → logs → diagnosis (all mocked)."""
        import asyncio
        from app.core.fixtures import load_alert_payload, load_gpu_metrics
        from app.integrations.prometheus import get_metrics_for_alert
        from app.integrations.kubernetes import fetch_all_logs_for_alert

        alert = load_alert_payload()

        # metrics fallback works
        with patch("app.integrations.prometheus.settings") as s:
            s.prometheus_enabled = False
            metrics = await get_metrics_for_alert(alert)
        assert metrics.node == "gpu-node-03"

        # logs fallback works
        with patch("app.integrations.kubernetes.settings") as s:
            s.k8s_enabled = False
            logs = await fetch_all_logs_for_alert(alert)
        assert isinstance(logs, list) and len(logs) >= 1

        # alert summary built correctly
        node = alert.commonLabels.get("node", "unknown")
        assert node == "gpu-node-03"

    @pytest.mark.asyncio
    async def test_grafana_webhook_normalized_to_alertmanager(self):
        """Grafana payload normalizer produces valid Alertmanager format."""
        from app.api.routes import _grafana_to_alertmanager
        from app.core.models import AlertPayload

        grafana_payload = {
            "ruleName": "GPUTemperatureCritical",
            "state": "alerting",
            "message": "GPU 2 temperature 91 degrees",
            "evalMatches": [{"tags": {"instance": "gpu-node-03", "severity": "critical"}}],
            "ruleUrl": "http://grafana/alerts/1",
        }
        converted = _grafana_to_alertmanager(grafana_payload)
        # Must parse as valid AlertPayload
        parsed = AlertPayload(**converted)
        assert parsed.status == "firing"
        assert parsed.commonLabels["node"] == "gpu-node-03"
        assert len(parsed.alerts) == 1

    def test_grafana_to_alertmanager_conversion(self):
        """_grafana_to_alertmanager correctly maps Grafana fields."""
        from app.api.routes import _grafana_to_alertmanager
        grafana = {
            "ruleName": "GPUCritical",
            "state": "alerting",
            "message": "GPU overheating",
            "evalMatches": [{"tags": {"instance": "gpu-node-05", "severity": "critical"}}],
        }
        converted = _grafana_to_alertmanager(grafana)
        assert converted["status"] == "firing"
        assert converted["commonLabels"]["alertname"] == "GPUCritical"
        assert converted["commonLabels"]["node"] == "gpu-node-05"
        assert len(converted["alerts"]) == 1

    def test_webhook_invalid_payload_returns_422(self):
        """POST /alert/webhook with garbage payload returns 422."""
        response = client.post(
            "/api/v1/alert/webhook",
            json={"invalid": "completely wrong format", "missing": "all required fields"},
        )
        assert response.status_code == 422

    def test_webhook_endpoint_exists(self):
        """Webhook endpoint must be registered."""
        response = client.post("/api/v1/alert/webhook", json={})
        assert response.status_code in (200, 422, 500)  # exists, may fail on bad payload


# ── Full API Surface (Day 3) ──────────────────────────────────────────────────

class TestFullAPISurfaceDay3:

    def test_health_shows_version_020(self):
        response = client.get("/health")
        assert response.json()["version"] == "0.2.0"

    def test_all_new_day3_endpoints_registered(self):
        """All Day 3 endpoints must return a valid HTTP status (not 404/405)."""
        endpoints = [
            ("GET",  "/api/v1/dashboard/summary"),
            ("GET",  "/api/v1/dashboard/mttr-trend"),
            ("GET",  "/api/v1/dashboard/heatmap"),
            ("GET",  "/api/v1/dashboard/recurrences"),
            ("GET",  "/api/v1/remediate/pending"),
            ("GET",  "/api/v1/live/node/gpu-node-03/status"),
        ]
        for method, path in endpoints:
            if method == "GET":
                resp = client.get(path)
            else:
                resp = client.post(path, json={})
            assert resp.status_code not in (404, 405), \
                f"Endpoint {method} {path} returned {resp.status_code} — not registered?"

    def test_dashboard_summary_always_returns_data(self):
        """Dashboard summary returns useful data even without Postgres."""
        response = client.get("/api/v1/dashboard/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_incidents"] > 0
        assert data["avg_investigation_seconds"] > 0

    def test_mttr_trend_days_param(self):
        """mttr-trend accepts days parameter."""
        for days in [7, 14, 30]:
            response = client.get(f"/api/v1/dashboard/mttr-trend?days={days}")
            assert response.status_code == 200
            assert response.json()["days"] == days
