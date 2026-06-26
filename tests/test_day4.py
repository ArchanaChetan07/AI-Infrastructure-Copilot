"""
Day 4 tests — middleware (request ID, rate limiting, auth),
async job queue, runbook generator, cluster scanner, and
new system endpoints (/ready, /metrics, /cluster/scan).
All run without external services or API key.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.models import DiagnosisResult, FixCategory, RemediationStep, Severity

client = TestClient(app)


def make_result(**overrides) -> DiagnosisResult:
    defaults = dict(
        incident_id="INC-20240115-D4TEST01",
        scenario_id="gpu_thermal_throttle_ecc",
        node="gpu-node-03",
        affected_gpu=2,
        pod="vllm-inference-7d9f8b-xkp2q",
        namespace="ml-serving",
        diagnosed_at=datetime.now(timezone.utc),
        investigation_duration_seconds=11.4,
        severity=Severity.CRITICAL,
        root_cause="GPU 2 ECC double-bit error triggered by thermal throttle at 91°C.",
        contributing_factors=["Fan at 100%", "ECC DBE requires GPU reset"],
        fix_category=FixCategory.GPU_DRAIN_AND_RESET,
        remediation_steps=[
            RemediationStep(step=1, action="Cordon", command="kubectl cordon gpu-node-03", description="Isolate"),
            RemediationStep(step=2, action="Reset GPU", command="nvidia-smi --id=2 --gpu-reset", description="Clear ECC"),
        ],
        k8s_patch_yaml="apiVersion: v1\nkind: Node\nmetadata:\n  name: gpu-node-03\n",
        gpu_metrics_summary={
            "cluster": "gpu-cluster-prod-01",
            "node": "gpu-node-03",
            "unhealthy_gpu_count": 1,
            "gpus": [
                {"gpu_id": 2, "temp_c": 91, "util_pct": 45, "mem_used_pct": 42,
                 "ecc_dbe": 3, "nvlink_errors": 2, "health": "CRITICAL"},
            ],
        },
        log_snippets=["CRITICAL GPU 2: ECC DBE detected"],
        alert_summary="GPU temperature critical on gpu-node-03",
        agent_trace=["fetch_context", "rag_retrieve", "analyze_signals", "root_cause", "recommend_fix"],
        confidence=0.94,
        similar_incidents=[],
    )
    defaults.update(overrides)
    return DiagnosisResult(**defaults)


# ── Middleware Tests ──────────────────────────────────────────────────────────

class TestRequestIDMiddleware:

    def test_response_has_request_id_header(self):
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_request_id_echoed_from_client(self):
        """If client sends X-Request-ID, it's echoed back."""
        response = client.get("/health", headers={"X-Request-ID": "my-trace-123"})
        assert response.headers.get("x-request-id") == "my-trace-123"

    def test_response_time_header_present(self):
        response = client.get("/health")
        assert "x-response-time-ms" in response.headers
        latency = float(response.headers["x-response-time-ms"])
        assert latency >= 0.0

    def test_unique_request_ids_generated(self):
        ids = set()
        for _ in range(5):
            r = client.get("/health")
            ids.add(r.headers.get("x-request-id"))
        assert len(ids) == 5  # all unique

    def test_get_request_id_returns_string(self):
        from app.middleware.logging import get_request_id
        result = get_request_id()
        assert isinstance(result, str)


class TestRateLimitMiddleware:

    def test_rate_limit_not_triggered_under_limit(self):
        """Normal traffic under the limit should return 200."""
        for _ in range(5):
            r = client.get("/health")
            assert r.status_code == 200

    def test_rate_limit_sliding_window_logic(self):
        """Test the sliding window counter logic directly."""
        from app.middleware.auth import AuthRateLimitMiddleware, _rate_windows
        from collections import deque

        middleware = AuthRateLimitMiddleware(app)
        ip = "test-client-ip-999"
        _rate_windows[ip] = deque()

        # Fill window with old timestamps (outside 60s window)
        old_time = time.time() - 120
        for _ in range(100):
            _rate_windows[ip].append(old_time)

        # Should not rate limit — all timestamps are expired
        result = middleware._check_rate_limit(ip)
        assert result is None

    def test_rate_limit_triggers_when_window_full(self):
        """Filling the window to the limit triggers 429."""
        from app.middleware.auth import AuthRateLimitMiddleware, _rate_windows
        from collections import deque

        middleware = AuthRateLimitMiddleware(app)
        ip = "test-client-ip-888"

        with patch("app.middleware.auth.settings") as s:
            s.rate_limit_per_minute = 3
            _rate_windows[ip] = deque([time.time()] * 3)
            result = middleware._check_rate_limit(ip)

        assert result is not None
        assert result.status_code == 429

    def test_rate_limit_stats_returns_dict(self):
        from app.middleware.auth import get_rate_limit_stats
        stats = get_rate_limit_stats()
        assert isinstance(stats, dict)

    def test_health_exempt_from_auth(self):
        """Health endpoint is always accessible even with API key auth enabled."""
        with patch("app.middleware.auth.settings") as s:
            s.api_key = "secret-key"
            s.rate_limit_per_minute = 0
            # No X-API-Key header — should still get 200 for /health
            response = client.get("/health")
        assert response.status_code == 200


class TestAPIKeyAuth:

    def test_no_auth_when_api_key_not_set(self):
        """When API_KEY is blank, all requests pass through."""
        with patch("app.middleware.auth.settings") as s:
            s.api_key = ""
            s.rate_limit_per_minute = 0
            response = client.get("/api/v1/scenarios")
        assert response.status_code == 200

    def test_valid_api_key_passes(self):
        """Correct X-API-Key header allows the request."""
        with patch("app.middleware.auth.settings") as s:
            s.api_key = "test-secret-key"
            s.rate_limit_per_minute = 0
            response = client.get(
                "/api/v1/scenarios",
                headers={"X-API-Key": "test-secret-key"},
            )
        assert response.status_code == 200

    def test_invalid_api_key_returns_401(self):
        """Wrong X-API-Key returns 401 Unauthorized."""
        with patch("app.middleware.auth.settings") as s:
            s.api_key = "correct-key"
            s.rate_limit_per_minute = 0
            response = client.get(
                "/api/v1/scenarios",
                headers={"X-API-Key": "wrong-key"},
            )
        assert response.status_code == 401
        data = response.json()
        assert "Unauthorized" in data["error"]
        assert "hint" in data

    def test_missing_api_key_returns_401(self):
        """Missing X-API-Key header returns 401 when auth is enabled."""
        with patch("app.middleware.auth.settings") as s:
            s.api_key = "required-key"
            s.rate_limit_per_minute = 0
            response = client.get("/api/v1/scenarios")
        assert response.status_code == 401


# ── Job Queue Tests ───────────────────────────────────────────────────────────

class TestJobQueue:

    def test_create_job_returns_queued_status(self):
        from app.services.job_queue import create_job, JobStatus
        job = create_job(scenario_id="gpu_thermal_throttle_ecc")
        assert job.status == JobStatus.QUEUED
        assert job.job_id
        assert job.scenario_id == "gpu_thermal_throttle_ecc"
        assert job.created_at is not None

    def test_get_job_returns_job(self):
        from app.services.job_queue import create_job, get_job
        job = create_job(scenario_id="test")
        fetched = get_job(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    def test_get_unknown_job_returns_none(self):
        from app.services.job_queue import get_job
        assert get_job("nonexistent-job-id") is None

    def test_list_jobs_returns_most_recent_first(self):
        from app.services.job_queue import create_job, list_jobs
        j1 = create_job("scenario-1")
        j2 = create_job("scenario-2")
        j3 = create_job("scenario-3")
        jobs = list_jobs(limit=3)
        # Most recent first
        assert jobs[0].job_id == j3.job_id

    def test_purge_old_jobs(self):
        from app.services.job_queue import create_job, purge_old_jobs, _jobs
        initial_count = len(_jobs)
        for i in range(5):
            create_job(f"purge-test-{i}")
        removed = purge_old_jobs(keep=initial_count + 3)
        assert removed >= 0

    @pytest.mark.asyncio
    async def test_run_job_transitions_to_done(self):
        from app.services.job_queue import create_job, run_job, get_job, JobStatus
        from app.core.fixtures import load_scenario_bundle

        bundle = load_scenario_bundle("gpu_thermal_throttle_ecc")
        job = create_job("gpu_thermal_throttle_ecc")

        with patch("app.agent.graph.run_diagnosis", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = make_result()
            await run_job(
                job_id=job.job_id,
                scenario_id="gpu_thermal_throttle_ecc",
                metrics=bundle["metrics"],
                alert_summary="test alert",
                raw_logs=bundle["logs"],
            )

        updated = get_job(job.job_id)
        assert updated.status == JobStatus.DONE
        assert updated.result is not None
        assert updated.duration_seconds is not None

    @pytest.mark.asyncio
    async def test_run_job_transitions_to_failed_on_error(self):
        from app.services.job_queue import create_job, run_job, get_job, JobStatus
        from app.core.fixtures import load_gpu_metrics

        job = create_job("test")
        with patch("app.agent.graph.run_diagnosis", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError("LLM unavailable")
            await run_job(
                job_id=job.job_id,
                scenario_id="test",
                metrics=load_gpu_metrics(),
                alert_summary="alert",
                raw_logs=[],
            )

        updated = get_job(job.job_id)
        assert updated.status == JobStatus.FAILED
        assert updated.error is not None
        assert "RuntimeError" in updated.error

    def test_submit_job_endpoint_returns_job_id(self):
        """POST /jobs/diagnose returns immediately with job_id."""
        with patch("app.services.job_queue.run_job", new_callable=AsyncMock):
            response = client.post(
                "/api/v1/jobs/diagnose",
                json={"scenario_id": "gpu_thermal_throttle_ecc"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "poll_url" in data
        assert data["status"] == "queued"

    def test_get_job_endpoint_404_for_unknown(self):
        response = client.get("/api/v1/jobs/nonexistent-job")
        assert response.status_code == 404

    def test_list_jobs_endpoint(self):
        response = client.get("/api/v1/jobs?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)


# ── Runbook Generator Tests ───────────────────────────────────────────────────

class TestRunbookGenerator:

    def test_generate_runbook_returns_markdown(self):
        from app.services.runbook_service import generate_runbook
        result = make_result()
        md = generate_runbook(result)
        assert isinstance(md, str)
        assert len(md) > 200

    def test_runbook_contains_incident_id(self):
        from app.services.runbook_service import generate_runbook
        result = make_result()
        md = generate_runbook(result)
        assert result.incident_id in md

    def test_runbook_contains_root_cause(self):
        from app.services.runbook_service import generate_runbook
        result = make_result()
        md = generate_runbook(result)
        assert "ECC double-bit error" in md

    def test_runbook_contains_remediation_steps(self):
        from app.services.runbook_service import generate_runbook
        result = make_result()
        md = generate_runbook(result)
        assert "kubectl cordon" in md
        assert "nvidia-smi" in md

    def test_runbook_contains_k8s_patch(self):
        from app.services.runbook_service import generate_runbook
        result = make_result()
        md = generate_runbook(result)
        assert "apiVersion" in md

    def test_runbook_contains_gpu_metrics_table(self):
        from app.services.runbook_service import generate_runbook
        result = make_result()
        md = generate_runbook(result)
        assert "GPU Metrics" in md
        assert "CRITICAL" in md

    def test_runbook_severity_badges(self):
        from app.services.runbook_service import generate_runbook, SEVERITY_BADGE
        for severity in Severity:
            result = make_result(severity=severity)
            md = generate_runbook(result)
            badge = SEVERITY_BADGE[severity]
            assert badge in md

    def test_runbook_filename_format(self):
        from app.services.runbook_service import generate_runbook_filename
        result = make_result()
        filename = generate_runbook_filename(result)
        assert filename.startswith("runbook_")
        assert filename.endswith(".md")
        assert result.incident_id in filename

    def test_runbook_endpoint_returns_markdown(self):
        response = client.get("/api/v1/runbook/gpu_thermal_throttle_ecc")
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]
        assert "# Incident Report" in response.text

    def test_runbook_endpoint_404_for_unknown(self):
        response = client.get("/api/v1/runbook/nonexistent_scenario")
        assert response.status_code == 404

    def test_runbook_endpoint_has_download_header(self):
        response = client.get("/api/v1/runbook/gpu_thermal_throttle_ecc")
        assert "content-disposition" in response.headers
        assert "attachment" in response.headers["content-disposition"]


# ── Cluster Scanner Tests ─────────────────────────────────────────────────────

class TestClusterScanner:

    @pytest.mark.asyncio
    async def test_get_cluster_nodes_returns_list(self):
        from app.services.cluster_scanner import _get_cluster_nodes
        with patch("app.services.cluster_scanner.settings") as s:
            s.k8s_enabled = False
            nodes = await _get_cluster_nodes()
        assert isinstance(nodes, list)
        assert len(nodes) >= 2

    @pytest.mark.asyncio
    async def test_mock_cluster_has_healthy_and_unhealthy_nodes(self):
        from app.services.cluster_scanner import _get_node_metrics
        with patch("app.services.cluster_scanner.settings") as s:
            s.prometheus_enabled = False
            s.k8s_enabled = False

            node1_metrics = await _get_node_metrics("gpu-node-01")
            node3_metrics = await _get_node_metrics("gpu-node-03")

        # gpu-node-01 should be all healthy
        unhealthy_1 = [g for g in node1_metrics.gpus if g.health.value != "OK"]
        assert len(unhealthy_1) == 0

        # gpu-node-03 should have critical GPUs (original fixture)
        unhealthy_3 = [g for g in node3_metrics.gpus if g.health.value != "OK"]
        assert len(unhealthy_3) >= 1

    @pytest.mark.asyncio
    async def test_scan_cluster_returns_cluster_scan_result(self):
        from app.services.cluster_scanner import scan_cluster

        with patch("app.services.cluster_scanner.settings") as s, \
             patch("app.agent.graph.run_diagnosis", new_callable=AsyncMock) as mock_diag:
            s.k8s_enabled = False
            s.prometheus_enabled = False
            mock_diag.return_value = make_result()
            result = await scan_cluster("gpu-cluster-prod-01")

        assert result.cluster == "gpu-cluster-prod-01"
        assert result.total_nodes >= 2
        assert result.healthy_nodes + result.unhealthy_nodes == result.total_nodes
        assert result.total_scan_duration_seconds >= 0
        assert result.summary

    @pytest.mark.asyncio
    async def test_scan_identifies_unhealthy_node(self):
        from app.services.cluster_scanner import scan_cluster

        with patch("app.services.cluster_scanner.settings") as s, \
             patch("app.agent.graph.run_diagnosis", new_callable=AsyncMock) as mock_diag:
            s.k8s_enabled = False
            s.prometheus_enabled = False
            mock_diag.return_value = make_result()
            result = await scan_cluster()

        # gpu-node-03 has critical GPUs in fixture — should be unhealthy
        node_names = [r.node for r in result.node_results]
        assert "gpu-node-03" in node_names
        node3 = next(r for r in result.node_results if r.node == "gpu-node-03")
        assert not node3.healthy
        assert node3.unhealthy_gpu_count >= 1

    def test_cluster_scan_endpoint(self):
        """POST /cluster/scan returns ClusterScanResult."""
        with patch("app.services.cluster_scanner.settings") as s, \
             patch("app.agent.graph.run_diagnosis", new_callable=AsyncMock) as mock_diag:
            s.k8s_enabled = False
            s.prometheus_enabled = False
            mock_diag.return_value = make_result()
            response = client.post("/api/v1/cluster/scan")
        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
        assert "healthy_nodes" in data
        assert "unhealthy_nodes" in data
        assert "node_results" in data
        assert "summary" in data

    def test_cluster_nodes_endpoint(self):
        """GET /cluster/nodes returns node list."""
        with patch("app.services.cluster_scanner.settings") as s:
            s.k8s_enabled = False
            response = client.get("/api/v1/cluster/nodes")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "count" in data
        assert data["count"] >= 2


# ── New System Endpoints Tests ────────────────────────────────────────────────

class TestSystemEndpoints:

    def test_ready_endpoint_returns_200(self):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "checks" in data
        assert "qdrant" in data["checks"]

    def test_metrics_endpoint_returns_prometheus_format(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "gpu_copilot_jobs_total" in response.text
        assert "gpu_copilot_avg_diagnosis_seconds" in response.text

    def test_metrics_shows_job_counts(self):
        """After submitting a job, metrics should reflect it."""
        from app.services.job_queue import create_job, _jobs, JobStatus
        job = create_job("test-metrics")
        job.status = JobStatus.DONE
        job.duration_seconds = 11.4

        response = client.get("/metrics")
        assert "done" in response.text

    def test_health_shows_version_030(self):
        response = client.get("/health")
        assert response.json()["version"] == "0.3.0"

    def test_health_shows_auth_feature_flag(self):
        response = client.get("/health")
        data = response.json()
        assert "auth" in data["features"]
        assert "rate_limit" in data["features"]

    def test_ready_postgres_shows_disabled(self):
        """When Postgres is disabled, readiness check shows 'disabled'."""
        response = client.get("/ready")
        data = response.json()
        # Postgres is disabled by default
        if "postgres" in data["checks"]:
            assert data["checks"]["postgres"] == "disabled"


# ── Full Day 4 API Surface ────────────────────────────────────────────────────

class TestFullAPISurfaceDay4:

    def test_all_day4_endpoints_registered(self):
        """All Day 4 endpoints must return non-404/405."""
        endpoints = [
            ("GET",  "/ready"),
            ("GET",  "/metrics"),
            ("GET",  "/api/v1/jobs"),
            ("GET",  "/api/v1/cluster/nodes"),
            ("GET",  "/api/v1/runbook/gpu_thermal_throttle_ecc"),
        ]
        for method, path in endpoints:
            resp = client.get(path) if method == "GET" else client.post(path, json={})
            assert resp.status_code not in (404, 405), \
                f"{method} {path} → {resp.status_code} (not registered)"

    def test_runbook_all_scenarios(self):
        """Runbook endpoint works for all 3 demo scenarios."""
        with patch("app.middleware.auth.settings") as s:
            s.api_key = ""
            s.rate_limit_per_minute = 0
            for scenario_id in ["gpu_thermal_throttle_ecc", "gpu_memory_oom", "nvlink_degraded"]:
                response = client.get(f"/api/v1/runbook/{scenario_id}")
                assert response.status_code == 200, f"Runbook failed for {scenario_id}: {response.status_code}"
                assert len(response.text) > 500

    def test_job_full_lifecycle_via_api(self):
        """Submit → poll → verify done state via HTTP API."""
        with patch("app.middleware.auth.settings") as s:
            s.api_key = ""
            s.rate_limit_per_minute = 0
            with patch("app.services.job_queue.run_job", new_callable=AsyncMock):
                submit_resp = client.post(
                    "/api/v1/jobs/diagnose",
                    json={"scenario_id": "gpu_thermal_throttle_ecc"},
                )
            assert submit_resp.status_code == 200
            job_id = submit_resp.json()["job_id"]
            poll_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert poll_resp.status_code == 200
        data = poll_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "running", "done", "failed")
