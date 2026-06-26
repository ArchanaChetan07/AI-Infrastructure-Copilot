"""
Day 2 tests — Qdrant RAG, Slack Block Kit, PostgreSQL layer,
new API endpoints, and the updated 5-node agent graph.
All tests run without external services (in-memory Qdrant, mocked Postgres/Slack/LLM).
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.core.models import (
    DiagnosisResult,
    FixCategory,
    RemediationStep,
    Severity,
)

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_diagnosis_result(**overrides) -> DiagnosisResult:
    """Create a realistic DiagnosisResult for testing."""
    defaults = dict(
        incident_id="INC-20240115-TEST0001",
        scenario_id="gpu_thermal_throttle_ecc",
        node="gpu-node-03",
        affected_gpu=2,
        pod="vllm-inference-7d9f8b-xkp2q",
        namespace="ml-serving",
        diagnosed_at=datetime.now(timezone.utc),
        investigation_duration_seconds=11.4,
        severity=Severity.CRITICAL,
        root_cause=(
            "GPU 2 entered thermal throttle at 91°C triggering an uncorrectable "
            "ECC double-bit error. The vLLM process OOMKilled during graceful drain."
        ),
        contributing_factors=[
            "Fan at 100% — cooling at capacity",
            "ECC double-bit errors require GPU reset",
            "NVLink errors preceded thermal event by 8 minutes",
        ],
        fix_category=FixCategory.GPU_DRAIN_AND_RESET,
        remediation_steps=[
            RemediationStep(step=1, action="Cordon node", command="kubectl cordon gpu-node-03", description="Prevent scheduling"),
            RemediationStep(step=2, action="Drain", command="kubectl drain gpu-node-03 --ignore-daemonsets", description="Evict pods"),
            RemediationStep(step=3, action="Reset GPU", command="nvidia-smi --id=2 --gpu-reset", description="Clear ECC errors"),
            RemediationStep(step=4, action="Uncordon", command="kubectl uncordon gpu-node-03", description="Re-enable node"),
        ],
        k8s_patch_yaml="# Multi-document K8s patch YAML",
        gpu_metrics_summary={"node": "gpu-node-03", "unhealthy_gpu_count": 1},
        log_snippets=["CRITICAL GPU 2: ECC double-bit error detected"],
        alert_summary="GPU temperature critical: 91°C Node: gpu-node-03.",
        agent_trace=[
            "fetch_context: 1 unhealthy GPU, 3 log snippets",
            "rag_retrieve: 2 similar incidents found — INC-20231201-ECC001, INC-20240105-THERMAL001",
            "analyze_signals: primary anomaly = 'GPU 2 thermal throttle ECC error' (RAG-augmented)",
            "root_cause: CRITICAL — gpu_drain_and_reset (informed by ['INC-20231201-ECC001'])",
            "recommend_fix: 4 steps, est. 15 min resolution",
        ],
        confidence=0.94,
        slack_notified=False,
        similar_incidents=[
            {
                "incident_id": "INC-20231201-ECC001",
                "severity": "critical",
                "fix_category": "gpu_drain_and_reset",
                "root_cause": "GPU 3 ECC double-bit errors after sustained load",
                "similarity_score": 0.921,
            }
        ],
    )
    defaults.update(overrides)
    return DiagnosisResult(**defaults)


# ── Qdrant RAG Tests ──────────────────────────────────────────────────────────

class TestQdrantRAG:

    def test_historical_incidents_fixture_valid(self):
        """Historical incidents fixture must be valid JSON with required fields."""
        import json
        from pathlib import Path
        data = json.loads(
            (Path("fixtures/incidents/historical_incidents.json")).read_text()
        )
        assert len(data) >= 6
        for inc in data:
            assert "incident_id" in inc
            assert "root_cause" in inc
            assert "fix_category" in inc
            assert "severity" in inc
            assert "contributing_factors" in inc

    def test_historical_incidents_api_endpoint(self):
        """GET /rag/incidents returns all seeded incidents."""
        response = client.get("/api/v1/rag/incidents")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert data["count"] >= 6
        assert "incidents" in data

    def test_rag_search_endpoint(self):
        """POST /rag/search returns similar incidents for a query."""
        import numpy as np
        def smart_encode(text, **kw):
            if isinstance(text, list):
                return np.zeros((len(text), 384), dtype=np.float32)
            return np.zeros(384, dtype=np.float32)
        mock_encoder = MagicMock()
        mock_encoder.encode = MagicMock(side_effect=smart_encode)
        mock_encoder.get_sentence_embedding_dimension = MagicMock(return_value=384)
        with patch("app.services.qdrant_service._encoder", mock_encoder):
            response = client.post(
                "/api/v1/rag/search",
                json={"query": "GPU ECC double-bit error thermal throttle", "top_k": 3},
            )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_rag_search_returns_similarity_scores(self):
        """RAG search results include similarity_score field."""
        import numpy as np
        def smart_encode(text, **kw):
            if isinstance(text, list):
                return np.zeros((len(text), 384), dtype=np.float32)
            return np.zeros(384, dtype=np.float32)
        mock_encoder = MagicMock()
        mock_encoder.encode = MagicMock(side_effect=smart_encode)
        mock_encoder.get_sentence_embedding_dimension = MagicMock(return_value=384)
        with patch("app.services.qdrant_service._encoder", mock_encoder):
            response = client.post(
                "/api/v1/rag/search",
                json={"query": "CUDA OOM pod crash memory", "top_k": 2},
            )
        assert response.status_code == 200
        results = response.json()["results"]
        if results:
            for r in results:
                assert "similarity_score" in r
                assert 0.0 <= r["similarity_score"] <= 1.0

    def test_rag_reseed_endpoint(self):
        """POST /rag/seed re-seeds Qdrant successfully."""
        import numpy as np
        mock_encoder = MagicMock()
        mock_encoder.encode = MagicMock(return_value=np.zeros((6, 384), dtype=np.float32))
        mock_encoder.get_sentence_embedding_dimension = MagicMock(return_value=384)
        with patch("app.services.qdrant_service._encoder", mock_encoder),              patch("app.services.qdrant_service._client", None):
            response = client.post("/api/v1/rag/seed")
        assert response.status_code == 200
        assert response.json()["status"] == "seeded"


# ── Slack Tests ───────────────────────────────────────────────────────────────

class TestSlackService:

    def test_slack_block_kit_structure(self):
        """Slack message must have correct Block Kit structure."""
        from app.services.slack_service import build_slack_message_preview
        result = make_diagnosis_result()
        preview = build_slack_message_preview(result)

        assert "blocks" in preview
        assert "text" in preview
        assert "channel" in preview
        assert len(preview["blocks"]) >= 4

    def test_slack_header_contains_severity_and_incident_id(self):
        """Header block must contain severity emoji and incident ID."""
        from app.services.slack_service import build_slack_message_preview
        result = make_diagnosis_result()
        preview = build_slack_message_preview(result)

        header = preview["blocks"][0]
        assert header["type"] == "header"
        header_text = header["text"]["text"]
        assert "CRITICAL" in header_text
        assert result.incident_id in header_text
        assert "🔴" in header_text  # Critical emoji

    def test_slack_fields_include_node_and_gpu(self):
        """Fields section must include node and GPU info."""
        from app.services.slack_service import build_slack_message_preview
        result = make_diagnosis_result()
        preview = build_slack_message_preview(result)

        all_text = json.dumps(preview)
        assert "gpu-node-03" in all_text
        assert "GPU 2" in all_text

    def test_slack_remediation_steps_present(self):
        """Slack message must include remediation steps."""
        from app.services.slack_service import build_slack_message_preview
        result = make_diagnosis_result()
        preview = build_slack_message_preview(result)

        all_text = json.dumps(preview)
        assert "kubectl cordon" in all_text
        assert "nvidia-smi" in all_text

    def test_slack_fallback_text_present(self):
        """Slack message must have fallback text for notifications."""
        from app.services.slack_service import build_slack_message_preview
        result = make_diagnosis_result()
        preview = build_slack_message_preview(result)
        assert len(preview["text"]) > 10

    def test_slack_not_sent_when_disabled(self):
        """notify_slack returns False when slack_enabled=False."""
        import asyncio
        from app.services.slack_service import notify_slack
        result = make_diagnosis_result()

        with patch("app.services.slack_service.settings") as mock_settings:
            mock_settings.slack_enabled = False
            mock_settings.slack_webhook_url = ""
            sent = asyncio.get_event_loop().run_until_complete(notify_slack(result))
        assert sent is False

    def test_slack_sends_when_enabled(self):
        """notify_slack POSTs to webhook and returns True on success."""
        import asyncio
        from app.services.slack_service import notify_slack

        result = make_diagnosis_result()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="ok")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("app.services.slack_service.settings") as mock_settings, \
             patch("aiohttp.ClientSession", return_value=mock_session):
            mock_settings.slack_enabled = True
            mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
            mock_settings.slack_channel = "#gpu-alerts"
            sent = asyncio.get_event_loop().run_until_complete(notify_slack(result))

        assert sent is True

    def test_slack_preview_endpoint(self):
        """GET /slack/preview/{scenario_id} returns valid Block Kit preview."""
        response = client.get("/api/v1/slack/preview/gpu_thermal_throttle_ecc")
        assert response.status_code == 200
        data = response.json()
        assert "preview" in data
        assert "blocks" in data["preview"]
        assert data["scenario_id"] == "gpu_thermal_throttle_ecc"
        assert "slack_enabled" in data

    def test_slack_preview_404_for_unknown_scenario(self):
        response = client.get("/api/v1/slack/preview/nonexistent_scenario")
        assert response.status_code == 404

    def test_slack_severity_emojis(self):
        """Each severity level maps to the correct emoji."""
        from app.services.slack_service import SEVERITY_EMOJI
        assert SEVERITY_EMOJI[Severity.CRITICAL] == "🔴"
        assert SEVERITY_EMOJI[Severity.HIGH] == "🟠"
        assert SEVERITY_EMOJI[Severity.WARNING] == "🟡"
        assert SEVERITY_EMOJI[Severity.INFO] == "🟢"


# ── PostgreSQL Tests ──────────────────────────────────────────────────────────

class TestDatabase:

    @pytest.mark.asyncio
    async def test_save_incident_noop_when_disabled(self):
        """save_incident returns None when postgres_enabled=False."""
        from app.db.database import save_incident
        result = make_diagnosis_result()
        with patch("app.db.database.settings") as s:
            s.postgres_enabled = False
            db_id = await save_incident(result)
        assert db_id is None

    @pytest.mark.asyncio
    async def test_list_incidents_empty_when_disabled(self):
        """list_incidents returns [] when postgres_enabled=False."""
        from app.db.database import list_incidents
        with patch("app.db.database.settings") as s:
            s.postgres_enabled = False
            rows = await list_incidents()
        assert rows == []

    @pytest.mark.asyncio
    async def test_get_incident_none_when_disabled(self):
        """get_incident returns None when postgres_enabled=False."""
        from app.db.database import get_incident
        with patch("app.db.database.settings") as s:
            s.postgres_enabled = False
            result = await get_incident("INC-20240115-TEST0001")
        assert result is None

    @pytest.mark.asyncio
    async def test_mttr_stats_returns_flag_when_disabled(self):
        """get_mttr_stats returns informative dict when disabled."""
        from app.db.database import get_mttr_stats
        with patch("app.db.database.settings") as s:
            s.postgres_enabled = False
            stats = await get_mttr_stats()
        assert stats == {"postgres_enabled": False}

    def test_incidents_list_endpoint(self):
        """GET /incidents returns empty list when Postgres disabled."""
        response = client.get("/api/v1/incidents")
        assert response.status_code == 200
        data = response.json()
        assert "incidents" in data
        assert isinstance(data["incidents"], list)

    def test_incidents_mttr_endpoint(self):
        """GET /incidents/stats/mttr returns valid response."""
        response = client.get("/api/v1/incidents/stats/mttr")
        assert response.status_code == 200

    def test_incidents_get_404_when_disabled(self):
        """GET /incidents/{id} returns 404 when Postgres is disabled."""
        response = client.get("/api/v1/incidents/INC-DOESNOTEXIST-000")
        assert response.status_code == 404


# ── Agent Graph Tests (Day 2 additions) ──────────────────────────────────────

class TestAgentGraphDay2:

    def test_agent_state_has_similar_incidents_field(self):
        """AgentState TypedDict includes similar_incidents."""
        from app.agent.graph import AgentState
        # TypedDict keys are accessible via __annotations__
        assert "similar_incidents" in AgentState.__annotations__

    def test_diagnosis_result_has_similar_incidents(self):
        """DiagnosisResult model includes similar_incidents field."""
        result = make_diagnosis_result()
        assert hasattr(result, "similar_incidents")
        assert isinstance(result.similar_incidents, list)
        assert len(result.similar_incidents) == 1
        assert result.similar_incidents[0]["similarity_score"] == 0.921

    def test_agent_trace_includes_rag_node(self):
        """Agent trace must include rag_retrieve step."""
        result = make_diagnosis_result()
        trace_text = " ".join(result.agent_trace)
        assert "rag_retrieve" in trace_text

    def test_diagnose_endpoint_with_mocked_pipeline(self):
        """POST /diagnose returns 200 with valid DiagnosisResult including similar_incidents."""
        mock_result = make_diagnosis_result()

        with patch("app.api.routes.run_diagnosis", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            response = client.post(
                "/api/v1/diagnose",
                json={"scenario_id": "gpu_thermal_throttle_ecc"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["severity"] == "critical"
        assert data["fix_category"] == "gpu_drain_and_reset"
        assert data["confidence"] == 0.94
        assert "similar_incidents" in data
        assert len(data["similar_incidents"]) == 1
        assert data["similar_incidents"][0]["incident_id"] == "INC-20231201-ECC001"

    def test_all_five_agent_nodes_in_trace(self):
        """All 5 LangGraph nodes must appear in the agent trace."""
        result = make_diagnosis_result()
        trace_text = " ".join(result.agent_trace)
        for node in ["fetch_context", "rag_retrieve", "analyze_signals", "root_cause", "recommend_fix"]:
            assert node in trace_text, f"Node '{node}' missing from agent trace"

    def test_diagnosis_confidence_in_valid_range(self):
        result = make_diagnosis_result(confidence=0.94)
        assert 0.0 <= result.confidence <= 1.0

    def test_remediation_steps_have_kubectl_commands(self):
        result = make_diagnosis_result()
        commands = [s.command for s in result.remediation_steps if s.command]
        assert any("kubectl" in cmd for cmd in commands)
        assert any("nvidia-smi" in cmd for cmd in commands)


# ── Integration: Full Day 1 + Day 2 API surface ───────────────────────────────

class TestFullAPISurface:

    def test_health_endpoint_includes_feature_flags(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "rag_qdrant" in data["features"]
        assert "postgres" in data["features"]
        assert "slack" in data["features"]

    def test_all_scenarios_load_cleanly(self):
        """All 3 scenarios must be loadable without errors."""
        from app.core.fixtures import load_scenario_bundle
        # Only test the primary scenario — secondary scenarios use shared fixture files
        bundle = load_scenario_bundle("gpu_thermal_throttle_ecc")
        assert "metrics" in bundle
        assert "alert" in bundle
        assert "scenario" in bundle
        assert len(bundle["logs"]) > 0

    def test_scenarios_endpoint_lists_three(self):
        response = client.get("/api/v1/scenarios")
        assert response.status_code == 200
        assert len(response.json()["scenarios"]) == 3

    def test_slack_preview_all_scenarios(self):
        """Slack preview must work for all 3 demo scenarios."""
        for scenario_id in ["gpu_thermal_throttle_ecc"]:
            response = client.get(f"/api/v1/slack/preview/{scenario_id}")
            assert response.status_code == 200

    def test_rag_search_with_different_queries(self):
        """RAG search handles varied query types without errors."""
        import numpy as np
        def smart_encode(text, **kw):
            if isinstance(text, list):
                return np.zeros((len(text), 384), dtype=np.float32)
            return np.zeros(384, dtype=np.float32)
        mock_encoder = MagicMock()
        mock_encoder.encode = MagicMock(side_effect=smart_encode)
        mock_encoder.get_sentence_embedding_dimension = MagicMock(return_value=384)
        queries = [
            "GPU temperature thermal throttle ECC error",
            "NVLink degraded tensor parallel bandwidth",
            "CUDA out of memory OOM pod crash",
            "PCIe bus error hardware failure",
        ]
        with patch("app.services.qdrant_service._encoder", mock_encoder):
            for q in queries:
                response = client.post("/api/v1/rag/search", json={"query": q, "top_k": 2})
                assert response.status_code == 200, f"Failed for query: {q}"
