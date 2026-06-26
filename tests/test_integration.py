"""
Integration Tests — Real LLM calls, real Qdrant RAG, full 5-node pipeline.

These tests require a real ANTHROPIC_API_KEY in your environment or .env file.
They make actual Claude API calls and validate the quality of the LLM output.

Run:
    ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_integration.py -v -s

Skip (CI / no key):
    pytest tests/test_integration.py -v --ignore=tests/test_integration.py
    # or just run: pytest tests/test_day1.py tests/test_day2.py

Markers:
    @pytest.mark.integration  — always requires API key
    @pytest.mark.slow         — runs all 3 scenarios end-to-end (~60s total)

Cost estimate: ~$0.03–0.08 per full run (3 scenarios × 3 LLM calls each).
"""

from __future__ import annotations

import os
import time

import pytest

from app.core.models import DiagnosisResult, FixCategory, Severity

# ── Skip guard ────────────────────────────────────────────────────────────────

def _has_api_key() -> bool:
    """Check .env file and environment for ANTHROPIC_API_KEY."""
    if os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-"):
        return True
    env_path = ".env"
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val.startswith("sk-ant-"):
                    return True
    return False


needs_api_key = pytest.mark.skipif(
    not _has_api_key(),
    reason="ANTHROPIC_API_KEY not set — set it in .env or environment to run integration tests",
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def primary_result() -> DiagnosisResult:
    """
    Run the full pipeline once for the primary scenario and cache the result.
    All tests in the primary group share this single LLM run to save API cost.
    """
    import asyncio
    from app.core.fixtures import load_scenario_bundle
    from app.agent.graph import run_diagnosis

    bundle = load_scenario_bundle("gpu_thermal_throttle_ecc")
    alert = bundle["alert"]
    alert_summary = (
        f"{alert.commonAnnotations.get('summary', '')} "
        f"Node: {alert.commonLabels.get('node', 'unknown')}. "
        f"Pod: {alert.alerts[0].labels.pod if alert.alerts else 'unknown'}. "
        f"Namespace: {alert.alerts[0].labels.namespace if alert.alerts else 'unknown'}."
    )

    result = asyncio.get_event_loop().run_until_complete(
        run_diagnosis(
            scenario_id="gpu_thermal_throttle_ecc",
            metrics=bundle["metrics"],
            alert_summary=alert_summary,
            raw_logs=bundle["logs"],
            k8s_patch_template=bundle["k8s_patch"],
        )
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — Response structure (shape is always correct regardless of LLM text)
# ══════════════════════════════════════════════════════════════════════════════

@needs_api_key
class TestResponseStructure:
    """
    Validate the shape and types of the DiagnosisResult.
    These pass as long as the LLM returns valid JSON — content-agnostic.
    """

    def test_result_is_diagnosis_result(self, primary_result):
        assert isinstance(primary_result, DiagnosisResult)

    def test_incident_id_format(self, primary_result):
        """incident_id must be INC-YYYYMMDD-XXXXXXXX."""
        parts = primary_result.incident_id.split("-")
        assert parts[0] == "INC"
        assert len(parts[1]) == 8 and parts[1].isdigit()
        assert len(parts[2]) == 8

    def test_severity_is_valid_enum(self, primary_result):
        assert primary_result.severity in list(Severity)

    def test_fix_category_is_valid_enum(self, primary_result):
        assert primary_result.fix_category in list(FixCategory)

    def test_confidence_in_range(self, primary_result):
        assert 0.0 <= primary_result.confidence <= 1.0

    def test_root_cause_is_non_empty_string(self, primary_result):
        assert isinstance(primary_result.root_cause, str)
        assert len(primary_result.root_cause) >= 50

    def test_contributing_factors_is_list(self, primary_result):
        assert isinstance(primary_result.contributing_factors, list)
        assert len(primary_result.contributing_factors) >= 1
        for f in primary_result.contributing_factors:
            assert isinstance(f, str) and len(f) > 5

    def test_remediation_steps_is_ordered_list(self, primary_result):
        steps = primary_result.remediation_steps
        assert len(steps) >= 2
        step_numbers = [s.step for s in steps]
        assert step_numbers == sorted(step_numbers)
        assert step_numbers[0] == 1

    def test_each_step_has_required_fields(self, primary_result):
        for step in primary_result.remediation_steps:
            assert isinstance(step.action, str) and len(step.action) > 2
            assert isinstance(step.description, str) and len(step.description) > 5

    def test_k8s_patch_yaml_present(self, primary_result):
        assert primary_result.k8s_patch_yaml is not None
        assert len(primary_result.k8s_patch_yaml) > 50

    def test_agent_trace_has_all_five_nodes(self, primary_result):
        trace = " ".join(primary_result.agent_trace)
        for node in ["fetch_context", "rag_retrieve", "analyze_signals", "root_cause", "recommend_fix"]:
            assert node in trace, f"Node '{node}' missing from agent trace: {primary_result.agent_trace}"

    def test_investigation_duration_under_120_seconds(self, primary_result):
        """Full pipeline must complete in under 2 minutes."""
        assert primary_result.investigation_duration_seconds < 120

    def test_similar_incidents_is_list(self, primary_result):
        assert isinstance(primary_result.similar_incidents, list)

    def test_log_snippets_populated(self, primary_result):
        assert len(primary_result.log_snippets) >= 1

    def test_node_matches_fixture(self, primary_result):
        assert primary_result.node == "gpu-node-03"

    def test_affected_gpu_detected(self, primary_result):
        """GPU 2 is the only CRITICAL one in the fixture."""
        assert primary_result.affected_gpu == 2


# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — LLM output quality (semantic correctness of the diagnosis)
# ══════════════════════════════════════════════════════════════════════════════

@needs_api_key
class TestDiagnosisQuality:
    """
    Validate that the LLM correctly diagnosed the primary scenario.
    GPU 2 overheating (91°C) → ECC double-bit error → vLLM OOMKill.
    Expected: CRITICAL severity, gpu_drain_and_reset fix.
    """

    def test_severity_is_critical(self, primary_result):
        assert primary_result.severity == Severity.CRITICAL, (
            f"Expected CRITICAL severity for ECC + thermal incident, got {primary_result.severity}. "
            f"Root cause: {primary_result.root_cause}"
        )

    def test_fix_category_is_gpu_drain_or_reset(self, primary_result):
        acceptable = {FixCategory.GPU_DRAIN_AND_RESET, FixCategory.NODE_DRAIN}
        assert primary_result.fix_category in acceptable, (
            f"Expected gpu_drain_and_reset or node_drain for ECC DBE incident, "
            f"got {primary_result.fix_category}"
        )

    def test_root_cause_mentions_ecc_or_thermal(self, primary_result):
        """Root cause must reference the primary failure signals."""
        rca = primary_result.root_cause.lower()
        ecc_mentioned = any(kw in rca for kw in ["ecc", "double-bit", "dbe", "uncorrectable"])
        thermal_mentioned = any(kw in rca for kw in ["thermal", "temperature", "throttle", "91", "overh"])
        assert ecc_mentioned or thermal_mentioned, (
            f"Root cause does not mention ECC errors or thermal throttle:\n{primary_result.root_cause}"
        )

    def test_root_cause_mentions_gpu(self, primary_result):
        rca = primary_result.root_cause.lower()
        assert "gpu" in rca or "cuda" in rca, (
            f"Root cause doesn't mention GPU/CUDA:\n{primary_result.root_cause}"
        )

    def test_contributing_factors_mention_relevant_signals(self, primary_result):
        """At least one factor must reference a real signal from the fixture."""
        all_factors = " ".join(primary_result.contributing_factors).lower()
        relevant = ["ecc", "temperature", "thermal", "fan", "nvlink", "memory", "cooling", "load"]
        matched = [kw for kw in relevant if kw in all_factors]
        assert len(matched) >= 2, (
            f"Contributing factors don't mention enough fixture signals.\n"
            f"Factors: {primary_result.contributing_factors}\n"
            f"Matched: {matched}"
        )

    def test_remediation_includes_cordon_or_drain(self, primary_result):
        """GPU drain/reset scenario must include node isolation step."""
        all_steps = " ".join(
            (s.command or "") + " " + s.action + " " + s.description
            for s in primary_result.remediation_steps
        ).lower()
        has_cordon = "cordon" in all_steps
        has_drain = "drain" in all_steps
        assert has_cordon or has_drain, (
            f"Remediation for critical GPU incident must include cordon or drain.\n"
            f"Steps: {[(s.action, s.command) for s in primary_result.remediation_steps]}"
        )

    def test_remediation_includes_gpu_reset(self, primary_result):
        """ECC double-bit error requires a GPU reset command."""
        all_text = " ".join(
            (s.command or "") + " " + s.description
            for s in primary_result.remediation_steps
        ).lower()
        assert "reset" in all_text or "nvidia-smi" in all_text, (
            f"Remediation for ECC DBE must include GPU reset.\n"
            f"Steps: {[(s.action, s.command) for s in primary_result.remediation_steps]}"
        )

    def test_k8s_patch_mentions_node(self, primary_result):
        """K8s patch must reference the affected node."""
        patch = primary_result.k8s_patch_yaml or ""
        assert "gpu-node-03" in patch or "gpu" in patch.lower(), (
            f"K8s patch doesn't reference the affected node.\n"
            f"Patch (first 200 chars): {patch[:200]}"
        )

    def test_confidence_above_threshold(self, primary_result):
        """LLM should be confident for a clear ECC + thermal incident."""
        assert primary_result.confidence >= 0.70, (
            f"Confidence {primary_result.confidence} too low for a clear-cut incident. "
            f"Root cause: {primary_result.root_cause}"
        )

    def test_rag_retrieved_similar_incidents(self, primary_result):
        """
        Qdrant should find at least one similar historical ECC/thermal incident.
        We seeded 2 matching incidents: INC-20231201-ECC001 and INC-20240105-THERMAL001.
        """
        assert len(primary_result.similar_incidents) >= 1, (
            "RAG retrieved no similar incidents for an ECC thermal event. "
            "Check that Qdrant was seeded correctly."
        )
        scores = [inc.get("similarity_score", 0) for inc in primary_result.similar_incidents]
        assert max(scores) >= 0.3, (
            f"Best similarity score {max(scores):.3f} too low — RAG may not be working correctly."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — All 3 scenarios end-to-end
# ══════════════════════════════════════════════════════════════════════════════

@needs_api_key
@pytest.mark.slow
class TestAllScenarios:
    """
    Run the full pipeline against all 3 demo scenarios and validate
    each one gets the correct severity + fix_category.
    Marked @slow — skipped in fast CI, run explicitly for full validation.
    """

    EXPECTED = {
        "gpu_thermal_throttle_ecc": {
            "severity": Severity.CRITICAL,
            "fix_categories": {FixCategory.GPU_DRAIN_AND_RESET, FixCategory.NODE_DRAIN},
            "root_cause_keywords": ["ecc", "thermal", "temperature", "91", "double-bit", "throttle"],
        },
        "gpu_memory_oom": {
            "severity": Severity.HIGH,
            "fix_categories": {FixCategory.CONFIG_PATCH, FixCategory.POD_RESTART},
            "root_cause_keywords": ["oom", "memory", "cuda", "batch", "out of memory"],
        },
        "nvlink_degraded": {
            "severity": Severity.WARNING,
            "fix_categories": {FixCategory.NVLINK_RESET, FixCategory.POD_RESTART, FixCategory.MANUAL_INTERVENTION},
            "root_cause_keywords": ["nvlink", "bandwidth", "replay", "tensor parallel"],
        },
    }

    def _run_scenario(self, scenario_id: str) -> DiagnosisResult:
        import asyncio
        from app.core.fixtures import load_scenario_bundle
        from app.agent.graph import run_diagnosis

        bundle = load_scenario_bundle(scenario_id)
        alert = bundle["alert"]
        alert_summary = (
            f"{alert.commonAnnotations.get('summary', '')} "
            f"Node: {alert.commonLabels.get('node', 'unknown')}."
        )
        return asyncio.get_event_loop().run_until_complete(
            run_diagnosis(
                scenario_id=scenario_id,
                metrics=bundle["metrics"],
                alert_summary=alert_summary,
                raw_logs=bundle["logs"],
                k8s_patch_template=bundle["k8s_patch"],
            )
        )

    def test_primary_scenario_severity_and_fix(self):
        result = self._run_scenario("gpu_thermal_throttle_ecc")
        expected = self.EXPECTED["gpu_thermal_throttle_ecc"]
        assert result.severity == expected["severity"], \
            f"Primary scenario severity: expected {expected['severity']}, got {result.severity}\nRCA: {result.root_cause}"
        assert result.fix_category in expected["fix_categories"], \
            f"Primary scenario fix: expected one of {expected['fix_categories']}, got {result.fix_category}"

    def test_oom_scenario_severity_and_fix(self):
        result = self._run_scenario("gpu_memory_oom")
        expected = self.EXPECTED["gpu_memory_oom"]
        # OOM can be high or critical depending on context — accept both
        assert result.severity in {Severity.HIGH, Severity.CRITICAL}, \
            f"OOM scenario severity: expected HIGH or CRITICAL, got {result.severity}\nRCA: {result.root_cause}"
        assert result.fix_category in expected["fix_categories"], \
            f"OOM scenario fix: expected one of {expected['fix_categories']}, got {result.fix_category}"

    def test_nvlink_scenario_severity_and_fix(self):
        result = self._run_scenario("nvlink_degraded")
        expected = self.EXPECTED["nvlink_degraded"]
        assert result.severity in {Severity.WARNING, Severity.HIGH}, \
            f"NVLink scenario severity: expected WARNING or HIGH, got {result.severity}\nRCA: {result.root_cause}"
        assert result.fix_category in expected["fix_categories"], \
            f"NVLink scenario fix: expected one of {expected['fix_categories']}, got {result.fix_category}"

    def test_all_scenarios_complete_in_reasonable_time(self):
        """Each scenario must diagnose in under 90 seconds."""
        for scenario_id in ["gpu_thermal_throttle_ecc", "gpu_memory_oom", "nvlink_degraded"]:
            start = time.time()
            result = self._run_scenario(scenario_id)
            elapsed = time.time() - start
            assert elapsed < 90, \
                f"Scenario '{scenario_id}' took {elapsed:.1f}s — exceeds 90s limit"
            assert result.investigation_duration_seconds < 90

    def test_all_scenarios_produce_kubectl_commands(self):
        """Every scenario must produce at least one kubectl command in remediation."""
        for scenario_id in ["gpu_thermal_throttle_ecc", "gpu_memory_oom", "nvlink_degraded"]:
            result = self._run_scenario(scenario_id)
            kubectl_steps = [s for s in result.remediation_steps if s.command and "kubectl" in s.command]
            assert len(kubectl_steps) >= 1, \
                f"Scenario '{scenario_id}' produced no kubectl commands.\n" \
                f"Steps: {[(s.action, s.command) for s in result.remediation_steps]}"

    def test_all_scenarios_root_cause_keywords(self):
        """Each scenario's root cause must contain scenario-relevant keywords."""
        for scenario_id, expected in self.EXPECTED.items():
            result = self._run_scenario(scenario_id)
            rca = result.root_cause.lower()
            matched = [kw for kw in expected["root_cause_keywords"] if kw in rca]
            assert len(matched) >= 1, \
                f"Scenario '{scenario_id}' root cause missing expected keywords.\n" \
                f"Expected one of: {expected['root_cause_keywords']}\n" \
                f"Root cause: {result.root_cause}"


# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — Pipeline timing and observability
# ══════════════════════════════════════════════════════════════════════════════

@needs_api_key
class TestPipelineObservability:
    """Validate timing, trace completeness, and agent state propagation."""

    def test_duration_under_60_seconds(self, primary_result):
        """Primary scenario should comfortably finish in under 60 seconds."""
        assert primary_result.investigation_duration_seconds < 60, (
            f"Pipeline took {primary_result.investigation_duration_seconds:.1f}s — "
            f"check for slow LLM responses or Qdrant issues."
        )

    def test_trace_has_five_entries(self, primary_result):
        assert len(primary_result.agent_trace) >= 5, (
            f"Expected at least 5 trace entries (one per node), got {len(primary_result.agent_trace)}.\n"
            f"Trace: {primary_result.agent_trace}"
        )

    def test_trace_shows_rag_result_count(self, primary_result):
        """rag_retrieve trace entry should mention how many incidents were found."""
        rag_entries = [t for t in primary_result.agent_trace if "rag_retrieve" in t]
        assert len(rag_entries) >= 1, "No rag_retrieve entry in agent trace"
        assert any(c.isdigit() for c in rag_entries[0]), \
            f"rag_retrieve trace entry doesn't report a count: {rag_entries[0]}"

    def test_trace_shows_anomaly_in_analyze_signals(self, primary_result):
        """analyze_signals trace entry should mention the primary anomaly."""
        analyze_entries = [t for t in primary_result.agent_trace if "analyze_signals" in t]
        assert len(analyze_entries) >= 1
        assert len(analyze_entries[0]) > 30, \
            f"analyze_signals trace entry too short — anomaly not captured: {analyze_entries[0]}"

    def test_trace_shows_severity_in_root_cause(self, primary_result):
        """root_cause trace entry should include severity level."""
        rc_entries = [t for t in primary_result.agent_trace if "root_cause:" in t]
        assert len(rc_entries) >= 1
        entry = rc_entries[0].upper()
        assert any(sev in entry for sev in ["CRITICAL", "HIGH", "WARNING", "INFO"]), \
            f"root_cause trace doesn't show severity: {rc_entries[0]}"

    def test_gpu_metrics_summary_in_result(self, primary_result):
        """gpu_metrics_summary must be populated from fetch_context."""
        summary = primary_result.gpu_metrics_summary
        assert "node" in summary
        assert "gpus" in summary
        assert summary["unhealthy_gpu_count"] >= 1

    def test_log_snippets_contain_error_keywords(self, primary_result):
        """Log snippets must have been filtered to ERROR/CRITICAL lines."""
        all_snippets = "\n".join(primary_result.log_snippets)
        has_error = any(kw in all_snippets for kw in ["ERROR", "CRITICAL", "WARNING", "ECC", "OOM"])
        assert has_error, (
            "Log snippets don't contain any ERROR/CRITICAL lines — "
            "fetch_context filtering may be broken."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Group 5 — RAG quality with real embeddings
# ══════════════════════════════════════════════════════════════════════════════

@needs_api_key
class TestRAGQuality:
    """Validate Qdrant retrieval correctness using real sentence-transformers."""

    def test_ecc_query_retrieves_ecc_incident(self):
        """ECC error query must retrieve at least one ECC-related incident."""
        import asyncio
        from app.services.qdrant_service import retrieve_similar_incidents

        results = asyncio.get_event_loop().run_until_complete(
            retrieve_similar_incidents(
                "GPU ECC double-bit uncorrectable error thermal throttle temperature",
                top_k=3,
            )
        )
        assert len(results) >= 1, "No results returned for ECC query"
        top_fix = results[0].get("fix_category", "")
        top_rca = results[0].get("root_cause", "").lower()
        assert "ecc" in top_rca or "gpu_drain" in top_fix, (
            f"Top RAG result for ECC query doesn't seem ECC-related.\n"
            f"Top result: fix={top_fix}, rca={top_rca[:100]}"
        )

    def test_oom_query_retrieves_oom_incident(self):
        """OOM query must retrieve an OOM or config-related incident."""
        import asyncio
        from app.services.qdrant_service import retrieve_similar_incidents

        results = asyncio.get_event_loop().run_until_complete(
            retrieve_similar_incidents("CUDA out of memory OOM pod crash batch size", top_k=3)
        )
        assert len(results) >= 1
        all_text = " ".join(r.get("root_cause", "") + r.get("fix_category", "") for r in results).lower()
        assert any(kw in all_text for kw in ["oom", "memory", "config", "batch"]), \
            f"OOM query didn't retrieve OOM-related incidents. Results: {[r.get('incident_id') for r in results]}"

    def test_nvlink_query_retrieves_nvlink_incident(self):
        """NVLink query must retrieve the NVLink incident."""
        import asyncio
        from app.services.qdrant_service import retrieve_similar_incidents

        results = asyncio.get_event_loop().run_until_complete(
            retrieve_similar_incidents("NVLink replay error bandwidth degraded tensor parallel", top_k=3)
        )
        assert len(results) >= 1
        incident_ids = [r.get("incident_id", "") for r in results]
        assert "INC-20231220-NVLINK001" in incident_ids, (
            f"NVLink query didn't surface INC-20231220-NVLINK001. Got: {incident_ids}"
        )

    def test_all_results_have_similarity_scores(self):
        """All RAG results must include a similarity_score."""
        import asyncio
        from app.services.qdrant_service import retrieve_similar_incidents

        results = asyncio.get_event_loop().run_until_complete(
            retrieve_similar_incidents("GPU failure incident", top_k=5)
        )
        for r in results:
            assert "similarity_score" in r
            assert 0.0 <= r["similarity_score"] <= 1.0

    def test_results_ordered_by_score_descending(self):
        """Results must be ranked highest-similarity first."""
        import asyncio
        from app.services.qdrant_service import retrieve_similar_incidents

        results = asyncio.get_event_loop().run_until_complete(
            retrieve_similar_incidents("GPU ECC error critical failure", top_k=4)
        )
        if len(results) >= 2:
            scores = [r["similarity_score"] for r in results]
            assert scores == sorted(scores, reverse=True), \
                f"RAG results not sorted by score: {scores}"


# ══════════════════════════════════════════════════════════════════════════════
# Group 6 — FastAPI endpoint integration
# ══════════════════════════════════════════════════════════════════════════════

@needs_api_key
class TestAPIEndpointIntegration:
    """
    Call the FastAPI /diagnose endpoint end-to-end (no mocking).
    Validates the full HTTP request → agent → response cycle.
    """

    def test_diagnose_endpoint_returns_200(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/v1/diagnose",
            json={"scenario_id": "gpu_thermal_throttle_ecc"},
        )
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}. Body: {response.text[:500]}"

    def test_diagnose_endpoint_response_schema(self):
        """Response must be parseable as a DiagnosisResult."""
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/v1/diagnose",
            json={"scenario_id": "gpu_thermal_throttle_ecc"},
        )
        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "incident_id", "severity", "fix_category", "root_cause",
            "contributing_factors", "remediation_steps", "k8s_patch_yaml",
            "confidence", "agent_trace", "investigation_duration_seconds",
            "similar_incidents",
        ]
        for field in required_fields:
            assert field in data, f"Missing field '{field}' in response"

    def test_diagnose_endpoint_severity_is_critical_for_primary(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/v1/diagnose",
            json={"scenario_id": "gpu_thermal_throttle_ecc"},
        )
        data = response.json()
        assert data["severity"] == "critical", \
            f"Expected critical severity from API, got {data['severity']}\nRCA: {data.get('root_cause')}"

    def test_diagnose_endpoint_remediation_steps_ordered(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/v1/diagnose",
            json={"scenario_id": "gpu_thermal_throttle_ecc"},
        )
        data = response.json()
        steps = data["remediation_steps"]
        assert len(steps) >= 2
        step_nums = [s["step"] for s in steps]
        assert step_nums == sorted(step_nums) and step_nums[0] == 1

    def test_rag_search_endpoint_with_real_embeddings(self):
        """POST /rag/search with no mocking — real sentence-transformers."""
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/v1/rag/search",
            json={"query": "GPU ECC double-bit uncorrectable error", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) >= 1
        for r in data["results"]:
            assert "similarity_score" in r
            assert "root_cause" in r
