"""
LangGraph Diagnosis Agent — Day 2
5-node agentic graph:

  fetch_context → rag_retrieve → analyze_signals → root_cause → recommend_fix

Day 2 additions:
  - rag_retrieve: queries Qdrant for similar past incidents (injected into LLM context)
  - AgentState extended with similar_incidents field
  - analyze_signals and root_cause now receive RAG context
  - run_diagnosis triggers Slack notification + Postgres persistence post-pipeline
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.core.logger import get_logger
from app.core.models import (
    ClusterMetrics,
    DiagnosisResult,
    FixCategory,
    RemediationStep,
    Severity,
)

logger = get_logger(__name__)


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # Inputs
    scenario_id: str
    metrics: ClusterMetrics
    alert_summary: str
    raw_logs: list[str]
    k8s_patch_template: str

    # Enriched by nodes
    metrics_summary: dict[str, Any]
    relevant_log_snippets: list[str]
    similar_incidents: list[dict]        # Day 2: RAG results from Qdrant
    signals: dict[str, Any]

    # LLM outputs
    root_cause: str
    contributing_factors: list[str]
    severity: str
    fix_category: str
    remediation_steps: list[dict]
    k8s_patch_yaml: str
    confidence: float

    # Observability
    trace: list[str]
    started_at: float


# ── LLM ──────────────────────────────────────────────────────────────────────

def _llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key,
    )


def _call_llm(system: str, user: str) -> str:
    client = _llm()
    response = client.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content


def _strip_json(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


# ── Node 1: fetch_context ─────────────────────────────────────────────────────

def fetch_context(state: AgentState) -> AgentState:
    """
    Summarize raw GPU metrics into a compact signal dict.
    Extract ERROR/CRITICAL lines from log files.
    """
    logger.info("[Agent] Node: fetch_context")
    state["trace"] = state.get("trace", [])
    state["trace"].append("fetch_context: extracting GPU metrics and log context")
    state["started_at"] = state.get("started_at", time.time())

    metrics: ClusterMetrics = state["metrics"]
    unhealthy_gpus = [g for g in metrics.gpus if g.health != "OK"]

    gpu_summary = []
    for g in metrics.gpus:
        gpu_summary.append({
            "gpu_id": g.gpu_id,
            "temp_c": g.temperature_celsius,
            "throttle_threshold": g.temperature_threshold_slowdown,
            "util_pct": g.utilization_gpu_percent,
            "mem_used_pct": round(g.memory_used_mb / g.memory_total_mb * 100, 1),
            "ecc_dbe": g.ecc_errors_double_bit,
            "ecc_sbe": g.ecc_errors_single_bit,
            "nvlink_errors": g.nvlink_errors,
            "fan_pct": g.fan_speed_percent,
            "sm_clock_mhz": g.sm_clock_mhz,
            "health": g.health,
        })

    state["metrics_summary"] = {
        "cluster": metrics.cluster,
        "node": metrics.node,
        "timestamp": metrics.timestamp.isoformat(),
        "gpu_count": len(metrics.gpus),
        "unhealthy_gpu_count": len(unhealthy_gpus),
        "gpus": gpu_summary,
        "node_ram_used_pct": round(
            metrics.node_metrics.ram_used_gb / metrics.node_metrics.ram_total_gb * 100, 1
        ),
    }

    snippets = []
    for log in state.get("raw_logs", []):
        lines = log.strip().split("\n")
        critical = [l for l in lines if any(kw in l for kw in ["ERROR", "CRITICAL", "WARNING", "OOMKill", "ECC", "NVLink", "CUDA"])]
        snippets.append("\n".join(critical[-20:] if critical else lines[-15:]))

    state["relevant_log_snippets"] = snippets
    state["similar_incidents"] = []  # populated by rag_retrieve

    state["trace"].append(
        f"fetch_context: {len(unhealthy_gpus)} unhealthy GPU(s), {len(snippets)} log snippet(s)"
    )
    return state


# ── Node 2: rag_retrieve (Day 2) ──────────────────────────────────────────────

def rag_retrieve(state: AgentState) -> AgentState:
    """
    Query Qdrant for similar historical incidents.
    Builds a search query from the most salient signals in the current state
    and injects the top-k results as RAG context for subsequent LLM nodes.

    Falls back gracefully if Qdrant is unavailable.
    """
    logger.info("[Agent] Node: rag_retrieve")
    state["trace"].append("rag_retrieve: querying Qdrant for similar historical incidents")

    # Build a query string from the most salient signals we have so far
    metrics = state["metrics_summary"]
    unhealthy = [g for g in metrics["gpus"] if g["health"] in ("CRITICAL", "WARNING", "UNHEALTHY")]

    query_parts = []
    for g in unhealthy:
        if g["ecc_dbe"] > 0:
            query_parts.append(f"GPU ECC double-bit error temp={g['temp_c']}C")
        if g["nvlink_errors"] > 0:
            query_parts.append(f"NVLink replay errors count={g['nvlink_errors']}")
        if g["temp_c"] > g["throttle_threshold"]:
            query_parts.append(f"thermal throttle temperature {g['temp_c']}C threshold {g['throttle_threshold']}C")
        if g["mem_used_pct"] > 95:
            query_parts.append(f"GPU memory exhaustion {g['mem_used_pct']}% used")

    # Add log keywords
    all_logs = " ".join(state.get("relevant_log_snippets", []))
    if "OOMKill" in all_logs or "OOM" in all_logs:
        query_parts.append("OOMKilled CUDA out of memory pod crash")
    if "ECC" in all_logs:
        query_parts.append("ECC uncorrectable error CUDA driver")

    query = " ".join(query_parts) if query_parts else state.get("alert_summary", "GPU incident")

    try:
        import asyncio
        from app.services.qdrant_service import retrieve_similar_incidents
        similar = asyncio.get_event_loop().run_until_complete(
            retrieve_similar_incidents(query, top_k=settings.rag_top_k)
        )
        state["similar_incidents"] = similar
        state["trace"].append(
            f"rag_retrieve: found {len(similar)} similar incident(s) — "
            + ", ".join(s.get("incident_id", "?") for s in similar)
        )
    except Exception as e:
        logger.warning(f"Qdrant retrieval failed (non-fatal): {e}")
        state["similar_incidents"] = []
        state["trace"].append(f"rag_retrieve: skipped (Qdrant unavailable: {e})")

    return state


# ── Node 3: analyze_signals ───────────────────────────────────────────────────

def analyze_signals(state: AgentState) -> AgentState:
    """
    LLM identifies anomalous signals, now augmented with RAG context
    from similar historical incidents.
    """
    logger.info("[Agent] Node: analyze_signals")
    state["trace"].append("analyze_signals: correlating signals with RAG-augmented context")

    metrics_json = json.dumps(state["metrics_summary"], indent=2)
    logs_text = "\n\n---\n\n".join(state["relevant_log_snippets"])

    # Format RAG context for the prompt
    rag_context = ""
    if state.get("similar_incidents"):
        rag_lines = []
        for inc in state["similar_incidents"]:
            rag_lines.append(
                f"[{inc.get('incident_id')} | similarity={inc.get('similarity_score', 0):.2f}] "
                f"severity={inc.get('severity')} fix={inc.get('fix_category')}\n"
                f"  Root cause: {inc.get('root_cause', '')[:200]}\n"
                f"  Prevention: {inc.get('prevention', '')}"
            )
        rag_context = "\n\nSIMILAR PAST INCIDENTS (from Qdrant RAG):\n" + "\n\n".join(rag_lines)

    system = """You are an expert GPU infrastructure SRE with deep knowledge of NVIDIA GPUs,
CUDA, Kubernetes, vLLM, and DCGM. Analyze GPU telemetry and logs to identify anomalous signals.
Use any similar past incidents to inform your analysis. Respond ONLY with valid JSON."""

    user = f"""Analyze these GPU cluster signals and identify the key anomalies.

GPU METRICS:
{metrics_json}

LOG SNIPPETS:
{logs_text}

ALERT SUMMARY:
{state.get('alert_summary', 'No alert summary provided')}
{rag_context}

Respond with this exact JSON structure:
{{
  "primary_anomaly": "one-sentence description of the most critical signal",
  "anomalous_signals": [
    {{"signal": "name", "value": "observed value", "threshold": "threshold", "severity": "critical|high|warning"}}
  ],
  "affected_components": ["list of affected GPU IDs, pods, nodes"],
  "signal_timeline": "brief chronological description of how the signals evolved",
  "rag_informed": true,
  "confidence": 0.0
}}"""

    response = _call_llm(system, user)
    signals = json.loads(_strip_json(response))
    state["signals"] = signals
    state["trace"].append(
        f"analyze_signals: primary anomaly = '{signals.get('primary_anomaly', '')}'"
        + (" (RAG-augmented)" if state.get("similar_incidents") else "")
    )
    return state


# ── Node 4: root_cause ────────────────────────────────────────────────────────

def root_cause(state: AgentState) -> AgentState:
    """
    Determine root cause, informed by both signal analysis and RAG context.
    """
    logger.info("[Agent] Node: root_cause")
    state["trace"].append("root_cause: determining root cause with RAG context")

    rag_context = ""
    if state.get("similar_incidents"):
        rag_lines = []
        for inc in state["similar_incidents"]:
            rag_lines.append(
                f"• [{inc.get('incident_id')}] {inc.get('root_cause', '')[:250]} "
                f"→ fix: {inc.get('fix_category')} ({inc.get('resolution_minutes')} min)"
            )
        rag_context = "\n\nSIMILAR RESOLVED INCIDENTS:\n" + "\n".join(rag_lines)

    system = """You are a senior GPU infrastructure engineer performing root cause analysis.
You have deep expertise in NVIDIA GPU failure modes: ECC errors, thermal throttling, NVLink
failures, CUDA OOM, PCIe issues, and Kubernetes scheduling. Be precise and technical.
Respond ONLY with valid JSON."""

    user = f"""Perform root cause analysis for this GPU incident.

ANOMALY ANALYSIS:
{json.dumps(state['signals'], indent=2)}

GPU METRICS SUMMARY:
{json.dumps(state['metrics_summary'], indent=2)}

LOG SNIPPETS:
{chr(10).join(state['relevant_log_snippets'])}
{rag_context}

Respond with this exact JSON structure:
{{
  "root_cause": "2-3 sentence technical root cause explanation",
  "contributing_factors": ["factor 1", "factor 2", "factor 3"],
  "severity": "critical|high|warning|info",
  "fix_category": "gpu_drain_and_reset|config_patch|nvlink_reset|pod_restart|node_drain|manual_intervention",
  "confidence": 0.0,
  "business_impact": "one sentence on user-facing impact",
  "similar_incident_ids": ["INC-... if RAG informed this conclusion"]
}}"""

    response = _call_llm(system, user)
    rca = json.loads(_strip_json(response))

    state["root_cause"] = rca["root_cause"]
    state["contributing_factors"] = rca["contributing_factors"]
    state["severity"] = rca["severity"]
    state["fix_category"] = rca["fix_category"]
    state["confidence"] = rca.get("confidence", 0.85)
    state["trace"].append(
        f"root_cause: {rca['severity'].upper()} — {rca['fix_category']}"
        + (f" (informed by {rca.get('similar_incident_ids', [])})" if rca.get("similar_incident_ids") else "")
    )
    return state


# ── Node 5: recommend_fix ─────────────────────────────────────────────────────

def recommend_fix(state: AgentState) -> AgentState:
    """
    Generate step-by-step runbook and Kubernetes patch YAML.
    Past incident remediation steps are injected as reference examples.
    """
    logger.info("[Agent] Node: recommend_fix")
    state["trace"].append("recommend_fix: generating remediation steps and K8s patch")

    rag_examples = ""
    if state.get("similar_incidents"):
        examples = []
        for inc in state["similar_incidents"][:2]:
            steps = inc.get("remediation_steps", [])
            if steps:
                examples.append(
                    f"[{inc.get('incident_id')} — {inc.get('fix_category')}]:\n"
                    + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
                )
        if examples:
            rag_examples = "\n\nSIMILAR PAST REMEDIATION STEPS (use as reference):\n" + "\n\n".join(examples)

    system = """You are a Kubernetes and GPU infrastructure expert. Generate precise, executable
remediation steps and Kubernetes patch YAML. Use exact kubectl commands. Respond ONLY with valid JSON."""

    user = f"""Generate remediation for this GPU incident.

ROOT CAUSE: {state['root_cause']}
FIX CATEGORY: {state['fix_category']}
SEVERITY: {state['severity']}
AFFECTED NODE: {state['metrics_summary']['node']}
ANOMALOUS SIGNALS: {json.dumps(state['signals'].get('anomalous_signals', []), indent=2)}

K8S PATCH TEMPLATE:
{state.get('k8s_patch_template', 'No template provided')}
{rag_examples}

Respond with this exact JSON structure:
{{
  "remediation_steps": [
    {{"step": 1, "action": "action name", "command": "kubectl command or null", "description": "what this does and why"}}
  ],
  "k8s_patch_yaml": "complete multi-document YAML as a string",
  "estimated_resolution_minutes": 0,
  "prevention": "one sentence on how to prevent this in future"
}}"""

    response = _call_llm(system, user)
    fix = json.loads(_strip_json(response))

    state["remediation_steps"] = fix["remediation_steps"]
    state["k8s_patch_yaml"] = fix.get("k8s_patch_yaml", state.get("k8s_patch_template", ""))
    state["trace"].append(
        f"recommend_fix: {len(fix['remediation_steps'])} steps, "
        f"est. {fix.get('estimated_resolution_minutes', '?')} min resolution"
    )
    return state


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("fetch_context", fetch_context)
    graph.add_node("rag_retrieve", rag_retrieve)
    graph.add_node("analyze_signals", analyze_signals)
    graph.add_node("root_cause", root_cause)
    graph.add_node("recommend_fix", recommend_fix)

    graph.set_entry_point("fetch_context")
    graph.add_edge("fetch_context", "rag_retrieve")
    graph.add_edge("rag_retrieve", "analyze_signals")
    graph.add_edge("analyze_signals", "root_cause")
    graph.add_edge("root_cause", "recommend_fix")
    graph.add_edge("recommend_fix", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

async def run_diagnosis(
    scenario_id: str,
    metrics: ClusterMetrics,
    alert_summary: str,
    raw_logs: list[str],
    k8s_patch_template: str = "",
) -> DiagnosisResult:
    """
    Run the full 5-node LangGraph pipeline and return a DiagnosisResult.
    Post-pipeline: saves to Postgres, notifies Slack, upserts into Qdrant.
    """
    started = time.time()
    incident_id = f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    logger.info(f"Starting diagnosis {incident_id} for scenario '{scenario_id}'")

    graph = build_graph()

    initial_state: AgentState = {
        "scenario_id": scenario_id,
        "metrics": metrics,
        "alert_summary": alert_summary,
        "raw_logs": raw_logs,
        "k8s_patch_template": k8s_patch_template,
        "trace": [],
        "started_at": started,
    }

    final_state = graph.invoke(initial_state)
    duration = time.time() - started
    logger.info(f"Diagnosis {incident_id} complete in {duration:.2f}s")

    # Find the most affected GPU
    affected_gpu = None
    for g in metrics.gpus:
        if g.health in ("CRITICAL", "UNHEALTHY"):
            affected_gpu = g.gpu_id
            break

    # Extract pod/namespace from alert summary
    pod, namespace = _extract_pod_info(alert_summary)

    result = DiagnosisResult(
        incident_id=incident_id,
        scenario_id=scenario_id,
        node=metrics.node,
        affected_gpu=affected_gpu,
        pod=pod,
        namespace=namespace,
        diagnosed_at=datetime.now(timezone.utc),
        investigation_duration_seconds=round(duration, 2),
        severity=Severity(final_state.get("severity", "high")),
        root_cause=final_state.get("root_cause", ""),
        contributing_factors=final_state.get("contributing_factors", []),
        fix_category=FixCategory(final_state.get("fix_category", "manual_intervention")),
        remediation_steps=[
            RemediationStep(**step) for step in final_state.get("remediation_steps", [])
        ],
        k8s_patch_yaml=final_state.get("k8s_patch_yaml"),
        gpu_metrics_summary=final_state.get("metrics_summary", {}),
        log_snippets=final_state.get("relevant_log_snippets", []),
        alert_summary=alert_summary,
        agent_trace=final_state.get("trace", []),
        confidence=final_state.get("confidence", 0.85),
        similar_incidents=final_state.get("similar_incidents", []),
    )

    # ── Post-pipeline side effects ────────────────────────────────────────────
    await _post_pipeline(result)

    return result


async def _post_pipeline(result: DiagnosisResult) -> None:
    """Run post-pipeline side effects concurrently: Slack + Postgres + Qdrant upsert."""
    import asyncio

    tasks = []

    # 1. Slack notification
    from app.services.slack_service import notify_slack
    tasks.append(notify_slack(result))

    # 2. Postgres persistence
    from app.db.database import save_incident
    tasks.append(save_incident(result))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Update slack_notified flag
    if isinstance(results[0], bool) and results[0]:
        result.slack_notified = True

    # 3. Qdrant upsert (learn from this new incident)
    try:
        from app.services.qdrant_service import upsert_incident
        incident_text = (
            f"severity={result.severity.value} fix={result.fix_category.value} "
            f"node={result.node} root_cause={result.root_cause} "
            f"factors={' '.join(result.contributing_factors)}"
        )
        await upsert_incident(
            incident_id=result.incident_id,
            text=incident_text,
            payload={
                "incident_id": result.incident_id,
                "severity": result.severity.value,
                "fix_category": result.fix_category.value,
                "root_cause": result.root_cause,
                "contributing_factors": result.contributing_factors,
                "remediation_steps": [s.command for s in result.remediation_steps if s.command],
                "resolution_minutes": round(result.investigation_duration_seconds / 60, 1),
                "node": result.node,
                "pod": result.pod or "",
            },
        )
    except Exception as e:
        logger.warning(f"Qdrant upsert failed (non-fatal): {e}")


def _extract_pod_info(alert_summary: str) -> tuple[str | None, str | None]:
    """Extract pod and namespace from alert summary string."""
    pod, namespace = None, None
    parts = alert_summary.lower()
    if "pod:" in parts:
        try:
            pod = alert_summary.split("Pod:")[1].split(".")[0].strip()
        except Exception:
            pass
    if "namespace:" in parts:
        try:
            namespace = alert_summary.split("Namespace:")[1].split(".")[0].strip()
        except Exception:
            pass
    return pod, namespace
