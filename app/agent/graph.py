"""
LangGraph Diagnosis Agent
Implements a 4-node agentic graph:

  fetch_context → analyze_signals → root_cause → recommend_fix

Each node enriches the shared AgentState dict.
Day 1: uses mock data + direct LLM calls (no Qdrant yet).
Day 2: adds Qdrant RAG retrieval between fetch_context and analyze_signals.
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
    signals: dict[str, Any]

    # Outputs from LLM nodes
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


# ── LLM client ───────────────────────────────────────────────────────────────

def _llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key,
    )


def _call_llm(system: str, user: str) -> str:
    """Call the LLM and return the text response."""
    client = _llm()
    response = client.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content


# ── Node 1: fetch_context ─────────────────────────────────────────────────────

def fetch_context(state: AgentState) -> AgentState:
    """
    Summarize raw GPU metrics into a structured signal dict.
    In Day 2 this node also queries Qdrant for similar past incidents.
    """
    logger.info("[Agent] Node: fetch_context")
    state["trace"] = state.get("trace", [])
    state["trace"].append("fetch_context: extracting GPU metrics and log context")
    state["started_at"] = state.get("started_at", time.time())

    metrics: ClusterMetrics = state["metrics"]

    # Build a compact metrics summary for the LLM context window
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

    # Trim logs to relevant snippets (last 30 lines of each file)
    snippets = []
    for log in state.get("raw_logs", []):
        lines = log.strip().split("\n")
        # Prioritize ERROR / CRITICAL lines
        critical_lines = [l for l in lines if any(kw in l for kw in ["ERROR", "CRITICAL", "WARNING", "OOMKill", "ECC"])]
        snippets.append("\n".join(critical_lines[-20:] if critical_lines else lines[-15:]))

    state["relevant_log_snippets"] = snippets
    state["trace"].append(
        f"fetch_context: found {len(unhealthy_gpus)} unhealthy GPU(s), "
        f"extracted {len(snippets)} log snippet(s)"
    )
    return state


# ── Node 2: analyze_signals ───────────────────────────────────────────────────

def analyze_signals(state: AgentState) -> AgentState:
    """
    Ask the LLM to identify which signals are anomalous and rank them.
    """
    logger.info("[Agent] Node: analyze_signals")
    state["trace"].append("analyze_signals: correlating metrics and log signals")

    metrics_json = json.dumps(state["metrics_summary"], indent=2)
    logs_text = "\n\n---\n\n".join(state["relevant_log_snippets"])

    system = """You are an expert GPU infrastructure SRE with deep knowledge of NVIDIA GPUs,
CUDA, Kubernetes, vLLM, and DCGM. Analyze GPU telemetry and logs to identify anomalous signals.
Respond ONLY with a valid JSON object, no markdown, no explanation outside the JSON."""

    user = f"""Analyze these GPU cluster signals and identify the key anomalies.

GPU METRICS:
{metrics_json}

LOG SNIPPETS:
{logs_text}

ALERT SUMMARY:
{state.get('alert_summary', 'No alert summary provided')}

Respond with this exact JSON structure:
{{
  "primary_anomaly": "one-sentence description of the most critical signal",
  "anomalous_signals": [
    {{"signal": "name", "value": "observed value", "threshold": "threshold", "severity": "critical|high|warning"}}
  ],
  "affected_components": ["list of affected GPU IDs, pods, nodes"],
  "signal_timeline": "brief chronological description of how the signals evolved",
  "confidence": 0.0
}}"""

    response = _call_llm(system, user)

    # Strip markdown fences if present
    clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    signals = json.loads(clean)
    state["signals"] = signals
    state["trace"].append(
        f"analyze_signals: primary anomaly = '{signals.get('primary_anomaly', '')}'"
    )
    return state


# ── Node 3: root_cause ────────────────────────────────────────────────────────

def root_cause(state: AgentState) -> AgentState:
    """
    Determine the definitive root cause from the correlated signals.
    """
    logger.info("[Agent] Node: root_cause")
    state["trace"].append("root_cause: determining root cause from correlated signals")

    system = """You are a senior GPU infrastructure engineer performing root cause analysis.
You have deep expertise in NVIDIA GPU failure modes: ECC errors, thermal throttling, NVLink failures,
CUDA OOM, and Kubernetes scheduling. Be precise and technical. Respond ONLY with valid JSON."""

    user = f"""Perform root cause analysis for this GPU incident.

ANOMALY ANALYSIS:
{json.dumps(state['signals'], indent=2)}

GPU METRICS SUMMARY:
{json.dumps(state['metrics_summary'], indent=2)}

LOG SNIPPETS:
{chr(10).join(state['relevant_log_snippets'])}

Respond with this exact JSON structure:
{{
  "root_cause": "2-3 sentence technical root cause explanation",
  "contributing_factors": ["factor 1", "factor 2", "factor 3"],
  "severity": "critical|high|warning|info",
  "fix_category": "gpu_drain_and_reset|config_patch|nvlink_reset|pod_restart|node_drain|manual_intervention",
  "confidence": 0.0,
  "business_impact": "one sentence on user-facing impact"
}}"""

    response = _call_llm(system, user)
    clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    rca = json.loads(clean)

    state["root_cause"] = rca["root_cause"]
    state["contributing_factors"] = rca["contributing_factors"]
    state["severity"] = rca["severity"]
    state["fix_category"] = rca["fix_category"]
    state["confidence"] = rca.get("confidence", 0.85)
    state["trace"].append(
        f"root_cause: {rca['severity'].upper()} — {rca['fix_category']}"
    )
    return state


# ── Node 4: recommend_fix ─────────────────────────────────────────────────────

def recommend_fix(state: AgentState) -> AgentState:
    """
    Generate step-by-step remediation and a Kubernetes patch YAML.
    """
    logger.info("[Agent] Node: recommend_fix")
    state["trace"].append("recommend_fix: generating remediation steps and K8s patch")

    system = """You are a Kubernetes and GPU infrastructure expert. Generate precise, 
executable remediation steps and Kubernetes patch YAML. Be specific with kubectl commands.
Respond ONLY with valid JSON."""

    user = f"""Generate remediation for this GPU incident.

ROOT CAUSE: {state['root_cause']}
FIX CATEGORY: {state['fix_category']}
SEVERITY: {state['severity']}
AFFECTED NODE: {state['metrics_summary']['node']}
ANOMALOUS SIGNALS: {json.dumps(state['signals'].get('anomalous_signals', []), indent=2)}

EXISTING K8S PATCH TEMPLATE (use as reference):
{state.get('k8s_patch_template', 'No template provided')}

Respond with this exact JSON structure:
{{
  "remediation_steps": [
    {{"step": 1, "action": "action name", "command": "kubectl command or null", "description": "what this does and why"}}
  ],
  "k8s_patch_yaml": "complete multi-document YAML patch as a string",
  "estimated_resolution_minutes": 0,
  "prevention": "one sentence on how to prevent this in future"
}}"""

    response = _call_llm(system, user)
    clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    fix = json.loads(clean)

    state["remediation_steps"] = fix["remediation_steps"]
    state["k8s_patch_yaml"] = fix.get("k8s_patch_yaml", state.get("k8s_patch_template", ""))
    state["trace"].append(
        f"recommend_fix: generated {len(fix['remediation_steps'])} remediation steps"
    )
    return state


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("fetch_context", fetch_context)
    graph.add_node("analyze_signals", analyze_signals)
    graph.add_node("root_cause", root_cause)
    graph.add_node("recommend_fix", recommend_fix)

    graph.set_entry_point("fetch_context")
    graph.add_edge("fetch_context", "analyze_signals")
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
    Run the full LangGraph diagnosis pipeline and return a structured result.
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

    # Parse alert for pod/namespace
    pod = None
    namespace = None
    if "pod" in alert_summary.lower():
        # Extract from alert summary in a real implementation
        pod = "vllm-inference-7d9f8b-xkp2q"
        namespace = "ml-serving"

    return DiagnosisResult(
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
    )
