"""
Runbook Generator — Day 4
Exports a full incident diagnosis as a professional Markdown runbook.

The generated document is suitable for:
  - Pasting into Confluence / Notion / GitHub Wiki
  - Attaching to a Jira/Linear ticket
  - Archiving in a runbook library for future RAG retrieval
  - Training data for future LLM fine-tuning

Structure:
  # Incident <ID>
  ## Executive Summary
  ## Timeline
  ## Root Cause Analysis
  ## Contributing Factors
  ## Affected Components
  ## Remediation Steps
  ## Kubernetes Patch
  ## Similar Past Incidents (RAG)
  ## Prevention
  ## Agent Trace (Observability)
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.models import DiagnosisResult, Severity


SEVERITY_BADGE = {
    Severity.CRITICAL: "🔴 CRITICAL",
    Severity.HIGH:     "🟠 HIGH",
    Severity.WARNING:  "🟡 WARNING",
    Severity.INFO:     "🟢 INFO",
}


def generate_runbook(result: DiagnosisResult) -> str:
    """Generate a complete Markdown runbook from a DiagnosisResult."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    severity_label = SEVERITY_BADGE.get(result.severity, result.severity.value.upper())

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# Incident Report — {result.incident_id}",
        "",
        f"> **Severity:** {severity_label}  ",
        f"> **Generated:** {now}  ",
        f"> **Diagnosed in:** {result.investigation_duration_seconds:.1f}s  ",
        f"> **Confidence:** {int(result.confidence * 100)}%  ",
        "",
        "---",
        "",
    ]

    # ── Executive Summary ─────────────────────────────────────────────────────
    lines += [
        "## Executive Summary",
        "",
        result.root_cause,
        "",
    ]

    # ── Affected Components ───────────────────────────────────────────────────
    lines += [
        "## Affected Components",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| **Cluster** | `{result.gpu_metrics_summary.get('cluster', 'unknown')}` |",
        f"| **Node** | `{result.node}` |",
        f"| **GPU** | `GPU {result.affected_gpu}` |" if result.affected_gpu is not None else "| **GPU** | Unknown |",
        f"| **Pod** | `{result.pod}` |" if result.pod else "| **Pod** | N/A |",
        f"| **Namespace** | `{result.namespace}` |" if result.namespace else "| **Namespace** | N/A |",
        f"| **Fix Category** | `{result.fix_category.value}` |",
        "",
    ]

    # ── GPU Metrics at Incident Time ──────────────────────────────────────────
    gpus = result.gpu_metrics_summary.get("gpus", [])
    if gpus:
        lines += [
            "## GPU Metrics at Incident Time",
            "",
            "| GPU | Temp (°C) | Util % | Mem % | ECC DBE | NVLink Err | Health |",
            "|---|---|---|---|---|---|---|",
        ]
        for g in gpus:
            health_icon = {"OK": "✅", "WARNING": "⚠️", "CRITICAL": "🔴", "UNHEALTHY": "💀"}.get(g.get("health", ""), "❓")
            lines.append(
                f"| GPU {g['gpu_id']} | {g['temp_c']} | {g['util_pct']}% | "
                f"{g['mem_used_pct']}% | {g['ecc_dbe']} | {g['nvlink_errors']} | "
                f"{health_icon} {g['health']} |"
            )
        lines.append("")

    # ── Root Cause Analysis ───────────────────────────────────────────────────
    lines += [
        "## Root Cause Analysis",
        "",
        result.root_cause,
        "",
    ]

    # ── Contributing Factors ──────────────────────────────────────────────────
    if result.contributing_factors:
        lines += ["## Contributing Factors", ""]
        for i, factor in enumerate(result.contributing_factors, 1):
            lines.append(f"{i}. {factor}")
        lines.append("")

    # ── Remediation Steps ─────────────────────────────────────────────────────
    lines += ["## Remediation Steps", ""]
    for step in result.remediation_steps:
        lines.append(f"### Step {step.step}: {step.action}")
        lines.append("")
        lines.append(step.description)
        if step.command:
            lines += ["", f"```bash", f"{step.command}", "```"]
        lines.append("")

    # ── Kubernetes Patch ──────────────────────────────────────────────────────
    if result.k8s_patch_yaml:
        lines += [
            "## Kubernetes Patch",
            "",
            "Apply with:",
            "```bash",
            "kubectl apply -f patch.yaml --dry-run=server  # verify first",
            "kubectl apply -f patch.yaml                   # apply for real",
            "```",
            "",
            "```yaml",
            result.k8s_patch_yaml.strip(),
            "```",
            "",
        ]

    # ── Log Evidence ──────────────────────────────────────────────────────────
    if result.log_snippets:
        lines += ["## Log Evidence", ""]
        for i, snippet in enumerate(result.log_snippets[:3], 1):
            lines += [f"**Log {i}:**", "```", snippet[:800].strip(), "```", ""]

    # ── Similar Past Incidents (RAG) ──────────────────────────────────────────
    if result.similar_incidents:
        lines += [
            "## Similar Past Incidents",
            "",
            "Retrieved by the RAG pipeline from the incident vector store:",
            "",
        ]
        for inc in result.similar_incidents:
            score = inc.get("similarity_score", 0)
            lines += [
                f"### {inc.get('incident_id', 'Unknown')} (similarity: {score:.0%})",
                "",
                f"**Fix:** `{inc.get('fix_category', '')}`  ",
                f"**Root cause:** {inc.get('root_cause', '')[:200]}",
                "",
            ]

    # ── Agent Trace (Observability) ───────────────────────────────────────────
    lines += [
        "## Agent Trace",
        "",
        "LangGraph node execution trace:",
        "",
        "```",
    ]
    for entry in result.agent_trace:
        lines.append(f"  {entry}")
    lines += ["```", ""]

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        f"*Generated by AI Infrastructure Copilot v0.3.0 — {now}*  ",
        f"*Incident ID: `{result.incident_id}`*",
    ]

    return "\n".join(lines)


def generate_runbook_filename(result: DiagnosisResult) -> str:
    """Return a safe filename for the runbook."""
    date = result.diagnosed_at.strftime("%Y%m%d")
    node = result.node.replace("-", "_")
    return f"runbook_{result.incident_id}_{date}_{node}.md"
