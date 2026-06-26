"""
Slack Notification Service
Posts rich incident diagnosis summaries to a Slack channel
via Incoming Webhooks (no bot token needed).

Message format uses Slack Block Kit for structured, scannable output:
  - Header block with severity emoji + incident ID
  - Section with root cause
  - Fields: node, GPU, fix category, confidence, duration
  - Remediation steps as numbered list
  - K8s patch snippet (truncated)
  - Actions: link to runbook / docs
"""

from __future__ import annotations

import json

import aiohttp

from app.core.config import settings
from app.core.logger import get_logger
from app.core.models import DiagnosisResult, Severity

logger = get_logger(__name__)

SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.WARNING: "🟡",
    Severity.INFO: "🟢",
}

FIX_CATEGORY_LABELS = {
    "gpu_drain_and_reset": "GPU Drain & Reset",
    "config_patch": "Config Patch",
    "nvlink_reset": "NVLink Reset",
    "pod_restart": "Pod Restart",
    "node_drain": "Node Drain",
    "manual_intervention": "Manual Intervention Required",
}


def _build_blocks(result: DiagnosisResult) -> list[dict]:
    """Build Slack Block Kit blocks for a DiagnosisResult."""
    emoji = SEVERITY_EMOJI.get(result.severity, "⚪")
    severity_label = result.severity.value.upper()
    fix_label = FIX_CATEGORY_LABELS.get(result.fix_category.value, result.fix_category.value)

    steps_text = "\n".join(
        f"{s.step}. *{s.action}*"
        + (f"\n   `{s.command}`" if s.command else "")
        for s in result.remediation_steps[:5]  # cap at 5 for Slack readability
    )

    k8s_snippet = ""
    if result.k8s_patch_yaml:
        lines = result.k8s_patch_yaml.strip().split("\n")
        k8s_snippet = "\n".join(lines[:15])
        if len(lines) > 15:
            k8s_snippet += f"\n# ... ({len(lines) - 15} more lines)"

    blocks = [
        # Header
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} GPU Incident — {severity_label} | {result.incident_id}",
                "emoji": True,
            },
        },
        # Divider
        {"type": "divider"},
        # Root cause
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Root Cause*\n{result.root_cause}",
            },
        },
        # Key fields
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Node*\n`{result.node}`"},
                {"type": "mrkdwn", "text": f"*Affected GPU*\n`GPU {result.affected_gpu}`" if result.affected_gpu is not None else "*Affected GPU*\nUnknown"},
                {"type": "mrkdwn", "text": f"*Pod*\n`{result.pod or 'N/A'}`"},
                {"type": "mrkdwn", "text": f"*Namespace*\n`{result.namespace or 'N/A'}`"},
                {"type": "mrkdwn", "text": f"*Fix Category*\n{fix_label}"},
                {"type": "mrkdwn", "text": f"*Confidence*\n{int(result.confidence * 100)}%"},
                {"type": "mrkdwn", "text": f"*Diagnosed In*\n{result.investigation_duration_seconds:.1f}s"},
                {"type": "mrkdwn", "text": f"*Severity*\n{severity_label}"},
            ],
        },
        {"type": "divider"},
        # Contributing factors
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Contributing Factors*\n"
                + "\n".join(f"• {f}" for f in result.contributing_factors),
            },
        },
        # Remediation steps
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Remediation Steps*\n{steps_text}",
            },
        },
    ]

    # K8s patch snippet (only if present)
    if k8s_snippet:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Kubernetes Patch (first 15 lines)*\n```{k8s_snippet}```",
            },
        })

    # Agent trace (collapsed-style)
    if result.agent_trace:
        trace_text = "\n".join(f"> {t}" for t in result.agent_trace)
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Agent Trace*\n{trace_text}",
            },
        })

    # Footer
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"🤖 GPU Copilot v0.2.0  |  Diagnosed at {result.diagnosed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  |  `{result.incident_id}`",
            }
        ],
    })

    return blocks


async def notify_slack(result: DiagnosisResult) -> bool:
    """
    Post a diagnosis result to Slack via Incoming Webhook.
    Returns True on success, False if Slack is disabled or call fails.
    """
    if not settings.slack_enabled or not settings.slack_webhook_url:
        logger.info("Slack notifications disabled (SLACK_ENABLED=false or no webhook URL)")
        return False

    payload = {
        "channel": settings.slack_channel,
        "username": "GPU Copilot",
        "icon_emoji": ":gpu:",
        "blocks": _build_blocks(result),
        # Fallback text for notifications
        "text": (
            f"{SEVERITY_EMOJI.get(result.severity, '⚪')} "
            f"[{result.severity.value.upper()}] GPU incident on {result.node}: {result.root_cause[:120]}..."
        ),
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                settings.slack_webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.text()
                if resp.status == 200 and body == "ok":
                    logger.info(f"Slack notification sent for {result.incident_id}")
                    return True
                else:
                    logger.warning(f"Slack returned {resp.status}: {body}")
                    return False

    except Exception as e:
        logger.error(f"Slack notification failed: {e}")
        return False


def build_slack_message_preview(result: DiagnosisResult) -> dict:
    """
    Return the Slack payload as a dict without sending it.
    Used in tests and the /diagnose response for debuggability.
    """
    return {
        "channel": settings.slack_channel,
        "blocks": _build_blocks(result),
        "text": f"[{result.severity.value.upper()}] {result.incident_id}: {result.root_cause[:120]}",
    }
