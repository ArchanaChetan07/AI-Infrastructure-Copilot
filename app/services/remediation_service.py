"""
Auto-Remediation Service — Day 3
Executes the AI-generated K8s patch with configurable safety gates.

Safety modes:
  DRY_RUN   — kubectl apply --dry-run=server, never touches the cluster
  CONFIRM   — applies patch only if explicitly confirmed via API call
  AUTO      — applies immediately for CRITICAL incidents (use with care)

All executions are logged to Postgres (if enabled) and posted to Slack.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import get_logger
from app.core.models import DiagnosisResult, Severity

logger = get_logger(__name__)


class RemediationMode(str, Enum):
    DRY_RUN = "dry_run"    # always safe — just shows what would happen
    CONFIRM = "confirm"    # queues patch, waits for POST /remediate/{id}/confirm
    AUTO    = "auto"       # applies immediately (critical incidents only)


class RemediationRequest(BaseModel):
    incident_id: str
    mode: RemediationMode = RemediationMode.DRY_RUN
    patch_yaml: str | None = None  # override if you want a custom patch


class RemediationResult(BaseModel):
    incident_id: str
    mode: RemediationMode
    executed_at: datetime
    success: bool
    dry_run: bool
    applied_resources: list[str] = []
    steps_executed: list[dict[str, Any]] = []
    output: str = ""
    error: str | None = None
    slack_notified: bool = False


# In-memory queue for CONFIRM mode (Day 3: replace with Redis/Postgres in prod)
_pending_confirmations: dict[str, dict] = {}


async def execute_remediation(
    result: DiagnosisResult,
    mode: RemediationMode = RemediationMode.DRY_RUN,
    patch_yaml_override: str | None = None,
) -> RemediationResult:
    """
    Execute the remediation for a diagnosed incident.

    Safety gate logic:
      - DRY_RUN: always runs dry, returns what would happen
      - CONFIRM: stores patch in memory, returns pending status
      - AUTO: only executes for CRITICAL/HIGH, with cordon → drain → reset sequence
    """
    patch_yaml = patch_yaml_override or result.k8s_patch_yaml or ""
    started = datetime.now(timezone.utc)

    logger.info(
        f"Remediation requested: incident={result.incident_id} "
        f"mode={mode} severity={result.severity}"
    )

    if mode == RemediationMode.CONFIRM:
        # Queue for manual confirmation
        _pending_confirmations[result.incident_id] = {
            "result": result,
            "patch_yaml": patch_yaml,
            "queued_at": started.isoformat(),
        }
        logger.info(f"Remediation queued for confirmation: {result.incident_id}")
        return RemediationResult(
            incident_id=result.incident_id,
            mode=mode,
            executed_at=started,
            success=True,
            dry_run=True,
            output=(
                f"Remediation queued. POST /api/v1/remediate/{result.incident_id}/confirm to apply.\n"
                f"Patch preview:\n{patch_yaml[:300]}..."
            ),
        )

    is_dry = mode == RemediationMode.DRY_RUN
    if mode == RemediationMode.AUTO and result.severity not in (Severity.CRITICAL, Severity.HIGH):
        logger.warning(
            f"AUTO mode requested for {result.severity} incident — downgrading to DRY_RUN"
        )
        is_dry = True

    return await _execute_steps(result, patch_yaml, is_dry, mode, started)


async def confirm_remediation(incident_id: str) -> RemediationResult:
    """
    Confirm and execute a queued remediation (CONFIRM mode).
    Called by POST /api/v1/remediate/{incident_id}/confirm
    """
    if incident_id not in _pending_confirmations:
        return RemediationResult(
            incident_id=incident_id,
            mode=RemediationMode.CONFIRM,
            executed_at=datetime.now(timezone.utc),
            success=False,
            dry_run=False,
            error=f"No pending remediation found for {incident_id}. Already confirmed or expired.",
        )

    pending = _pending_confirmations.pop(incident_id)
    result: DiagnosisResult = pending["result"]
    patch_yaml: str = pending["patch_yaml"]

    logger.info(f"Executing confirmed remediation for {incident_id}")
    return await _execute_steps(
        result, patch_yaml, dry_run=False,
        mode=RemediationMode.CONFIRM,
        started=datetime.now(timezone.utc),
    )


async def _execute_steps(
    result: DiagnosisResult,
    patch_yaml: str,
    dry_run: bool,
    mode: RemediationMode,
    started: datetime,
) -> RemediationResult:
    """Execute the step-by-step remediation sequence."""
    from app.integrations.kubernetes import apply_k8s_patch, cordon_node, drain_node

    steps_executed = []
    all_applied = []
    errors = []

    node = result.node
    fix = result.fix_category.value

    # ── Step 1: Cordon node (always first for node-level fixes) ───────────────
    if fix in ("gpu_drain_and_reset", "node_drain"):
        cordon_result = await cordon_node(node, dry_run=dry_run)
        steps_executed.append({
            "step": 1,
            "action": "cordon_node",
            "command": f"kubectl cordon {node}",
            "success": cordon_result["success"],
            "output": cordon_result.get("output", ""),
            "dry_run": dry_run,
        })
        if not cordon_result["success"] and not dry_run:
            errors.append(f"Cordon failed: {cordon_result.get('error', '')}")

    # ── Step 2: Apply the main K8s patch YAML ─────────────────────────────────
    if patch_yaml:
        patch_result = await apply_k8s_patch(patch_yaml, dry_run=dry_run, incident_id=result.incident_id)
        steps_executed.append({
            "step": 2,
            "action": "apply_k8s_patch",
            "command": f"kubectl apply -f patch.yaml{'  --dry-run=server' if dry_run else ''}",
            "success": patch_result["success"],
            "output": patch_result.get("output", "")[:500],
            "dry_run": dry_run,
        })
        all_applied.extend(patch_result.get("applied_resources", []))
        if not patch_result["success"] and not dry_run:
            errors.append(f"Patch failed: {patch_result.get('error', '')}")

    # ── Step 3: Drain node (only for gpu_drain_and_reset) ────────────────────
    if fix == "gpu_drain_and_reset" and not errors:
        drain_result = await drain_node(node, dry_run=dry_run)
        steps_executed.append({
            "step": 3,
            "action": "drain_node",
            "command": f"kubectl drain {node} --ignore-daemonsets --delete-emptydir-data",
            "success": drain_result["success"],
            "output": drain_result.get("output", ""),
            "dry_run": dry_run,
        })

    success = len(errors) == 0
    output_lines = [f"[{'DRY RUN' if dry_run else 'LIVE'}] Remediation for {result.incident_id}"]
    for step in steps_executed:
        status = "✓" if step["success"] else "✗"
        output_lines.append(f"  {status} Step {step['step']}: {step['action']}")
        if step.get("output"):
            output_lines.append(f"    {step['output'][:200]}")
    if errors:
        output_lines.append(f"ERRORS: {'; '.join(errors)}")

    remediation_result = RemediationResult(
        incident_id=result.incident_id,
        mode=mode,
        executed_at=started,
        success=success,
        dry_run=dry_run,
        applied_resources=all_applied,
        steps_executed=steps_executed,
        output="\n".join(output_lines),
        error="; ".join(errors) if errors else None,
    )

    # Post to Slack
    await _notify_slack_remediation(result, remediation_result)

    return remediation_result


async def list_pending_confirmations() -> list[dict]:
    """Return all incidents waiting for manual confirmation."""
    return [
        {
            "incident_id": inc_id,
            "severity": data["result"].severity.value,
            "node": data["result"].node,
            "fix_category": data["result"].fix_category.value,
            "queued_at": data["queued_at"],
        }
        for inc_id, data in _pending_confirmations.items()
    ]


async def _notify_slack_remediation(
    diagnosis: DiagnosisResult,
    remediation: RemediationResult,
) -> None:
    """Post remediation result to Slack."""
    if not settings.slack_enabled or not settings.slack_webhook_url:
        return

    import json
    import aiohttp

    status_emoji = "✅" if remediation.success else "❌"
    mode_label = f"{'DRY RUN' if remediation.dry_run else 'LIVE'}"
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{status_emoji} Remediation {mode_label} — {diagnosis.incident_id}"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Node*\n`{diagnosis.node}`"},
                {"type": "mrkdwn", "text": f"*Fix*\n{diagnosis.fix_category.value}"},
                {"type": "mrkdwn", "text": f"*Mode*\n{remediation.mode.value}"},
                {"type": "mrkdwn", "text": f"*Success*\n{'Yes' if remediation.success else 'No'}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Output*\n```{remediation.output[:600]}```"},
        },
    ]
    payload = {
        "channel": settings.slack_channel,
        "username": "GPU Copilot — Remediation",
        "blocks": blocks,
        "text": f"{status_emoji} Remediation {mode_label} for {diagnosis.incident_id}",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                settings.slack_webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                remediation.slack_notified = resp.status == 200
    except Exception as e:
        logger.warning(f"Remediation Slack notify failed: {e}")
