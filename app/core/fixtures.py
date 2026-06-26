"""
Fixture loader — reads mock GPU metrics, alert payloads, and logs from disk.
In Day 1, this replaces live DCGM / Prometheus / Kubernetes calls.
"""

import json
from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger
from app.core.models import AlertPayload, ClusterMetrics

logger = get_logger(__name__)

FIXTURES = Path(settings.fixtures_dir)


def load_gpu_metrics(metrics_path: str | None = None) -> ClusterMetrics:
    path = Path(metrics_path) if metrics_path else FIXTURES / "gpu_metrics.json"
    logger.info(f"Loading GPU metrics from {path}")
    data = json.loads(path.read_text())
    return ClusterMetrics(**data)


def load_alert_payload(alert_path: str | None = None) -> AlertPayload:
    path = Path(alert_path) if alert_path else FIXTURES / "alert_payload.json"
    logger.info(f"Loading alert payload from {path}")
    data = json.loads(path.read_text())
    return AlertPayload(**data)


def load_logs(log_paths: list[str]) -> list[str]:
    """Load raw log text from fixture files."""
    logs = []
    for lp in log_paths:
        p = Path(lp)
        if p.exists():
            logger.info(f"Loading log: {p}")
            logs.append(p.read_text())
        else:
            logger.warning(f"Log fixture not found: {p}")
    return logs


def load_scenario(scenario_id: str) -> dict:
    """Look up a scenario definition by ID."""
    scenario_file = FIXTURES / "scenarios" / "scenarios.json"
    data = json.loads(scenario_file.read_text())
    for s in data["scenarios"]:
        if s["id"] == scenario_id:
            return s
    raise ValueError(f"Scenario '{scenario_id}' not found in {scenario_file}")


def load_k8s_patch(patch_path: str) -> str:
    """Load the expected Kubernetes patch YAML."""
    p = Path(patch_path)
    if p.exists():
        return p.read_text()
    return ""


def load_scenario_bundle(scenario_id: str) -> dict:
    """
    Load everything needed for a scenario:
    - scenario metadata
    - GPU metrics
    - alert payload
    - all log files
    - expected K8s patch YAML
    """
    scenario = load_scenario(scenario_id)
    files = scenario["fixture_files"]

    metrics = load_gpu_metrics(files.get("metrics"))
    alert = load_alert_payload(files.get("alert"))
    logs = load_logs(files.get("logs", []))
    k8s_patch = load_k8s_patch(scenario.get("expected_k8s_patch", ""))

    return {
        "scenario": scenario,
        "metrics": metrics,
        "alert": alert,
        "logs": logs,
        "k8s_patch": k8s_patch,
    }
