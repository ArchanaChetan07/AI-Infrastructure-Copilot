"""
Pydantic models — request payloads, response schemas, and internal data types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    WARNING = "warning"
    INFO = "info"


class GPUHealth(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNHEALTHY = "UNHEALTHY"


class FixCategory(str, Enum):
    GPU_DRAIN_AND_RESET = "gpu_drain_and_reset"
    CONFIG_PATCH = "config_patch"
    NVLINK_RESET = "nvlink_reset"
    POD_RESTART = "pod_restart"
    NODE_DRAIN = "node_drain"
    MANUAL_INTERVENTION = "manual_intervention"


# ── GPU Metrics ───────────────────────────────────────────────────────────────

class GPUMetric(BaseModel):
    gpu_id: int
    name: str
    uuid: str
    temperature_celsius: float
    temperature_threshold_slowdown: float
    temperature_threshold_shutdown: float
    utilization_gpu_percent: float
    utilization_memory_percent: float
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    power_draw_watts: float
    power_limit_watts: float
    ecc_errors_single_bit: int
    ecc_errors_double_bit: int
    nvlink_errors: int
    sm_clock_mhz: int
    mem_clock_mhz: int
    fan_speed_percent: float
    health: GPUHealth


class NodeMetrics(BaseModel):
    cpu_utilization_percent: float
    ram_used_gb: float
    ram_total_gb: float
    disk_io_read_mb_per_s: float
    disk_io_write_mb_per_s: float
    network_rx_gb_per_s: float
    network_tx_gb_per_s: float
    uptime_hours: float


class ClusterMetrics(BaseModel):
    cluster: str
    node: str
    timestamp: datetime
    gpus: list[GPUMetric]
    node_metrics: NodeMetrics


# ── Alert Payload (Alertmanager webhook schema) ───────────────────────────────

class AlertLabel(BaseModel):
    model_config = {"extra": "allow"}
    alertname: str
    severity: str
    cluster: str | None = None
    node: str | None = None
    gpu_id: str | None = None
    namespace: str | None = None
    pod: str | None = None


class AlertAnnotation(BaseModel):
    model_config = {"extra": "allow"}
    summary: str
    description: str | None = None


class Alert(BaseModel):
    status: str
    labels: AlertLabel
    annotations: AlertAnnotation
    startsAt: datetime
    fingerprint: str | None = None


class AlertPayload(BaseModel):
    """Alertmanager webhook payload."""
    version: str = "4"
    status: str
    receiver: str
    groupLabels: dict[str, str] = Field(default_factory=dict)
    commonLabels: dict[str, str] = Field(default_factory=dict)
    commonAnnotations: dict[str, str] = Field(default_factory=dict)
    alerts: list[Alert]


# ── Diagnosis Request / Response ──────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    """
    POST /diagnose — trigger a diagnosis run.
    In mock mode (use_mock_data=True), scenario_id selects the fixture.
    In live mode, alert_payload drives everything.
    """
    scenario_id: str | None = Field(
        default="gpu_thermal_throttle_ecc",
        description="Demo scenario ID from fixtures/scenarios/scenarios.json",
    )
    alert_payload: AlertPayload | None = Field(
        default=None,
        description="Live Alertmanager webhook payload (used in production mode)",
    )


class RemediationStep(BaseModel):
    step: int
    action: str
    command: str | None = None
    description: str


class DiagnosisResult(BaseModel):
    incident_id: str
    scenario_id: str | None
    node: str
    affected_gpu: int | None
    pod: str | None
    namespace: str | None

    # Timing
    diagnosed_at: datetime
    investigation_duration_seconds: float

    # Agent output
    severity: Severity
    root_cause: str
    contributing_factors: list[str]
    fix_category: FixCategory
    remediation_steps: list[RemediationStep]
    k8s_patch_yaml: str | None

    # Signals used
    gpu_metrics_summary: dict[str, Any]
    log_snippets: list[str]
    alert_summary: str

    # Meta
    agent_trace: list[str] = Field(
        description="LangGraph node execution trace for observability"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    slack_notified: bool = False


# ── Health / Status ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str = "0.1.0"
    mode: str  # "mock" or "live"
