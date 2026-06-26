"""
Cluster Scanner — Day 4
Scans all GPU nodes in a cluster, identifies unhealthy ones,
and runs parallel diagnosis pipelines for each.

In mock mode: generates synthetic multi-node scenarios from the fixture.
In live mode: discovers nodes via Kubernetes node labels + Prometheus.

This is the most impressive demo endpoint — one call, full cluster view.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import get_logger
from app.core.models import ClusterMetrics, DiagnosisResult, GPUHealth

logger = get_logger(__name__)


class NodeScanResult(BaseModel):
    node: str
    healthy: bool
    unhealthy_gpu_count: int
    total_gpu_count: int
    diagnosis: DiagnosisResult | None = None
    skipped_reason: str | None = None
    scan_duration_seconds: float = 0.0


class ClusterScanResult(BaseModel):
    cluster: str
    scanned_at: datetime
    total_nodes: int
    healthy_nodes: int
    unhealthy_nodes: int
    total_scan_duration_seconds: float
    node_results: list[NodeScanResult]
    summary: str


async def _get_cluster_nodes() -> list[str]:
    """
    Return list of GPU node names.
    Live: queries kubectl for nodes with nvidia.com/gpu label.
    Mock: returns synthetic multi-node list.
    """
    if settings.k8s_enabled:
        from app.integrations.kubernetes import _kubectl
        rc, stdout, _ = _kubectl(
            "get", "nodes",
            "-l", "nvidia.com/gpu",
            "-o", "jsonpath={.items[*].metadata.name}",
        )
        if rc == 0 and stdout.strip():
            return stdout.strip().split()
        logger.warning("kubectl node discovery failed — using mock nodes")

    # Mock: simulate a 3-node GPU cluster
    return ["gpu-node-01", "gpu-node-02", "gpu-node-03"]


async def _get_node_metrics(node: str) -> ClusterMetrics:
    """Fetch GPU metrics for a single node (live or fixture)."""
    if settings.prometheus_enabled:
        from app.integrations.prometheus import fetch_live_gpu_metrics
        try:
            return await fetch_live_gpu_metrics(node)
        except Exception as e:
            logger.warning(f"Prometheus failed for {node}: {e} — using fixture")

    from app.core.fixtures import load_gpu_metrics
    metrics = load_gpu_metrics()

    # For multi-node mock, synthesize different health states per node
    import copy, json
    from app.core.models import ClusterMetrics
    raw = json.loads(metrics.model_dump_json())
    raw["node"] = node

    # node-01: all healthy, node-02: one warning, node-03: one critical (real fixture)
    if node == "gpu-node-01":
        for g in raw["gpus"]:
            g["health"] = "OK"
            g["temperature_celsius"] = 65.0
            g["ecc_errors_double_bit"] = 0
    elif node == "gpu-node-02":
        for g in raw["gpus"]:
            g["health"] = "OK"
            g["temperature_celsius"] = 70.0
            g["ecc_errors_double_bit"] = 0
        raw["gpus"][1]["health"] = "WARNING"
        raw["gpus"][1]["temperature_celsius"] = 88.0

    return ClusterMetrics(**raw)


async def _scan_single_node(node: str) -> NodeScanResult:
    """Run a full scan + optional diagnosis on a single node."""
    import time
    start = time.time()

    try:
        metrics = await _get_node_metrics(node)
    except Exception as e:
        return NodeScanResult(
            node=node, healthy=True,
            unhealthy_gpu_count=0, total_gpu_count=0,
            skipped_reason=f"Metrics unavailable: {e}",
            scan_duration_seconds=time.time() - start,
        )

    unhealthy = [g for g in metrics.gpus if g.health in (GPUHealth.CRITICAL, GPUHealth.WARNING, GPUHealth.UNHEALTHY)]
    is_healthy = len(unhealthy) == 0

    if is_healthy:
        return NodeScanResult(
            node=node, healthy=True,
            unhealthy_gpu_count=0, total_gpu_count=len(metrics.gpus),
            scan_duration_seconds=round(time.time() - start, 2),
        )

    # Node has unhealthy GPUs — run diagnosis
    logger.info(f"Cluster scan: {node} has {len(unhealthy)} unhealthy GPU(s) — diagnosing")
    try:
        from app.agent.graph import run_diagnosis
        from app.core.fixtures import load_logs, load_k8s_patch

        raw_logs = load_logs([
            "fixtures/logs/cuda_ecc_error.txt",
            "fixtures/logs/oom_killed.txt",
            "fixtures/logs/node_pressure.txt",
        ])
        k8s_patch = load_k8s_patch("fixtures/expected/k8s_patch_gpu_drain.yaml")

        alert_summary = (
            f"{len(unhealthy)} unhealthy GPU(s) detected on {node}. "
            f"GPUs: {[g.gpu_id for g in unhealthy]}. "
            f"Max temp: {max(g.temperature_celsius for g in unhealthy)}°C."
        )

        diagnosis = await run_diagnosis(
            scenario_id=f"cluster_scan_{node}",
            metrics=metrics,
            alert_summary=alert_summary,
            raw_logs=raw_logs,
            k8s_patch_template=k8s_patch,
        )

        return NodeScanResult(
            node=node, healthy=False,
            unhealthy_gpu_count=len(unhealthy),
            total_gpu_count=len(metrics.gpus),
            diagnosis=diagnosis,
            scan_duration_seconds=round(time.time() - start, 2),
        )

    except Exception as e:
        logger.error(f"Diagnosis failed for {node}: {e}")
        return NodeScanResult(
            node=node, healthy=False,
            unhealthy_gpu_count=len(unhealthy),
            total_gpu_count=len(metrics.gpus),
            skipped_reason=f"Diagnosis failed: {e}",
            scan_duration_seconds=round(time.time() - start, 2),
        )


async def scan_cluster(cluster: str = "gpu-cluster-prod-01") -> ClusterScanResult:
    """
    Scan all nodes in the cluster in parallel.
    Returns a ClusterScanResult with per-node diagnoses.
    """
    import time
    start = time.time()
    logger.info(f"Starting cluster scan for {cluster}")

    nodes = await _get_cluster_nodes()
    logger.info(f"Discovered {len(nodes)} GPU node(s): {nodes}")

    # Parallel scan — all nodes concurrently
    node_results = await asyncio.gather(
        *[_scan_single_node(node) for node in nodes],
        return_exceptions=False,
    )

    healthy = [r for r in node_results if r.healthy]
    unhealthy = [r for r in node_results if not r.healthy]
    total_duration = round(time.time() - start, 2)

    if unhealthy:
        severities = [r.diagnosis.severity.value for r in unhealthy if r.diagnosis]
        summary = (
            f"{len(unhealthy)}/{len(nodes)} nodes unhealthy. "
            f"Severities: {', '.join(severities)}. "
            f"Immediate action required on: {[r.node for r in unhealthy]}."
        )
    else:
        summary = f"All {len(nodes)} nodes healthy. No action required."

    logger.info(f"Cluster scan complete in {total_duration}s: {summary}")

    return ClusterScanResult(
        cluster=cluster,
        scanned_at=datetime.now(timezone.utc),
        total_nodes=len(nodes),
        healthy_nodes=len(healthy),
        unhealthy_nodes=len(unhealthy),
        total_scan_duration_seconds=total_duration,
        node_results=list(node_results),
        summary=summary,
    )
