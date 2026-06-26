"""
Prometheus Live Integration — Day 3
Queries real DCGM + node-exporter metrics from a running Prometheus instance.
Falls back to fixture data when PROMETHEUS_ENABLED=false (dev/test mode).

Metric queries map directly to what DCGM exports:
  dcgm_gpu_temp               — GPU temperature per GPU
  dcgm_gpu_utilization        — SM utilization %
  dcgm_fb_used / dcgm_fb_free — framebuffer (VRAM) used/free
  dcgm_ecc_dbe_aggregate      — uncorrectable ECC double-bit errors
  dcgm_ecc_sbe_aggregate      — correctable ECC single-bit errors
  dcgm_nvlink_bandwidth_total — NVLink bandwidth
  dcgm_sm_clock               — SM clock frequency
  dcgm_power_usage            — GPU power draw
  node_memory_*               — host RAM from node_exporter
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from app.core.config import settings
from app.core.logger import get_logger
from app.core.models import ClusterMetrics, GPUHealth, GPUMetric, NodeMetrics

logger = get_logger(__name__)

# DCGM Prometheus queries — these are the real metric names DCGM exports
DCGM_QUERIES = {
    "temperature":    'dcgm_gpu_temp{{instance="{node}"}}',
    "gpu_util":       'dcgm_gpu_utilization{{instance="{node}"}}',
    "mem_used":       'dcgm_fb_used{{instance="{node}"}}',
    "mem_free":       'dcgm_fb_free{{instance="{node}"}}',
    "ecc_dbe":        'dcgm_ecc_dbe_aggregate_total{{instance="{node}"}}',
    "ecc_sbe":        'dcgm_ecc_sbe_aggregate_total{{instance="{node}"}}',
    "sm_clock":       'dcgm_sm_clock{{instance="{node}"}}',
    "mem_clock":      'dcgm_mem_clock{{instance="{node}"}}',
    "power":          'dcgm_power_usage{{instance="{node}"}}',
    "fan_speed":      'dcgm_fan_speed_percent{{instance="{node}"}}',
    "nvlink_errors":  'dcgm_nvlink_bandwidth_total{{instance="{node}"}}',
    "node_ram_used":  'node_memory_MemTotal_bytes{{instance="{node}"}} - node_memory_MemAvailable_bytes{{instance="{node}"}}',
    "node_ram_total": 'node_memory_MemTotal_bytes{{instance="{node}"}}',
    "node_cpu_util":  '100 - (avg by (instance) (rate(node_cpu_seconds_total{{mode="idle",instance="{node}"}}[5m])) * 100)',
}


async def _prom_query(metric_expr: str) -> list[dict]:
    """Execute a single instant Prometheus query. Returns list of result dicts."""
    url = f"{settings.prometheus_url}/api/v1/query"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        async with session.get(url, params={"query": metric_expr}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("result", []) if data.get("status") == "success" else []


async def _prom_range_query(metric_expr: str, duration: str = "1h") -> list[dict]:
    """Execute a Prometheus range query for trend data."""
    import time
    end = int(time.time())
    start = end - _parse_duration_seconds(duration)
    url = f"{settings.prometheus_url}/api/v1/query_range"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        async with session.get(url, params={
            "query": metric_expr, "start": start, "end": end, "step": "60"
        }) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("result", []) if data.get("status") == "success" else []


def _parse_duration_seconds(duration: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return int(duration[:-1]) * units.get(duration[-1], 60)


def _extract_by_gpu(results: list[dict]) -> dict[int, float]:
    """Extract {gpu_id: value} from Prometheus instant query results."""
    out = {}
    for r in results:
        labels = r.get("metric", {})
        gpu_id = int(labels.get("gpu", labels.get("GPU", -1)))
        value = float(r.get("value", [0, "0"])[1])
        if gpu_id >= 0:
            out[gpu_id] = value
    return out


async def fetch_live_gpu_metrics(node: str, cluster: str = "gpu-cluster-prod-01") -> ClusterMetrics:
    """
    Query Prometheus for all GPU metrics on a given node.
    Returns a ClusterMetrics object in the same format as fixture data.
    """
    logger.info(f"Fetching live GPU metrics from Prometheus for node: {node}")

    # Run all queries concurrently
    import asyncio
    query_results = {}
    tasks = {
        name: _prom_query(expr.format(node=node))
        for name, expr in DCGM_QUERIES.items()
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for name, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.warning(f"Prometheus query '{name}' failed: {result}")
            query_results[name] = []
        else:
            query_results[name] = result

    # Extract per-GPU values
    temps       = _extract_by_gpu(query_results["temperature"])
    util        = _extract_by_gpu(query_results["gpu_util"])
    mem_used    = _extract_by_gpu(query_results["mem_used"])
    mem_free    = _extract_by_gpu(query_results["mem_free"])
    ecc_dbe     = _extract_by_gpu(query_results["ecc_dbe"])
    ecc_sbe     = _extract_by_gpu(query_results["ecc_sbe"])
    sm_clock    = _extract_by_gpu(query_results["sm_clock"])
    mem_clock   = _extract_by_gpu(query_results["mem_clock"])
    power       = _extract_by_gpu(query_results["power"])
    fan_speed   = _extract_by_gpu(query_results["fan_speed"])
    nvlink_err  = _extract_by_gpu(query_results["nvlink_errors"])

    gpu_ids = sorted(set(temps.keys()) | set(util.keys()) | set(mem_used.keys()))
    if not gpu_ids:
        logger.warning("No GPU data from Prometheus — falling back to fixtures")
        from app.core.fixtures import load_gpu_metrics
        return load_gpu_metrics()

    gpus = []
    for gpu_id in gpu_ids:
        temp = temps.get(gpu_id, 0.0)
        used_mb = int(mem_used.get(gpu_id, 0))
        free_mb = int(mem_free.get(gpu_id, 0))
        total_mb = used_mb + free_mb if free_mb > 0 else max(used_mb, 81920)
        dbe = int(ecc_dbe.get(gpu_id, 0))
        temp_c = temp

        # Determine health from raw signals
        if dbe > 0 or temp_c >= 95:
            health = GPUHealth.CRITICAL
        elif temp_c >= 87 or int(ecc_sbe.get(gpu_id, 0)) > 5:
            health = GPUHealth.WARNING
        else:
            health = GPUHealth.OK

        gpus.append(GPUMetric(
            gpu_id=gpu_id,
            name="NVIDIA A100 80GB",
            uuid=f"GPU-live-{node}-{gpu_id}",
            temperature_celsius=temp_c,
            temperature_threshold_slowdown=90.0,
            temperature_threshold_shutdown=95.0,
            utilization_gpu_percent=util.get(gpu_id, 0.0),
            utilization_memory_percent=round(used_mb / total_mb * 100, 1) if total_mb else 0,
            memory_total_mb=total_mb,
            memory_used_mb=used_mb,
            memory_free_mb=free_mb,
            power_draw_watts=power.get(gpu_id, 0.0),
            power_limit_watts=400.0,
            ecc_errors_single_bit=int(ecc_sbe.get(gpu_id, 0)),
            ecc_errors_double_bit=dbe,
            nvlink_errors=int(nvlink_err.get(gpu_id, 0)),
            sm_clock_mhz=int(sm_clock.get(gpu_id, 1410)),
            mem_clock_mhz=int(mem_clock.get(gpu_id, 1593)),
            fan_speed_percent=fan_speed.get(gpu_id, 0.0),
            health=health,
        ))

    # Node-level metrics
    ram_used_bytes  = float(query_results["node_ram_used"][0]["value"][1]) if query_results["node_ram_used"] else 0
    ram_total_bytes = float(query_results["node_ram_total"][0]["value"][1]) if query_results["node_ram_total"] else 512 * 1024**3
    cpu_util = float(query_results["node_cpu_util"][0]["value"][1]) if query_results["node_cpu_util"] else 0

    return ClusterMetrics(
        cluster=cluster,
        node=node,
        timestamp=datetime.now(timezone.utc),
        gpus=gpus,
        node_metrics=NodeMetrics(
            cpu_utilization_percent=cpu_util,
            ram_used_gb=round(ram_used_bytes / 1024**3, 1),
            ram_total_gb=round(ram_total_bytes / 1024**3, 1),
            disk_io_read_mb_per_s=0.0,
            disk_io_write_mb_per_s=0.0,
            network_rx_gb_per_s=0.0,
            network_tx_gb_per_s=0.0,
            uptime_hours=0.0,
        ),
    )


async def fetch_gpu_trends(node: str, gpu_id: int, duration: str = "1h") -> dict[str, Any]:
    """
    Fetch time-series trend data for a single GPU over a time window.
    Used by the dashboard endpoint for sparklines and anomaly detection.
    """
    logger.info(f"Fetching GPU {gpu_id} trends on {node} for last {duration}")
    temp_data = await _prom_range_query(
        f'dcgm_gpu_temp{{instance="{node}",gpu="{gpu_id}"}}', duration
    )
    util_data = await _prom_range_query(
        f'dcgm_gpu_utilization{{instance="{node}",gpu="{gpu_id}"}}', duration
    )
    ecc_data = await _prom_range_query(
        f'dcgm_ecc_dbe_aggregate_total{{instance="{node}",gpu="{gpu_id}"}}', duration
    )

    def extract_series(result: list[dict]) -> list[dict]:
        if not result:
            return []
        values = result[0].get("values", [])
        return [{"ts": int(v[0]), "v": float(v[1])} for v in values]

    return {
        "node": node,
        "gpu_id": gpu_id,
        "duration": duration,
        "temperature_series": extract_series(temp_data),
        "utilization_series": extract_series(util_data),
        "ecc_dbe_series": extract_series(ecc_data),
    }


async def get_metrics_for_alert(alert: "AlertPayload") -> ClusterMetrics:  # noqa: F821
    """
    Entry point called by the /alert/webhook route.
    Chooses live Prometheus or fixture data based on PROMETHEUS_ENABLED.
    """
    node = alert.commonLabels.get("node", "gpu-node-03")
    cluster = alert.commonLabels.get("cluster", "gpu-cluster-prod-01")

    if settings.prometheus_enabled:
        try:
            return await fetch_live_gpu_metrics(node, cluster)
        except Exception as e:
            logger.error(f"Live Prometheus fetch failed: {e} — falling back to fixtures")

    from app.core.fixtures import load_gpu_metrics
    return load_gpu_metrics()
