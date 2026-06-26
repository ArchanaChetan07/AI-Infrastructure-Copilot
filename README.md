<div align="center">

<img src="https://img.shields.io/badge/status-active-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/LangGraph-0.2-FF6B35?style=flat-square" />
<img src="https://img.shields.io/badge/Claude_3.5_Sonnet-CC785C?style=flat-square" />
<img src="https://img.shields.io/badge/tests-136_passing-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" />

# AI Infrastructure Copilot

**Autonomous GPU incident diagnosis, remediation, and cluster-wide scanning**

*Reduced mean investigation time from 40 minutes to under 3 minutes using LLM agents, RAG over historical incidents, and automated Kubernetes remediation.*

[Quick Start](#quick-start) · [Architecture](#architecture) · [API Reference](#api-reference) · [Tests](#tests) · [Deployment](#deployment)

</div>

---

## The Problem

GPU clusters fail in ways that are expensive to debug. When a vLLM inference pod crashes at 2am, an on-call SRE must manually:

- Parse DCGM telemetry across 8+ GPUs per node
- Correlate ECC errors, thermal events, NVLink faults, and Kubernetes pod logs
- Identify whether the failure is hardware (GPU reset required) or software (config patch)
- Determine the blast radius across the cluster
- Generate correct `kubectl` patch YAML and remediation steps
- Write up the incident for Slack and the runbook library

This takes **30–45 minutes** and requires deep expertise across NVIDIA GPU internals, CUDA, and Kubernetes. A single on-call rotation can exhaust an SRE team.

---

## The Solution

An AI SRE Copilot that runs a 5-node LangGraph reasoning pipeline over live GPU telemetry, pod logs, and Alertmanager webhooks — and produces a complete diagnosis in under 3 minutes, including root cause, Kubernetes patch YAML, Slack notification, and a downloadable Markdown runbook.

The agent improves over time: every new diagnosis is embedded into the Qdrant vector store and retrieved as context for future incidents of the same type.

---

## Key Features

**Autonomous Diagnosis**
The LangGraph agent ingests GPU metrics, pod logs, and alert context, then runs structured reasoning across 5 nodes to produce a precise, technical root cause with confidence score.

**RAG-Augmented Analysis**
Before reasoning, the agent queries a Qdrant vector store of historical GPU incidents using semantic search. Past resolutions directly inform the current diagnosis, reducing hallucination and improving fix category accuracy.

**Kubernetes Remediation**
The agent generates multi-document YAML patches with three execution modes — dry-run (always safe), confirm (human-in-the-loop), and auto (critical incidents only). Every patch includes cordon, drain, GPU reset Job, and PodDisruptionBudget.

**Cluster-Wide Scanning**
A single API call discovers all GPU nodes, queries their metrics in parallel, and runs diagnosis pipelines concurrently on any node with unhealthy GPUs. The entire cluster is assessed in one shot.

**Live Infrastructure Integration**
Connects to real Prometheus/DCGM for GPU metrics and real Kubernetes for pod logs and patch execution. Falls back to realistic fixture data with zero configuration for development and CI.

**Alertmanager + Grafana Webhooks**
Drop the webhook URL into your Alertmanager `receivers` config. The full pipeline triggers automatically on every GPU alert, with Grafana unified alerting format also supported.

**Async Job Queue**
Long-running diagnoses run in background tasks so HTTP clients aren't blocked. Submit a job, get a job ID, poll for results.

**Analytics Dashboard**
MTTR trend data, node-level failure heatmaps, and recurrence detection (same node, same failure type) with actionable recommendations per pattern.

**Production Middleware**
Request ID tracing on every response, structured JSON request logging with latency, per-IP rate limiting with sliding window, and API key authentication — all configurable via environment variables.

---

## Architecture

```
Alertmanager / Grafana Webhook ──► POST /api/v1/alert/webhook
                                          │
                              ┌───────────▼───────────┐
                              │  RequestID Middleware  │  ← X-Request-ID header
                              │  Auth + Rate Limiter   │  ← X-API-Key / 60 req/min
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │  FastAPI Route Layer   │
                              │  BackgroundTasks       │  ← async job queue
                              └───────────┬───────────┘
                                          │
                    ┌─────────────────────▼──────────────────────┐
                    │           LangGraph Agent Pipeline          │
                    │                                            │
                    │  ① fetch_context                          │
                    │     GPU metrics → compact signal dict      │
                    │     Log files → ERROR/CRITICAL lines       │
                    │                   │                        │
                    │  ② rag_retrieve   │                        │
                    │     Build query from signal signatures ◄───┤
                    │     Qdrant cosine similarity search        │
                    │     Inject top-3 historical incidents      │
                    │                   │                        │
                    │  ③ analyze_signals│                        │
                    │     LLM: rank anomalies vs thresholds      │
                    │     RAG context informs signal weighting   │
                    │                   │                        │
                    │  ④ root_cause     │                        │
                    │     LLM: definitive RCA from all signals   │
                    │     Severity + fix_category classification │
                    │                   │                        │
                    │  ⑤ recommend_fix  │                        │
                    │     LLM: ordered runbook + K8s patch YAML  │
                    │     Past remediation steps as examples     │
                    └─────────────────┬──────────────────────────┘
                                      │
                    ┌─────────────────▼──────────────────────────┐
                    │              Post-Pipeline                  │
                    │  asyncio.gather(slack, postgres, qdrant)   │
                    │  Slack Block Kit → #gpu-alerts             │
                    │  PostgreSQL → incident history             │
                    │  Qdrant upsert → self-improving corpus     │
                    └────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API server | FastAPI + Uvicorn | REST API, async request handling, middleware |
| LLM agent | LangGraph + Claude 3.5 Sonnet | 5-node reasoning graph |
| Vector store | Qdrant + all-MiniLM-L6-v2 | RAG over historical incidents |
| GPU telemetry | DCGM + Prometheus | Real-time GPU health metrics (12 series) |
| Orchestration | Kubernetes API + kubectl | Pod logs, node cordon/drain, patch application |
| Database | PostgreSQL + async SQLAlchemy | Incident history, MTTR analytics |
| Notifications | Slack Incoming Webhooks | Block Kit messages with full diagnosis |
| Monitoring | Alertmanager + Grafana | Alert routing and webhook integration |
| Inference targets | NVIDIA Triton + vLLM | The workloads being monitored |

---

## Project Structure

```
AI-Infrastructure-Copilot/
│
├── app/
│   ├── main.py                        # App entrypoint, middleware, K8s probes
│   ├── agent/
│   │   └── graph.py                   # LangGraph 5-node pipeline (549 lines)
│   ├── api/
│   │   └── routes.py                  # 30+ REST endpoints (753 lines)
│   ├── core/
│   │   ├── config.py                  # Pydantic settings with feature flags
│   │   ├── fixtures.py                # Mock data loader
│   │   ├── logger.py                  # Structured logger
│   │   └── models.py                  # Request/response Pydantic schemas
│   ├── db/
│   │   └── database.py                # Async SQLAlchemy ORM + CRUD
│   ├── integrations/
│   │   ├── prometheus.py              # Live DCGM metric queries (12 series)
│   │   └── kubernetes.py              # kubectl: logs, patch, cordon, drain
│   ├── middleware/
│   │   ├── auth.py                    # API key auth + sliding window rate limit
│   │   └── logging.py                 # Request ID injection + latency logging
│   └── services/
│       ├── analytics_service.py       # MTTR trends, failure heatmaps, recurrences
│       ├── cluster_scanner.py         # Parallel multi-node scan + diagnosis
│       ├── job_queue.py               # Async background job queue
│       ├── qdrant_service.py          # RAG: embed, retrieve, upsert
│       ├── remediation_service.py     # Three-mode K8s patch executor
│       ├── runbook_service.py         # Markdown incident runbook generator
│       └── slack_service.py           # Slack Block Kit message builder
│
├── fixtures/                          # Realistic mock data (no GPU required)
│   ├── gpu_metrics.json               # 4× NVIDIA A100 80GB (GPU 2 critical)
│   ├── alert_payload.json             # Alertmanager webhook (GPUTemperatureCritical)
│   ├── incidents/
│   │   └── historical_incidents.json  # 6 past incidents used to seed Qdrant
│   ├── logs/
│   │   ├── oom_killed.txt             # vLLM OOMKill with full CUDA stack trace
│   │   ├── cuda_ecc_error.txt         # DCGM ECC double-bit error + nvidia-smi
│   │   └── node_pressure.txt          # Kubelet eviction + kubectl describe node
│   ├── expected/
│   │   └── k8s_patch_gpu_drain.yaml   # AI-generated K8s remediation (4 documents)
│   └── scenarios/
│       └── scenarios.json             # 3 demo failure scenarios
│
├── tests/
│   ├── test_day1.py                   #  11 tests — fixtures, models, API skeleton
│   ├── test_day2.py                   #  34 tests — RAG, Slack, PostgreSQL
│   ├── test_day3.py                   #  41 tests — Prometheus, K8s, remediation, analytics
│   ├── test_day4.py                   #  50 tests — middleware, jobs, cluster scan, runbook
│   └── test_integration.py            #  49 tests — real LLM calls (requires API key)
│
├── Dockerfile
├── docker-compose.yml                 # Copilot + Qdrant + PostgreSQL
└── .github/workflows/ci.yml           # Matrix CI: unit + integration + ruff lint
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) (free tier works)

### 1. Clone and install

```bash
git clone https://github.com/ArchanaChetan07/AI-Infrastructure-Copilot.git
cd AI-Infrastructure-Copilot

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — the only required value is ANTHROPIC_API_KEY
```

### 3. Run

```bash
uvicorn app.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000`. Open `http://localhost:8000/docs` for interactive Swagger UI.

### 4. Try the primary demo scenario

```bash
# GPU thermal throttle → ECC double-bit error → vLLM OOMKill
curl -s -X POST http://localhost:8000/api/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "gpu_thermal_throttle_ecc"}' | python3 -m json.tool
```

**Expected output (abbreviated):**

```json
{
  "incident_id": "INC-20240115-A3F2BC8E",
  "severity": "critical",
  "fix_category": "gpu_drain_and_reset",
  "investigation_duration_seconds": 12.4,
  "confidence": 0.94,
  "root_cause": "GPU 2 entered thermal throttle at 91°C due to sustained load. Thermal stress triggered an uncorrectable DRAM double-bit ECC error (row=0x1F8C, bank=7), causing the CUDA driver to signal cudaErrorECCUncorrectable. The vLLM process OOMKilled during graceful drain.",
  "contributing_factors": [
    "GPU 2 fan at 100% — cooling system at capacity",
    "Double-bit ECC errors are uncorrectable and require a full GPU reset",
    "NVLink errors preceded the thermal event by 8 minutes — early warning signal"
  ],
  "remediation_steps": [
    {"step": 1, "action": "Cordon node", "command": "kubectl cordon gpu-node-03"},
    {"step": 2, "action": "Drain workloads", "command": "kubectl drain gpu-node-03 --ignore-daemonsets"},
    {"step": 3, "action": "Reset GPU ECC", "command": "nvidia-smi --id=2 --gpu-reset"},
    {"step": 4, "action": "Uncordon node", "command": "kubectl uncordon gpu-node-03"}
  ],
  "agent_trace": [
    "fetch_context: 2 unhealthy GPU(s), 3 log snippet(s)",
    "rag_retrieve: 2 similar incidents — INC-20231201-ECC001, INC-20240105-THERMAL001",
    "analyze_signals: primary anomaly = 'GPU 2 thermal throttle + ECC DBE' (RAG-augmented)",
    "root_cause: CRITICAL — gpu_drain_and_reset (informed by ['INC-20231201-ECC001'])",
    "recommend_fix: 4 steps, est. 15 min resolution"
  ]
}
```

---

## Demo Scenarios

Three pre-built GPU failure scenarios exercise different diagnosis paths:

| Scenario | Failure Mode | Severity | Expected Fix |
|---|---|---|---|
| `gpu_thermal_throttle_ecc` | GPU overheating (91°C) → uncorrectable ECC double-bit error → vLLM OOMKill | **Critical** | Node cordon + GPU drain + ECC reset |
| `gpu_memory_oom` | Oversized inference batch exceeds VRAM → CUDA OOM → pod crash loop | High | Config patch (batch size + CUDA_VISIBLE_DEVICES) |
| `nvlink_degraded` | Persistent NVLink replay errors → tensor-parallel bandwidth loss → P99 latency SLA breach | Warning | NVLink reset + fabricmanager restart |

---

## API Reference

### Diagnosis

```
POST /api/v1/diagnose                    Synchronous diagnosis (12–45s)
POST /api/v1/jobs/diagnose               Submit async job, returns immediately
GET  /api/v1/jobs/{job_id}               Poll job status and result
GET  /api/v1/jobs                        List recent jobs
```

### Cluster Operations

```
POST /api/v1/cluster/scan                Scan all nodes, diagnose unhealthy in parallel
GET  /api/v1/cluster/nodes               List discovered GPU-capable nodes
```

### Remediation

```
POST /api/v1/remediate/{id}              Execute K8s patch (mode: dry_run | confirm | auto)
POST /api/v1/remediate/{id}/confirm      Approve a queued remediation
GET  /api/v1/remediate/pending           List pending human approvals
```

### Analytics Dashboard

```
GET  /api/v1/dashboard/summary           Incident counts, avg MTTR, top failing nodes
GET  /api/v1/dashboard/mttr-trend        Daily MTTR trend over N days
GET  /api/v1/dashboard/heatmap           Node × fix_category incident heatmap
GET  /api/v1/dashboard/recurrences       Recurring failure patterns + recommendations
```

### Incident History

```
GET  /api/v1/incidents                   List persisted incidents (requires Postgres)
GET  /api/v1/incidents/{id}              Get single incident
GET  /api/v1/incidents/stats/mttr        MTTR stats grouped by severity
```

### RAG

```
POST /api/v1/rag/search                  Semantic search over the incident corpus
GET  /api/v1/rag/incidents               List all seeded historical incidents
POST /api/v1/rag/seed                    Force re-seed Qdrant from fixtures
```

### Runbook & Webhooks

```
GET  /api/v1/runbook/{scenario_id}       Download Markdown incident runbook
POST /api/v1/alert/webhook               Alertmanager + Grafana webhook receiver
GET  /api/v1/slack/preview/{scenario_id} Preview Slack Block Kit message
```

### System

```
GET  /health                             Liveness probe
GET  /ready                              Readiness probe (Qdrant + Postgres checks)
GET  /metrics                            Prometheus metrics (job counts, avg duration)
```

---

## Tests

136 unit tests run with no external dependencies. 49 integration tests make real Claude API calls when an API key is present.

```bash
# Unit tests — no API key required (~6s)
pytest tests/test_day1.py tests/test_day2.py tests/test_day3.py tests/test_day4.py -v

# Integration tests — requires ANTHROPIC_API_KEY (~60s, ~$0.05)
ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_integration.py -v -m "not slow"

# Full suite including all 3 scenario end-to-end runs
ANTHROPIC_API_KEY=sk-ant-... pytest tests/ -v

# Single test group
ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_integration.py::TestDiagnosisQuality -v -s
```

### Test coverage by area

| Test file | Tests | What's covered |
|---|---|---|
| `test_day1.py` | 11 | Fixture loading, model validation, API skeleton |
| `test_day2.py` | 34 | Qdrant RAG retrieval, Slack Block Kit structure, PostgreSQL CRUD |
| `test_day3.py` | 41 | Prometheus fallback, kubectl mocking, remediation safety modes, analytics |
| `test_day4.py` | 50 | Request ID headers, rate limit sliding window, auth middleware, job queue lifecycle, runbook Markdown output, cluster scanner parallel diagnosis |
| `test_integration.py` | 49 | Real LLM output quality — severity correctness, root cause keywords, RAG retrieval accuracy, K8s command presence, confidence thresholds |

---

## Configuration

All settings are read from `.env`. Copy `.env.example` to get started.

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Feature flags — all default to false (safe to run without any external services)
QDRANT_ENABLED=false          # true = persistent Qdrant; false = in-memory
POSTGRES_ENABLED=false        # true = persist incidents; false = no-op
SLACK_ENABLED=false           # true = post to Slack on each diagnosis
PROMETHEUS_ENABLED=false      # true = query real Prometheus for GPU metrics
K8S_ENABLED=false             # true = real kubectl calls; false = dry output only

# Security (Day 4)
API_KEY=                      # blank = open (dev mode); set to require X-API-Key header
RATE_LIMIT_PER_MINUTE=60      # 0 = disabled

# Slack (requires SLACK_ENABLED=true)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_CHANNEL=#gpu-alerts
```

---

## Deployment

### Docker Compose (full stack)

```bash
# Start app + Qdrant + PostgreSQL
ANTHROPIC_API_KEY=sk-ant-... \
QDRANT_ENABLED=true \
POSTGRES_ENABLED=true \
docker compose up --build
```

### Alertmanager Integration

Add to your `alertmanager.yml`:

```yaml
receivers:
  - name: gpu-copilot
    webhook_configs:
      - url: http://gpu-copilot:8000/api/v1/alert/webhook
        send_resolved: false
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 15
```

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: gpu-copilot
    static_configs:
      - targets: ['gpu-copilot:8000']
    metrics_path: /metrics
```

---

## LangGraph Agent — Node Detail

```python
# Node 1 — fetch_context
# Input:  raw GPU metrics + log files + alert summary
# Output: compact metrics dict, filtered log snippets (ERROR/CRITICAL lines only)
# Note:   Day 2 added similar_incidents field for RAG injection

# Node 2 — rag_retrieve  (added Day 2)
# Input:  metrics_summary, log snippets
# Action: builds semantic query from signal signatures
#         (e.g. "GPU ECC double-bit uncorrectable thermal throttle 91C")
#         queries Qdrant for top-3 similar historical incidents
# Output: similar_incidents[] with similarity scores
# Fails:  gracefully — empty list if Qdrant unavailable

# Node 3 — analyze_signals
# Input:  metrics, logs, alert, similar_incidents (RAG context)
# LLM:    identifies primary anomaly, ranks all signals by severity
# Output: anomalous_signals[], signal_timeline, confidence

# Node 4 — root_cause
# Input:  signals, metrics, logs, similar resolved incidents
# LLM:    determines definitive root cause + fix_category
# Output: root_cause str, contributing_factors[], severity, fix_category

# Node 5 — recommend_fix
# Input:  root_cause, fix_category, past remediation steps (RAG)
# LLM:    generates ordered runbook + multi-document K8s patch YAML
# Output: remediation_steps[], k8s_patch_yaml
```

---

## Remediation Safety Model

The remediation service uses three execution modes to balance speed against safety:

```
DRY_RUN (default)
  kubectl apply --dry-run=server
  Nothing touches the cluster. Always safe to call.
  Returns exactly what would happen.

CONFIRM
  Patch is queued in memory.
  POST /remediate/{id}/confirm required to execute.
  Human-in-the-loop gate for production.

AUTO
  Executes immediately.
  Only permitted for CRITICAL and HIGH severity.
  Auto-downgrades to DRY_RUN for WARNING and INFO.
```

Step sequence for `gpu_drain_and_reset`:

```
① kubectl cordon {node}                   (isolate — no new pods)
② kubectl apply -f patch.yaml             (deploy reset Job, update Deployment)
③ kubectl drain {node} --ignore-daemonsets (evict running pods)
④ [GPU reset Job runs nvidia-smi --gpu-reset]
⑤ kubectl uncordon {node}                 (re-enable after health verified)
```

---

## RAG Incident Corpus

Six historical GPU incidents are pre-loaded into Qdrant at startup. Each incident becomes a vector that the agent retrieves during diagnosis:

| Incident ID | Severity | Failure | Fix |
|---|---|---|---|
| INC-20231201-ECC001 | Critical | GPU ECC double-bit error after 18h training run | GPU drain + reset |
| INC-20231215-OOM001 | High | Triton CUDA OOM from oversized batch request | Config patch |
| INC-20231220-NVLINK001 | Warning | NVLink replay errors from ungraceful shutdown | NVLink reset |
| INC-20240105-THERMAL001 | Critical | GPU thermal shutdown — fan controller firmware bug | Firmware update + reset |
| INC-20240110-MEM001 | High | vLLM KV cache memory leak — v0.2.7 bug | Pod restart + version upgrade |
| INC-20240112-PCIE001 | High | PCIe bus errors from loose connector post-maintenance | Physical re-seat |

Every newly diagnosed incident is automatically upserted into the corpus, making the system incrementally smarter with each incident.

---

## Metrics and Observability

Every HTTP response includes:
- `X-Request-ID` — unique per request, propagates through logs
- `X-Response-Time-Ms` — end-to-end latency in milliseconds

Every log line is structured JSON with `request_id`, `method`, `path`, `status`, and `latency_ms` — ready to ingest into Loki, Datadog, or any JSON log aggregator.

`GET /metrics` exposes Prometheus-format counters:

```
gpu_copilot_jobs_total{status="done"}     12
gpu_copilot_jobs_total{status="failed"}    1
gpu_copilot_jobs_total{status="running"}   0
gpu_copilot_avg_diagnosis_seconds         11.40
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with FastAPI · LangGraph · Claude 3.5 Sonnet · Qdrant · Kubernetes

*4,283 lines of application code · 136 unit tests · 49 integration tests*

</div>
