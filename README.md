<div align="center">

# AI Infrastructure Copilot

**Conversational AI for LLM deployment, GPU management, and infrastructure automation**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LLM](https://img.shields.io/badge/LLM-Powered-FF6B35?style=flat-square)](https://github.com/ArchanaChetan07/AI-Infrastructure-Copilot)
[![MLOps](https://img.shields.io/badge/MLOps-Automation-6E40C9?style=flat-square)](https://github.com/ArchanaChetan07/AI-Infrastructure-Copilot)
[![GPU](https://img.shields.io/badge/NVIDIA-GPU%20Management-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat-square)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](Dockerfile)

*"How many GPU replicas do I need for 1000 concurrent users?" — ask the copilot.*

</div>

---

## What is AI Infrastructure Copilot?

A conversational AI assistant purpose-built for LLM infrastructure decisions. Instead of reading documentation and doing math manually, you describe your workload and the copilot reasons through deployment configurations, capacity planning, and troubleshooting steps.

**Who it's for:** ML engineers deploying models · Platform teams managing GPU clusters · Startups scaling their first LLM API

---

## What it can do

### Deployment planning
```
User: "I want to serve Llama-3.1-70B to 500 concurrent users with < 3s TTFT SLO."

Copilot: "For Llama-3.1-70B at FP8, you need ~80GB VRAM per replica.
          With 4× A100-80GB nodes (tensor parallel size 4), each replica
          handles ~20 concurrent requests. For 500 users with 3s TTFT SLO,
          you'll need:
          - Min replicas: 25 (500 / 20)
          - Recommended: 30 (20% headroom for burst)
          - HPA range: 20–40 replicas
          - PVC size: 160Gi (model weights + cache)
          - Estimated GPU cost: ~$18/hr on AWS (p4d.24xlarge)"
```

### GPU resource diagnosis
```
User: "My vLLM pod is getting OOM-killed after 10 minutes."

Copilot: "Most likely cause: KV cache overflow. Check:
          1. kubectl describe pod <pod> | grep -A5 'Last State'
             → Look for OOMKilled
          2. vllm:gpu_cache_usage_perc — if hitting 100%, preemptions
             cascade into full OOM
          Fix: reduce --gpu-memory-utilization from 0.95 to 0.85,
          or reduce --max-model-len to free KV cache headroom."
```

### Helm configuration generation
```
User: "Generate prod values for Mistral-7B on 2× A10G nodes."

Copilot: [generates complete prod/values.yaml with correct
          resource limits, nodeSelector, tolerations, HPA thresholds,
          PVC size, and tensor-parallel-size for this exact config]
```

---

## Architecture

```
User query (natural language)
        │
        ▼
┌───────────────────────────────┐
│  Query Router                 │
│  classify: deploy / debug /   │
│           scale / config      │
└──────────────┬────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌───────┐ ┌────────┐ ┌─────────┐
│Deploy │ │Debug   │ │Config   │
│Planner│ │Agent   │ │Generator│
└───┬───┘ └───┬────┘ └────┬────┘
    │         │            │
    └─────────┴────────────┘
               │
        ┌──────▼──────┐
        │  LLM Core   │  Reasoning engine
        │  + RAG over │  over infra docs,
        │  K8s docs   │  vLLM docs, GPU specs
        └─────────────┘
               │
        Structured output
        (YAML / kubectl commands / explanations)
```

---

## Knowledge base

The copilot is grounded in:

- vLLM documentation and source (model configs, serving args, metrics)
- Kubernetes API reference (HPA, PDB, NetworkPolicy, resource limits)
- NVIDIA GPU specs (A10G, A100, H100 — VRAM, bandwidth, CUDA cores)
- KubeInfer architecture patterns (from the companion project)
- Common failure modes from production LLM deployments

---

## Tech stack

```
Core LLM          OpenAI API / local vLLM endpoint
RAG               LangChain + FAISS vector store
Knowledge base    vLLM docs, K8s docs, GPU spec sheets (chunked + embedded)
Output parsing    Pydantic structured outputs → YAML / kubectl commands
Interface         CLI + optional web UI
Tests             pytest · Docker
```

---

## Quick start

```bash
git clone https://github.com/ArchanaChetan07/AI-Infrastructure-Copilot
cd AI-Infrastructure-Copilot
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
# or point at local vLLM:
export VLLM_BASE_URL=http://localhost:8000/v1

python copilot.py
```

---

## Example questions

```
"What's the minimum GPU setup to run Llama-3.1-405B?"
"Why is my P95 TTFT 30 seconds? How do I debug it?"
"Generate a Helm values file for staging — Mistral-7B, 1 GPU, no HPA"
"How do I set up KV cache prefix caching for a RAG chatbot?"
"What should my HPA minReplicas be for a 24/7 production API?"
"How do I safely drain a GPU node without dropping requests?"
```

---

## Part of the vLLM Observability Ecosystem

| Project | What it does |
|---------|-------------|
| **[AI Infrastructure Copilot](https://github.com/ArchanaChetan07/AI-Infrastructure-Copilot)** ← you are here | Conversational GPU capacity planning · Helm config generation · vLLM diagnosis |
| **[KubeInfer](https://github.com/ArchanaChetan07/KubeInfer)** | Production K8s deployment · queue-depth HPA · GitOps · 12 alert rules |
| **[AI Inference Observability Platform](https://github.com/ArchanaChetan07/ai-inference-observability-platform)** | FastAPI proxy · TTFT/TBT/E2E · Prometheus · Grafana · OpenTelemetry · 48 tests |
| **[KV Cache Profiler](https://github.com/ArchanaChetan07/KV-Cache-Profiler-)** | Real-time GPU KV cache hit rate · eviction · memory pressure |
| **[LLM Benchmarking Dashboard](https://github.com/ArchanaChetan07/LLM-Inference-Benchmarking-Dashboard)** | Live TTFT/TPOT/ITL/E2EL charts · DCGM GPU metrics |

---

## License

[MIT](LICENSE)

---

<div align="center">

**Archana Suresh Patil** — ML Platform & MLOps Engineer · Sunnyvale, CA  
[LinkedIn](https://linkedin.com/in/archana-suresh-patil-792213245) · [GitHub](https://github.com/ArchanaChetan07) · Open to full-time · No sponsorship needed

**[⭐ Star this repo](https://github.com/ArchanaChetan07/AI-Infrastructure-Copilot)** if it helps your LLM infrastructure decisions.

</div>
