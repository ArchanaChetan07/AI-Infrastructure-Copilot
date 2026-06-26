"""
Central configuration — reads from environment variables with sensible defaults.
All Day 2 settings are included and gated by feature flags.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── Qdrant (Day 2 — RAG over historical incidents) ───────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "gpu_incident_logs"
    qdrant_enabled: bool = False          # set True when Qdrant is running
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 3                    # how many similar incidents to retrieve

    # ── PostgreSQL (Day 2 — incident history) ────────────────────────────────
    postgres_url: str = "postgresql+asyncpg://copilot:copilot@localhost:5432/gpu_copilot"
    postgres_enabled: bool = False        # set True when Postgres is running

    # ── Slack (Day 2 — notifications) ────────────────────────────────────────
    slack_webhook_url: str = ""
    slack_channel: str = "#gpu-alerts"
    slack_enabled: bool = False           # set True when webhook is configured

    # ── Mock / fixture mode ───────────────────────────────────────────────────
    fixtures_dir: str = "fixtures"
    use_mock_data: bool = True

    # ── Prometheus (live mode) ────────────────────────────────────────────────
    prometheus_url: str = "http://localhost:9090"
    prometheus_enabled: bool = False

    # ── Kubernetes (live mode) ────────────────────────────────────────────────
    kubeconfig_path: str = ""
    k8s_enabled: bool = False

    # ── Auto-remediation (Day 3) ──────────────────────────────────────────────
    remediation_default_mode: str = "dry_run"
    remediation_auto_severity: str = "critical"

    # ── Analytics (Day 3) ────────────────────────────────────────────────────
    analytics_enabled: bool = True


settings = Settings()
