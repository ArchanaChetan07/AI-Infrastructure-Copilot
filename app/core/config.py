"""
Central configuration — reads from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # Qdrant (Day 2)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "gpu_incident_logs"

    # PostgreSQL (Day 2)
    postgres_url: str = "postgresql://copilot:copilot@localhost:5432/gpu_copilot"

    # Slack (Day 2)
    slack_webhook_url: str = ""
    slack_channel: str = "#gpu-alerts"

    # Fixtures path (Day 1 — used instead of live infra)
    fixtures_dir: str = "fixtures"
    use_mock_data: bool = True

    # Prometheus (live mode, Day 2+)
    prometheus_url: str = "http://localhost:9090"

    # Kubernetes (live mode, Day 2+)
    kubeconfig_path: str = ""


settings = Settings()
