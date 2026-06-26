"""
Structured Logging Middleware — Day 4
Injects a unique request_id into every request context and logs
structured JSON with method, path, status, latency, and request_id.

Every log line emitted during a request includes the request_id,
making distributed tracing trivial across Prometheus, Loki, or Datadog.
"""

from __future__ import annotations

import json
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import get_logger

logger = get_logger(__name__)

# Context var — set per-request so any log call picks it up
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns a unique request_id to every incoming request.
    Clients can pass X-Request-ID header to propagate their own ID.
    The ID is echoed back in the response header.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        token = _request_id_var.set(req_id)
        request.state.request_id = req_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            _log_request(request, 500, elapsed, req_id)
            raise
        finally:
            _request_id_var.reset(token)

        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
        _log_request(request, response.status_code, elapsed, req_id)
        return response


def _log_request(request: Request, status: int, elapsed_ms: float, req_id: str) -> None:
    record = {
        "request_id": req_id,
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "latency_ms": round(elapsed_ms, 1),
        "client": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "")[:60],
    }
    level = "WARNING" if status >= 400 else "INFO"
    if level == "WARNING":
        logger.warning(json.dumps(record))
    else:
        logger.info(json.dumps(record))
