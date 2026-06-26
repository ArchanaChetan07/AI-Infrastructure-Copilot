"""
Auth + Rate Limiting Middleware — Day 4

API Key Auth:
  Pass X-API-Key header with any request.
  Set API_KEY in .env to enable. Leave blank to run open (dev mode).

Rate Limiting:
  Uses in-memory sliding window counter per client IP.
  Default: 60 requests/minute (configurable via RATE_LIMIT_PER_MINUTE).
  /health and /docs are exempt from both auth and rate limiting.

In production: swap in-memory store for Redis via slowapi + redis backend.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Exempt paths — no auth, no rate limit
EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}

# In-memory sliding window: {client_ip: deque of timestamps}
_rate_windows: dict[str, deque] = defaultdict(deque)


class AuthRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Applies API key authentication and per-IP rate limiting to all
    non-exempt paths.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Exempt health/docs
        if path in EXEMPT_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        # ── Rate limiting ─────────────────────────────────────────────────────
        if settings.rate_limit_per_minute > 0:
            limit_result = self._check_rate_limit(client_ip)
            if limit_result is not None:
                return limit_result

        # ── API key auth ──────────────────────────────────────────────────────
        if settings.api_key:
            provided = request.headers.get("X-API-Key", "")
            if provided != settings.api_key:
                logger.warning(
                    f"Auth failure: invalid API key from {client_ip} for {request.method} {path}"
                )
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Unauthorized",
                        "detail": "Invalid or missing X-API-Key header",
                        "hint": "Set X-API-Key: <your_key> — see .env.example for API_KEY",
                    },
                )

        return await call_next(request)

    def _check_rate_limit(self, client_ip: str) -> JSONResponse | None:
        """
        Sliding window rate limiter.
        Returns a 429 JSONResponse if the limit is exceeded, else None.
        """
        now = time.time()
        window = _rate_windows[client_ip]
        cutoff = now - 60.0

        # Remove timestamps outside the window
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= settings.rate_limit_per_minute:
            retry_after = int(60 - (now - window[0])) + 1
            logger.warning(f"Rate limit exceeded: {client_ip} ({len(window)} req/min)")
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": "Too Many Requests",
                    "detail": f"Rate limit: {settings.rate_limit_per_minute} requests/minute",
                    "retry_after_seconds": retry_after,
                },
            )

        window.append(now)
        return None


def get_rate_limit_stats() -> dict:
    """Return current rate limit counters — useful for monitoring."""
    now = time.time()
    cutoff = now - 60.0
    return {
        client: len([t for t in window if t > cutoff])
        for client, window in _rate_windows.items()
    }
