"""
api/middleware.py
=================
Purpose     : ASGI middleware stack for the multi-agent FastAPI application.
              Covers request tracing, structured access logging, and timing.
Imports     : time, uuid, logging, starlette, fastapi
Inputs      : ASGI scope / receive / send (standard Starlette middleware interface)
Outputs     : Mutates request state (adds trace_id, start_time); emits log records
Dependencies: starlette (bundled with FastAPI), Python stdlib only
Exceptions  :
  - Any exception from downstream middleware / route is allowed to propagate
    so that error_handlers.py can catch and format it correctly.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Request-ID / Trace-ID injection
# ---------------------------------------------------------------------------

class TraceIDMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique trace_id to every request.

    Input : HTTP request (reads X-Request-ID header if present)
    Output: request.state.trace_id (str); X-Trace-ID response header
    """

    HEADER_IN = "X-Request-ID"
    HEADER_OUT = "X-Trace-ID"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id: str = request.headers.get(self.HEADER_IN) or str(uuid.uuid4())
        request.state.trace_id = trace_id

        response = await call_next(request)
        response.headers[self.HEADER_OUT] = trace_id
        return response


# ---------------------------------------------------------------------------
# 2. Structured access logging + latency
# ---------------------------------------------------------------------------

class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with method, path, status, and elapsed time.

    Input : HTTP request / response
    Output: Structured log record at INFO level
    Raises: Re-raises any exception from downstream after logging
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        trace_id = getattr(request.state, "trace_id", "-")

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.error(
                "request_error trace_id=%s method=%s path=%s elapsed_ms=%d",
                trace_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "request_complete trace_id=%s method=%s path=%s status=%d elapsed_ms=%d",
            trace_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
        return response


# ---------------------------------------------------------------------------
# Registration helper (called from app factory)
# ---------------------------------------------------------------------------

def register_middleware(app) -> None:  # type: ignore[type-arg]
    """
    Attach all middleware to the FastAPI app in the correct order.
    Starlette applies middleware in reverse-registration order (last added = outermost).

    Input : FastAPI application instance
    Output: None (mutates app in-place)
    """
    # Outermost layer — runs first on ingress, last on egress
    app.add_middleware(AccessLogMiddleware)
    # Inner layer — trace ID must be available before access log reads it
    app.add_middleware(TraceIDMiddleware)
