"""
api/error_handlers.py
=====================
Purpose     : Maps domain exceptions and FastAPI/Pydantic errors to consistent
              JSON error envelopes using the ErrorResponse contract.
Imports     : logging, fastapi, pydantic, starlette, api.models, api.services
Inputs      : Request + exception (standard FastAPI exception handler signature)
Outputs     : JSONResponse with ErrorResponse body
Dependencies: fastapi, starlette, api.models.ErrorResponse, api.services.*Error
Exceptions  :
  - This module HANDLES exceptions; it does not raise new ones.
    If the serialisation itself fails, FastAPI's default 500 handler takes over.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.models import ErrorDetail, ErrorResponse
from api.services import (
    EvalNotFoundError,
    RewriteConflictError,
    RunNotFoundError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

def _json_error(
    request: Request,
    http_status: int,
    details: list[ErrorDetail],
) -> JSONResponse:
    """
    Build a JSONResponse from a list of ErrorDetail objects.

    Input : request (for trace_id), http_status (int), details (list)
    Output: JSONResponse
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    body = ErrorResponse(status=http_status, errors=details, trace_id=trace_id)
    return JSONResponse(
        status_code=http_status,
        content=body.model_dump(mode="json", exclude_none=True),
    )


# ---------------------------------------------------------------------------
# Individual handlers
# ---------------------------------------------------------------------------

async def handle_run_not_found(request: Request, exc: RunNotFoundError) -> JSONResponse:
    """
    Input : RunNotFoundError
    Output: 404 JSONResponse
    """
    logger.warning("RunNotFoundError: %s", exc)
    return _json_error(
        request,
        status.HTTP_404_NOT_FOUND,
        [ErrorDetail(code="RUN_NOT_FOUND", message=str(exc))],
    )


async def handle_eval_not_found(request: Request, exc: EvalNotFoundError) -> JSONResponse:
    """
    Input : EvalNotFoundError
    Output: 404 JSONResponse
    """
    logger.warning("EvalNotFoundError: %s", exc)
    return _json_error(
        request,
        status.HTTP_404_NOT_FOUND,
        [ErrorDetail(code="EVAL_NOT_FOUND", message=str(exc))],
    )


async def handle_rewrite_conflict(request: Request, exc: RewriteConflictError) -> JSONResponse:
    """
    Input : RewriteConflictError
    Output: 409 JSONResponse
    """
    logger.warning("RewriteConflictError: %s", exc)
    return _json_error(
        request,
        status.HTTP_409_CONFLICT,
        [ErrorDetail(code="REWRITE_CONFLICT", message=str(exc))],
    )


async def handle_service_unavailable(
    request: Request, exc: ServiceUnavailableError
) -> JSONResponse:
    """
    Input : ServiceUnavailableError
    Output: 503 JSONResponse
    """
    logger.error("ServiceUnavailableError: %s", exc)
    return _json_error(
        request,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        [ErrorDetail(code="SERVICE_UNAVAILABLE", message=str(exc))],
    )


async def handle_request_validation(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handles FastAPI/Pydantic validation errors from request parsing.

    Input : RequestValidationError (list of Pydantic error dicts)
    Output: 422 JSONResponse
    """
    details = [
        ErrorDetail(
            code="VALIDATION_ERROR",
            message=err.get("msg", "Invalid value"),
            field=".".join(str(loc) for loc in err.get("loc", [])),
        )
        for err in exc.errors()
    ]
    return _json_error(request, status.HTTP_422_UNPROCESSABLE_ENTITY, details)


async def handle_pydantic_validation(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Catches Pydantic ValidationError raised inside route logic (not at parse time).

    Input : ValidationError
    Output: 422 JSONResponse
    """
    details = [
        ErrorDetail(
            code="VALIDATION_ERROR",
            message=err.get("msg", "Invalid value"),
            field=".".join(str(loc) for loc in err.get("loc", [])),
        )
        for err in exc.errors()
    ]
    return _json_error(request, status.HTTP_422_UNPROCESSABLE_ENTITY, details)


async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for any exception not handled above.

    Input : Exception
    Output: 500 JSONResponse
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return _json_error(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        [ErrorDetail(code="INTERNAL_ERROR", message="An unexpected error occurred")],
    )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_error_handlers(app: FastAPI) -> None:
    """
    Attach all exception handlers to the FastAPI application.

    Input : FastAPI app instance
    Output: None (mutates app in-place)
    """
    app.add_exception_handler(RunNotFoundError, handle_run_not_found)           # type: ignore[arg-type]
    app.add_exception_handler(EvalNotFoundError, handle_eval_not_found)         # type: ignore[arg-type]
    app.add_exception_handler(RewriteConflictError, handle_rewrite_conflict)    # type: ignore[arg-type]
    app.add_exception_handler(ServiceUnavailableError, handle_service_unavailable)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, handle_request_validation)    # type: ignore[arg-type]
    app.add_exception_handler(ValidationError, handle_pydantic_validation)      # type: ignore[arg-type]
    app.add_exception_handler(Exception, handle_unhandled_exception)            # type: ignore[arg-type]
