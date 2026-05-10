"""
api/app.py
==========
Purpose     : FastAPI application factory.
              Composes middleware, error handlers, and routes without coupling
              concrete service implementations to the API layer.
Imports     : fastapi, api.middleware, api.error_handlers, api.routes
Inputs      : Optional service overrides (for testing / DI)
Outputs     : FastAPI application instance
Dependencies: api.middleware.register_middleware
              api.error_handlers.register_error_handlers
              api.routes.router
Exceptions  : None raised by the factory itself.
"""

from __future__ import annotations

from typing import Callable

from fastapi import FastAPI

from api.error_handlers import register_error_handlers
from api.middleware import register_middleware
from api.routes import (
    router,
    get_eval_service,
    get_query_service,
    get_rewrite_service,
    get_trace_service,
)
from api.services import IEvalService, IQueryService, IRewriteService, ITraceService


def create_app(
    *,
    query_service_factory: Callable[[], IQueryService] | None = None,
    trace_service_factory: Callable[[], ITraceService] | None = None,
    eval_service_factory: Callable[[], IEvalService] | None = None,
    rewrite_service_factory: Callable[[], IRewriteService] | None = None,
    title: str = "Multi-Agent System API",
    version: str = "1.0.0",
) -> FastAPI:
    """
    Input:
      *_service_factory — optional zero-arg callables that return concrete
                          service implementations; used for DI / testing.
    Output:
      Configured FastAPI instance.
    """
    app = FastAPI(
        title=title,
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Middleware (outermost to innermost) --------------------------------
    register_middleware(app)

    # --- Error handlers -----------------------------------------------------
    register_error_handlers(app)

    # --- Dependency overrides -----------------------------------------------
    if query_service_factory:
        app.dependency_overrides[get_query_service] = query_service_factory
    if trace_service_factory:
        app.dependency_overrides[get_trace_service] = trace_service_factory
    if eval_service_factory:
        app.dependency_overrides[get_eval_service] = eval_service_factory
    if rewrite_service_factory:
        app.dependency_overrides[get_rewrite_service] = rewrite_service_factory

    # --- Routes -------------------------------------------------------------
    app.include_router(router, prefix="/v1")

    return app
