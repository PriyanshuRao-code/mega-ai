"""
api/routes.py
=============
Purpose     : FastAPI APIRouter containing all five required endpoints.
              Route handlers are thin: they validate input, delegate to injected
              service interfaces, and return typed response models.
Imports     : fastapi, api.models.*, api.services.*, api.sse
Inputs      : (per endpoint — see individual docstrings)
Outputs     : Pydantic response models or StreamingResponse (SSE)
Dependencies:
  - api.models  — request/response contracts
  - api.services — IQueryService, ITraceService, IEvalService, IRewriteService
  - api.sse     — build_sse_response
Exceptions  :
  - RunNotFoundError      → 404 (handled by error_handlers)
  - EvalNotFoundError     → 404
  - RewriteConflictError  → 409
  - ServiceUnavailableError → 503
  - RequestValidationError → 422 (auto by FastAPI)
"""

from __future__ import annotations

import uuid
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, status
from fastapi.responses import StreamingResponse

from api.models import (
    EvalSummaryResponse,
    ExecutionTraceResponse,
    QueryRequest,
    QueryResponse,
    ReEvalRequest,
    ReEvalResponse,
    RewriteDecisionRequest,
    RewriteDecisionResponse,
)
from api.services import IEvalService, IQueryService, IRewriteService, ITraceService
from api.sse import build_sse_response

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Concrete Implementations & Wiring
# ---------------------------------------------------------------------------
from api.query_service_impl import QueryService
from api.trace_service_impl import TraceService
from orchestrator.orchestrator import build_orchestrator
from agents.decomposition_agent import DecompositionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.critique_agent import CritiqueAgent
from agents.synthesis_agent import SynthesisAgent

# 1. Initialize the Orchestrator (Long-lived Singleton)
orchestrator_instance = build_orchestrator(
    agents=[
        DecompositionAgent(),
        RetrievalAgent(),
        CritiqueAgent(),
        SynthesisAgent()
    ]
)

trace_service_instance = TraceService()

# 2. Dependency Injection Providers
def get_query_service() -> IQueryService:
    """Returns the concrete QueryService implementation."""
    return QueryService(orchestrator=orchestrator_instance)


def get_trace_service() -> ITraceService:
    return trace_service_instance


def get_eval_service() -> IEvalService:
    raise NotImplementedError("Wire a concrete IEvalService via dependency injection")


def get_rewrite_service() -> IRewriteService:
    raise NotImplementedError("Wire a concrete IRewriteService via dependency injection")


# ---------------------------------------------------------------------------
# Type aliases for injected dependencies (DRY)
# ---------------------------------------------------------------------------

QueryServiceDep  = Annotated[IQueryService,   Depends(get_query_service)]
TraceServiceDep  = Annotated[ITraceService,   Depends(get_trace_service)]
EvalServiceDep   = Annotated[IEvalService,    Depends(get_eval_service)]
RewriteServiceDep = Annotated[IRewriteService, Depends(get_rewrite_service)]


# ---------------------------------------------------------------------------
# 1. Submit Query
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a query to the multi-agent system",
    tags=["execution"],
)
async def submit_query(
    body: QueryRequest,
    http_request: Request,
    query_service: QueryServiceDep,
) -> QueryResponse | StreamingResponse:
    """
    Input  : QueryRequest (JSON body)
    Output :
      - QueryResponse (202) when body.stream is False
      - StreamingResponse / SSE when body.stream is True
    Raises :
      - ServiceUnavailableError → 503
    """
    logger.info("submit_query stream=%s", body.stream)

    if body.stream:
        # Allocate a run_id eagerly so the SSE stream can reference it.
        run_id = str(uuid.uuid4())
        return build_sse_response(
            run_id=run_id,
            query_service=query_service,
            request=body,
            http_request=http_request,
        )

    return await query_service.submit(body)


# ---------------------------------------------------------------------------
# 2. Execution Trace
# ---------------------------------------------------------------------------

@router.get(
    "/runs/{run_id}/trace",
    response_model=ExecutionTraceResponse,
    summary="Retrieve full execution trace for a run",
    tags=["observability"],
)
async def get_execution_trace(
    run_id: Annotated[str, Path(description="UUID of the run")],
    trace_service: TraceServiceDep,
) -> ExecutionTraceResponse:
    """
    Input  : run_id (path parameter)
    Output : ExecutionTraceResponse (200)
    Raises :
      - RunNotFoundError → 404
    """
    logger.info("get_execution_trace run_id=%s", run_id)
    return await trace_service.get_trace(run_id)


# ---------------------------------------------------------------------------
# 3. Latest Eval Summary
# ---------------------------------------------------------------------------

@router.get(
    "/runs/{run_id}/eval",
    response_model=EvalSummaryResponse,
    summary="Retrieve the latest evaluation summary for a run",
    tags=["evaluation"],
)
async def get_eval_summary(
    run_id: Annotated[str, Path(description="UUID of the run")],
    eval_service: EvalServiceDep,
) -> EvalSummaryResponse:
    """
    Input  : run_id (path parameter)
    Output : EvalSummaryResponse (200)
    Raises :
      - RunNotFoundError  → 404
      - EvalNotFoundError → 404
    """
    logger.info("get_eval_summary run_id=%s", run_id)
    return await eval_service.get_latest_eval(run_id)


# ---------------------------------------------------------------------------
# 4. Approve / Reject Rewrite
# ---------------------------------------------------------------------------

@router.post(
    "/runs/{run_id}/rewrite",
    response_model=RewriteDecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve or reject a proposed rewrite for a run",
    tags=["review"],
)
async def decide_rewrite(
    run_id: Annotated[str, Path(description="UUID of the run")],
    body: RewriteDecisionRequest,
    rewrite_service: RewriteServiceDep,
) -> RewriteDecisionResponse:
    """
    Input  : run_id (path), RewriteDecisionRequest (JSON body)
    Output : RewriteDecisionResponse (200)
    Raises :
      - RunNotFoundError     → 404
      - RewriteConflictError → 409
    """
    logger.info(
        "decide_rewrite run_id=%s decision=%s reviewer=%s",
        run_id,
        body.decision,
        body.reviewer_id,
    )
    return await rewrite_service.decide(run_id, body)


# ---------------------------------------------------------------------------
# 5. Targeted Re-evaluation
# ---------------------------------------------------------------------------

@router.post(
    "/runs/{run_id}/reeval",
    response_model=ReEvalResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a targeted re-evaluation for specific metrics",
    tags=["evaluation"],
)
async def targeted_reeval(
    run_id: Annotated[str, Path(description="UUID of the run")],
    body: ReEvalRequest,
    eval_service: EvalServiceDep,
) -> ReEvalResponse:
    """
    Input  : run_id (path), ReEvalRequest (JSON body)
    Output : ReEvalResponse (202)
    Raises :
      - RunNotFoundError → 404
    """
    logger.info(
        "targeted_reeval run_id=%s metrics=%s",
        run_id,
        body.metric_names,
    )
    return await eval_service.reeval(run_id, body)
