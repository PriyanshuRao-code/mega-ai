"""
api/sse.py
==========
Purpose     : SSE serialisation helpers and the streaming endpoint generator.
              Keeps all SSE concerns in one place (SRP).
Imports     : asyncio, json, typing, fastapi, api.models, api.services
Inputs      : run_id (str), IQueryService instance, QueryRequest
Outputs     : EventSourceResponse / async generator of raw SSE text frames
Dependencies: fastapi, sse-starlette, api.models.SSEEvent, api.services.IQueryService
Exceptions  :
  - asyncio.CancelledError  — client disconnect; generator exits cleanly
  - ServiceUnavailableError — propagated as SSEEventType.ERROR frame then stream ends
  - Any unhandled exception — emitted as SSEEventType.ERROR frame then re-raised
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import Request
from fastapi.responses import StreamingResponse

from api.models import SSEEvent, SSEEventType
from api.services import IQueryService, QueryRequest, ServiceUnavailableError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialise_event(event: SSEEvent) -> str:
    """
    Convert an SSEEvent into the wire format required by the EventSource spec:

        event: <type>
        data: <json>
        id: <run_id>

    Input : SSEEvent
    Output: str  (ready to send over HTTP)
    """
    payload = json.dumps(event.model_dump(mode="json"), default=str)
    lines = [
        f"event: {event.event.value}",
        f"data: {payload}",
        f"id: {event.run_id}",
        "",  # blank line = end of event
        "",
    ]
    return "\n".join(lines)


def _error_frame(run_id: str, message: str) -> str:
    """
    Build a terminal ERROR SSE frame.

    Input : run_id (str), message (str)
    Output: str
    """
    error_event = SSEEvent(
        event=SSEEventType.ERROR,
        run_id=run_id,
        data={"message": message},
    )
    return _serialise_event(error_event)


def _done_frame(run_id: str) -> str:
    """
    Build the terminal DONE SSE frame.

    Input : run_id (str)
    Output: str
    """
    done_event = SSEEvent(
        event=SSEEventType.DONE,
        run_id=run_id,
        data={},
    )
    return _serialise_event(done_event)


# ---------------------------------------------------------------------------
# Public streaming generator
# ---------------------------------------------------------------------------

async def event_generator(
    run_id: str,
    query_service: IQueryService,
    request: QueryRequest,
    http_request: Request,
) -> AsyncIterator[str]:
    """
    Async generator that drives the SSE stream for a single run.

    Input:
      run_id        — identifier for the run being streamed
      query_service — IQueryService implementation
      request       — original QueryRequest
      http_request  — FastAPI Request (used to detect client disconnection)

    Output:
      Yields raw SSE-formatted strings.

    Raises:
      asyncio.CancelledError — swallowed; loop exits and logs at DEBUG level
      ServiceUnavailableError — emits ERROR frame; generator returns
      Exception              — emits ERROR frame; exception is re-raised
    """
    logger.info("SSE stream started for run_id=%s", run_id)
    try:
        async for sse_event in query_service.stream(request):
            # Respect client disconnects between yields
            if await http_request.is_disconnected():
                logger.debug("Client disconnected; stopping SSE for run_id=%s", run_id)
                return

            yield _serialise_event(sse_event)
            await asyncio.sleep(0)  # yield control to the event loop

        yield _done_frame(run_id)
        logger.info("SSE stream completed for run_id=%s", run_id)

    except asyncio.CancelledError:
        logger.debug("SSE generator cancelled for run_id=%s", run_id)

    except ServiceUnavailableError as exc:
        logger.warning("ServiceUnavailableError during SSE for run_id=%s: %s", run_id, exc)
        yield _error_frame(run_id, str(exc))

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during SSE for run_id=%s", run_id)
        yield _error_frame(run_id, "Internal streaming error")
        raise


# ---------------------------------------------------------------------------
# Public factory — builds a StreamingResponse for use in route handlers
# ---------------------------------------------------------------------------

def build_sse_response(
    run_id: str,
    query_service: IQueryService,
    request: QueryRequest,
    http_request: Request,
) -> StreamingResponse:
    """
    Factory that wires event_generator into a StreamingResponse with the
    correct Content-Type for SSE.

    Input:
      run_id        — str
      query_service — IQueryService
      request       — QueryRequest
      http_request  — FastAPI Request

    Output:
      StreamingResponse with media_type="text/event-stream"

    Raises: delegates to event_generator
    """
    generator = event_generator(run_id, query_service, request, http_request)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
