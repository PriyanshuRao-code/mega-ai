"""
debug/run_api_debug.py
======================
Purpose     : Standalone debug harness that validates:
              1. All routes are registered and reachable
              2. Request schemas are enforced (422 on bad input)
              3. SSE streaming produces correctly-framed events

Imports     : asyncio, json, httpx, sys, pathlib, api.app, api.services, api.models
Inputs      : None (run directly: `python debug/run_api_debug.py`)
Outputs     : Colour-coded PASS / FAIL lines to stdout; exits 0 on all-pass, 1 otherwise
Dependencies:
  - httpx[asyncio]  (`pip install httpx`)
  - FastAPI / Starlette (app under test)
  - api.app.create_app
  - api.services (stub implementations below)
Exceptions  :
  - SystemExit(1) if any validation step fails
  - httpx.HTTPError on transport failures (prints and marks FAIL)
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

# Make sure the repo root is on sys.path when running from debug/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from httpx import AsyncClient, ASGITransport

from api.app import create_app
from api.models import (
    AgentStatus,
    EvalMetric,
    EvalSummaryResponse,
    ExecutionTraceResponse,
    QueryRequest,
    QueryResponse,
    ReEvalRequest,
    ReEvalResponse,
    RewriteDecision,
    RewriteDecisionRequest,
    RewriteDecisionResponse,
    SSEEvent,
    SSEEventType,
    TraceStep,
)
from api.services import (
    IEvalService,
    IQueryService,
    IRewriteService,
    ITraceService,
)

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"

_passed: list[str] = []
_failed: list[str] = []


def _pass(label: str) -> None:
    print(f"  {GREEN}✓ PASS{RESET}  {label}")
    _passed.append(label)


def _fail(label: str, reason: str) -> None:
    print(f"  {RED}✗ FAIL{RESET}  {label}")
    print(f"         {RED}{reason}{RESET}")
    _failed.append(label)


def _section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


# ---------------------------------------------------------------------------
# Stub service implementations for the harness
# ---------------------------------------------------------------------------

RUN_ID   = str(uuid.uuid4())
SESSION  = str(uuid.uuid4())

_TRACE_STEP = TraceStep(
    step_index=0,
    agent_name="planner",
    tool_name="search",
    input_summary="user query",
    output_summary="search results",
    latency_ms=120,
    tokens_used=512,
    timestamp=datetime.utcnow(),
)


class StubQueryService(IQueryService):
    async def submit(self, request: QueryRequest) -> QueryResponse:
        return QueryResponse(
            run_id=RUN_ID,
            session_id=SESSION,
            status=AgentStatus.COMPLETED,
            answer="Stub answer",
        )

    async def stream(self, request: QueryRequest) -> AsyncIterator[SSEEvent]:
        events = [
            SSEEvent(event=SSEEventType.ACTIVE_AGENT, run_id=RUN_ID, data={"agent": "planner"}),
            SSEEvent(event=SSEEventType.ACTIVE_TOOL,  run_id=RUN_ID, data={"tool": "search"}),
            SSEEvent(event=SSEEventType.TOKEN_STREAM, run_id=RUN_ID, data={"token": "Hello"}),
            SSEEvent(event=SSEEventType.TOKEN_STREAM, run_id=RUN_ID, data={"token": " world"}),
            SSEEvent(event=SSEEventType.CONTEXT_BUDGET, run_id=RUN_ID, data={"remaining": 3500}),
        ]
        for ev in events:
            yield ev


class StubTraceService(ITraceService):
    async def get_trace(self, run_id: str) -> ExecutionTraceResponse:
        if run_id != RUN_ID:
            from api.services import RunNotFoundError
            raise RunNotFoundError(f"run_id {run_id!r} not found")
        return ExecutionTraceResponse(
            run_id=run_id,
            session_id=SESSION,
            status=AgentStatus.COMPLETED,
            steps=[_TRACE_STEP],
            total_tokens=512,
            total_latency_ms=120,
        )


class StubEvalService(IEvalService):
    async def get_latest_eval(self, run_id: str) -> EvalSummaryResponse:
        if run_id != RUN_ID:
            from api.services import RunNotFoundError
            raise RunNotFoundError(f"run_id {run_id!r} not found")
        return EvalSummaryResponse(
            run_id=run_id,
            evaluated_at=datetime.utcnow(),
            overall_score=0.87,
            metrics=[EvalMetric(name="faithfulness", score=0.87)],
            passed=True,
        )

    async def reeval(self, run_id: str, request: ReEvalRequest) -> ReEvalResponse:
        if run_id != RUN_ID:
            from api.services import RunNotFoundError
            raise RunNotFoundError(f"run_id {run_id!r} not found")
        return ReEvalResponse(
            run_id=run_id,
            metric_names=request.metric_names,
        )


class StubRewriteService(IRewriteService):
    async def decide(
        self, run_id: str, request: RewriteDecisionRequest
    ) -> RewriteDecisionResponse:
        if run_id != RUN_ID:
            from api.services import RunNotFoundError
            raise RunNotFoundError(f"run_id {run_id!r} not found")
        return RewriteDecisionResponse(
            run_id=run_id,
            decision=request.decision,
            reviewer_id=request.reviewer_id,
        )


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _make_app() -> "FastAPI":  # noqa: F821
    return create_app(
        query_service_factory=StubQueryService,
        trace_service_factory=StubTraceService,
        eval_service_factory=StubEvalService,
        rewrite_service_factory=StubRewriteService,
    )


async def validate_routes(client: AsyncClient) -> None:
    _section("1 · Route registration")

    routes_to_check = [
        ("POST", "/v1/query"),
        ("GET",  f"/v1/runs/{RUN_ID}/trace"),
        ("GET",  f"/v1/runs/{RUN_ID}/eval"),
        ("POST", f"/v1/runs/{RUN_ID}/rewrite"),
        ("POST", f"/v1/runs/{RUN_ID}/reeval"),
    ]

    for method, path in routes_to_check:
        # A 405 means the route exists but method is wrong; 404 means missing.
        # We send a HEAD to avoid needing a body for POST routes here — we just
        # want to confirm the path is registered.
        r = await client.request("OPTIONS", path)
        label = f"{method} {path}"
        if r.status_code != 404:
            _pass(f"route registered: {label}")
        else:
            _fail(f"route registered: {label}", f"got 404 — route not found")


async def validate_request_schemas(client: AsyncClient) -> None:
    _section("2 · Request schema validation")

    # ── 2a. Submit query — valid payload ──────────────────────────────────
    label = "POST /v1/query — valid payload → 202"
    try:
        r = await client.post("/v1/query", json={"query": "What is 2+2?"})
        if r.status_code == 202:
            data = r.json()
            QueryResponse(**data)   # validates response shape
            _pass(label)
        else:
            _fail(label, f"expected 202, got {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        _fail(label, str(exc))

    # ── 2b. Submit query — missing required field ─────────────────────────
    label = "POST /v1/query — missing 'query' field → 422"
    try:
        r = await client.post("/v1/query", json={"stream": False})
        if r.status_code == 422:
            _pass(label)
        else:
            _fail(label, f"expected 422, got {r.status_code}")
    except Exception as exc:
        _fail(label, str(exc))

    # ── 2c. Submit query — empty string ───────────────────────────────────
    label = "POST /v1/query — empty query string → 422"
    try:
        r = await client.post("/v1/query", json={"query": ""})
        if r.status_code == 422:
            _pass(label)
        else:
            _fail(label, f"expected 422, got {r.status_code}")
    except Exception as exc:
        _fail(label, str(exc))

    # ── 2d. Execution trace — valid run_id ────────────────────────────────
    label = f"GET /v1/runs/{{run_id}}/trace — valid → 200"
    try:
        r = await client.get(f"/v1/runs/{RUN_ID}/trace")
        if r.status_code == 200:
            ExecutionTraceResponse(**r.json())
            _pass(label)
        else:
            _fail(label, f"expected 200, got {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        _fail(label, str(exc))

    # ── 2e. Execution trace — unknown run_id ──────────────────────────────
    label = "GET /v1/runs/{unknown}/trace — unknown → 404"
    try:
        r = await client.get(f"/v1/runs/nonexistent-id/trace")
        if r.status_code == 404:
            _pass(label)
        else:
            _fail(label, f"expected 404, got {r.status_code}")
    except Exception as exc:
        _fail(label, str(exc))

    # ── 2f. Eval summary ──────────────────────────────────────────────────
    label = "GET /v1/runs/{run_id}/eval — valid → 200"
    try:
        r = await client.get(f"/v1/runs/{RUN_ID}/eval")
        if r.status_code == 200:
            EvalSummaryResponse(**r.json())
            _pass(label)
        else:
            _fail(label, f"expected 200, got {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        _fail(label, str(exc))

    # ── 2g. Approve rewrite ───────────────────────────────────────────────
    label = "POST /v1/runs/{run_id}/rewrite — approve → 200"
    try:
        r = await client.post(
            f"/v1/runs/{RUN_ID}/rewrite",
            json={"decision": "approve", "reviewer_id": "debug-user"},
        )
        if r.status_code == 200:
            RewriteDecisionResponse(**r.json())
            _pass(label)
        else:
            _fail(label, f"expected 200, got {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        _fail(label, str(exc))

    # ── 2h. Rewrite — invalid decision value ─────────────────────────────
    label = "POST /v1/runs/{run_id}/rewrite — bad decision enum → 422"
    try:
        r = await client.post(
            f"/v1/runs/{RUN_ID}/rewrite",
            json={"decision": "maybe", "reviewer_id": "debug-user"},
        )
        if r.status_code == 422:
            _pass(label)
        else:
            _fail(label, f"expected 422, got {r.status_code}")
    except Exception as exc:
        _fail(label, str(exc))

    # ── 2i. Targeted re-eval ──────────────────────────────────────────────
    label = "POST /v1/runs/{run_id}/reeval — valid → 202"
    try:
        r = await client.post(
            f"/v1/runs/{RUN_ID}/reeval",
            json={"metric_names": ["faithfulness"]},
        )
        if r.status_code == 202:
            ReEvalResponse(**r.json())
            _pass(label)
        else:
            _fail(label, f"expected 202, got {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        _fail(label, str(exc))

    # ── 2j. Targeted re-eval — empty metric list ─────────────────────────
    label = "POST /v1/runs/{run_id}/reeval — empty metrics → 422"
    try:
        r = await client.post(
            f"/v1/runs/{RUN_ID}/reeval",
            json={"metric_names": []},
        )
        if r.status_code == 422:
            _pass(label)
        else:
            _fail(label, f"expected 422, got {r.status_code}")
    except Exception as exc:
        _fail(label, str(exc))


async def validate_streaming(client: AsyncClient) -> None:
    _section("3 · SSE streaming")

    label = "POST /v1/query?stream=true — receives all SSEEventType frames"
    try:
        seen_events: set[str] = set()
        required_events = {
            SSEEventType.ACTIVE_AGENT.value,
            SSEEventType.ACTIVE_TOOL.value,
            SSEEventType.TOKEN_STREAM.value,
            SSEEventType.CONTEXT_BUDGET.value,
            SSEEventType.DONE.value,
        }

        async with client.stream(
            "POST",
            "/v1/query",
            json={"query": "stream test", "stream": True},
        ) as response:
            if response.status_code != 200:
                _fail(label, f"expected 200, got {response.status_code}")
                return

            ct = response.headers.get("content-type", "")
            if "text/event-stream" not in ct:
                _fail(label, f"expected text/event-stream, got {ct!r}")
                return

            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    seen_events.add(line.split(":", 1)[1].strip())

        missing = required_events - seen_events
        if missing:
            _fail(label, f"missing event types: {missing}")
        else:
            _pass(label)

    except Exception as exc:
        _fail(label, str(exc))

    # ── SSE frame format check ────────────────────────────────────────────
    label = "SSE frames contain event:, data:, id: fields"
    try:
        frame_fields: set[str] = set()
        async with client.stream(
            "POST",
            "/v1/query",
            json={"query": "frame format test", "stream": True},
        ) as response:
            async for line in response.aiter_lines():
                for prefix in ("event:", "data:", "id:"):
                    if line.startswith(prefix):
                        frame_fields.add(prefix)

        required_fields = {"event:", "data:", "id:"}
        missing = required_fields - frame_fields
        if missing:
            _fail(label, f"missing SSE fields: {missing}")
        else:
            _pass(label)

    except Exception as exc:
        _fail(label, str(exc))

    # ── data: field is valid JSON ─────────────────────────────────────────
    label = "SSE data: fields are valid JSON"
    try:
        bad_lines: list[str] = []
        async with client.stream(
            "POST",
            "/v1/query",
            json={"query": "json check", "stream": True},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    if raw:
                        try:
                            json.loads(raw)
                        except json.JSONDecodeError:
                            bad_lines.append(raw[:80])

        if bad_lines:
            _fail(label, f"invalid JSON in data fields: {bad_lines[:3]}")
        else:
            _pass(label)

    except Exception as exc:
        _fail(label, str(exc))


async def _run_all() -> None:
    app = _make_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await validate_routes(client)
        await validate_request_schemas(client)
        await validate_streaming(client)

    # ── Summary ──────────────────────────────────────────────────────────
    total = len(_passed) + len(_failed)
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(
        f"{BOLD}  Results: {GREEN}{len(_passed)} passed{RESET}{BOLD}, "
        f"{RED}{len(_failed)} failed{RESET}{BOLD} / {total} total{RESET}"
    )
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    if _failed:
        print(f"{RED}Failed checks:{RESET}")
        for name in _failed:
            print(f"  • {name}")
        sys.exit(1)
    else:
        print(f"{GREEN}All checks passed.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(_run_all())
