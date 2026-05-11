"""
api/trace_service_impl.py
=========================
Purpose     : In-memory implementation of ITraceService.
              Provides a global registry for execution traces.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from api.models import ExecutionTraceResponse, TraceStep, AgentStatus
from contracts.models import ExecutionEvent, ExecutionStatus, EventType
from api.services import ITraceService, RunNotFoundError

logger = logging.getLogger(__name__)

# Simple in-memory storage (singleton-like for this process)
_TRACES: Dict[str, ExecutionTraceResponse] = {
    "123": ExecutionTraceResponse(
        run_id="123",
        session_id="sess_example",
        status=AgentStatus.COMPLETED,
        steps=[
            TraceStep(
                step_index=1,
                agent_name="DecompositionAgent",
                status="completed",
                input_summary="Example query",
                output_summary="Decomposed into 2 subtasks",
                latency_ms=150,
                tokens_used=45,
                timestamp=datetime.utcnow()
            )
        ],
        total_tokens=45,
        total_latency_ms=150
    )
}

class TraceService(ITraceService):
    async def get_trace(self, run_id: str) -> ExecutionTraceResponse:
        logger.info(f"Fetching trace for run_id: {run_id}")
        if run_id not in _TRACES:
            logger.warning(f"Trace not found for run_id: {run_id}")
            raise RunNotFoundError(f"Run {run_id} not found")
        return _TRACES[run_id]

def record_execution_result(run_id: str, context, execution_event: ExecutionEvent):
    """
    Converts Orchestrator results into an API-compatible ExecutionTraceResponse
    and stores it in the in-memory registry.
    """
    steps: List[TraceStep] = []
    
    # Filter for agent start events to build the timeline
    agent_executions = [
        e for e in execution_event.agent_events 
        if e.event_type == EventType.AGENT_STARTED
    ]
    
    for i, start_evt in enumerate(agent_executions):
        agent_name = start_evt.agent_name
        
        # Find the corresponding end or fail event for THIS specific execution
        # (Looking for the next event of type COMPLETED/FAILED for this agent)
        end_evt = next((
            e for e in execution_event.agent_events 
            if e.agent_name == agent_name 
            and e.event_type in {EventType.AGENT_COMPLETED, EventType.AGENT_FAILED}
            and e.timestamp >= start_evt.timestamp
        ), None)
        
        latency_ms = 0
        if end_evt:
            latency_ms = int((end_evt.timestamp - start_evt.timestamp).total_seconds() * 1000)
            
        # Get the output from context if available
        output_obj = context.get_agent_output(agent_name) # Uses unwrap helper
        # Since get_agent_output returns the UNWRAPPED result, we might need to get the ToolResponse
        # for tokens_used.
        raw_response = context.agent_outputs.get(agent_name)
        tokens = getattr(raw_response, "tokens_used", 0) if raw_response else 0
        
        steps.append(TraceStep(
            step_index=i + 1,
            agent_name=agent_name,
            status="completed" if end_evt and end_evt.event_type == EventType.AGENT_COMPLETED else "failed",
            input_summary=str(context.query)[:500],
            output_summary=str(output_obj)[:1000] if output_obj else "No output",
            latency_ms=max(0, latency_ms),
            tokens_used=tokens,
            timestamp=start_evt.timestamp
        ))
        
    trace = ExecutionTraceResponse(
        run_id=run_id,
        session_id=context.session_id,
        status=AgentStatus.COMPLETED if execution_event.status == ExecutionStatus.COMPLETED else AgentStatus.FAILED,
        steps=steps,
        total_tokens=execution_event.total_tokens_used,
        total_latency_ms=int((execution_event.finished_at - execution_event.started_at).total_seconds() * 1000) if execution_event.finished_at else 0
    )
    
    _TRACES[run_id] = trace
    logger.info(f"Recorded trace for run_id: {run_id} ({len(steps)} steps)")
