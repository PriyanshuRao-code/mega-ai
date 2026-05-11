import uuid
import logging
from datetime import datetime
from api.services import IQueryService
from api.models import QueryRequest, QueryResponse
from api.trace_service_impl import record_execution_result
from contracts.shared_context import SharedContext
from orchestrator.orchestrator import MultiAgentOrchestrator

logger = logging.getLogger(__name__)

class QueryService(IQueryService):
    """
    Concrete implementation of IQueryService.
    Binds the FastAPI layer to the MultiAgentOrchestrator.
    """
    def __init__(self, orchestrator: MultiAgentOrchestrator):
        self.orchestrator = orchestrator

    async def submit(self, request: QueryRequest) -> QueryResponse:
        # 1. Create a unique execution ID for tracking
        execution_id = str(uuid.uuid4())
        
        # 2. Initialize SharedContext (The 'source of truth' for this request)
        # Based on your contracts/shared_context requirements
        context = SharedContext(
            session_id=request.session_id or f"sess_{execution_id[:8]}",
            query=request.query,
            metadata={
                "execution_id": execution_id,
                "start_time": datetime.utcnow().isoformat(),
                "model_preference": getattr(request, "config_overrides", {}).get("model", "default") if getattr(request, "config_overrides", None) else "default"
            }
        )

        logger.info(f"Execution {execution_id}: Pipeline started for query.")

        try:
            # 3. Call the Orchestrator
            final_ctx, exec_event = await self.orchestrator.run(context)

            # 4. Record Trace for later retrieval
            record_execution_result(execution_id, final_ctx, exec_event)

            # 4. Map the Orchestrator result to the API QueryResponse
            synthesis_out = final_ctx.agent_outputs.get("SynthesisAgent")
            answer_text = str(synthesis_out.output.merged_output) if synthesis_out and hasattr(synthesis_out.output, "merged_output") else str(synthesis_out.output) if synthesis_out else "No synthesis output"
            
            return QueryResponse(
                answer=answer_text,
                session_id=final_ctx.session_id,
                trace_id=execution_id,
                status="completed",
                metadata={
                    "agent_steps": len(final_ctx.agent_outputs),
                    "processing_time_ms": int((exec_event.finished_at - exec_event.started_at).total_seconds() * 1000) if exec_event and exec_event.finished_at else 0
                }
            )

        except Exception as e:
            logger.error(f"Execution {execution_id}: Pipeline failed - {str(e)}", exc_info=True)
            # Re-raise so FastAPI's error handlers (api/error_handlers.py) can format the 500
            raise e
    async def stream(self, request: QueryRequest):
        import asyncio
        from api.models import SSEEvent, SSEEventType
        
        execution_id = str(uuid.uuid4())
        q = asyncio.Queue()
        
        context = SharedContext(
            session_id=request.session_id or f"sess_{execution_id[:8]}",
            query=request.query,
            metadata={
                "execution_id": execution_id,
                "start_time": datetime.utcnow().isoformat(),
                "model_preference": getattr(request, "config_overrides", {}).get("model", "default") if getattr(request, "config_overrides", None) else "default",
                "_stream_queue": q
            }
        )
        
        logger.info(f"Streaming execution {execution_id}: Pipeline started.")
        task = asyncio.create_task(self.orchestrator.run(context))
        
        try:
            while not task.done():
                try:
                    event = await asyncio.wait_for(q.get(), timeout=0.1)
                    yield SSEEvent(
                        event=SSEEventType.ACTIVE_AGENT,
                        run_id=execution_id,
                        data={
                            "agent": event.agent_name,
                            "type": event.event_type.value,
                            "metadata": event.metadata
                        }
                    )
                except asyncio.TimeoutError:
                    continue
            
            while not q.empty():
                event = q.get_nowait()
                yield SSEEvent(
                    event=SSEEventType.ACTIVE_AGENT,
                    run_id=execution_id,
                    data={
                        "agent": event.agent_name,
                        "type": event.event_type.value,
                        "metadata": event.metadata
                    }
                )
                
            final_ctx, exec_event = await task
            
            # Record Trace for later retrieval
            record_execution_result(execution_id, final_ctx, exec_event)

            # Get final answer for the stream
            synthesis_out = final_ctx.agent_outputs.get("SynthesisAgent")
            answer_text = str(synthesis_out.output.merged_output) if synthesis_out and hasattr(synthesis_out.output, "merged_output") else str(synthesis_out.output) if synthesis_out else "No synthesis output"
            
            yield SSEEvent(
                event=SSEEventType.DONE,
                run_id=execution_id,
                data={
                    "answer": answer_text,
                    "status": "completed",
                    "agent_steps": len(final_ctx.agent_outputs)
                }
            )
            
        except Exception as e:
            logger.error(f"Streaming execution {execution_id}: Pipeline failed - {str(e)}", exc_info=True)
            raise e
