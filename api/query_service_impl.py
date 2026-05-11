import uuid
import logging
from datetime import datetime
from api.services import IQueryService
from api.models import QueryRequest, QueryResponse
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
            query=request.prompt,
            metadata={
                "execution_id": execution_id,
                "start_time": datetime.utcnow().isoformat(),
                "model_preference": getattr(request, "model", "default")
            }
        )

        logger.info(f"Execution {execution_id}: Pipeline started for query.")

        try:
            # 3. Call the Orchestrator
            final_ctx, exec_event = await self.orchestrator.run(context)

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
    async def stream(self, request):
        raise NotImplementedError("Streaming not yet implemented")
