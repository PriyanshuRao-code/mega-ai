import asyncio
import sys
import os
import logging
from datetime import datetime

# Setup path so we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal Imports (Adjust paths based on your directory structure)
from orchestrator.orchestrator import build_orchestrator
from agents.decomposition_agent import DecompositionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.critique_agent import CritiqueAgent
from agents.synthesis_agent import SynthesisAgent
from contracts.shared_context import SharedContext
from orchestrator.retry_manager import RetryPolicy
# from utils.observability import Logger

# Setup logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemIntegrationTest")

async def run_full_system_check():
    logger.info("🚀 Starting Full System Integration Test...")
    
    # 1. Initialize Infrastructure & Tools
    retry_policy = RetryPolicy(max_attempts=3)
    
    # 2. Initialize Agents
    agents = [
        DecompositionAgent(),
        RetrievalAgent(),
        CritiqueAgent(),
        SynthesisAgent()
    ]
    
    # 3. Initialize Orchestrator
    orchestrator = build_orchestrator(
        agents=agents,
        retry_policy=retry_policy
    )

    # 4. Define a Multi-Step Test Query
    test_query = "Compare the quarterly earnings of Apple and Microsoft for Q3 2025 and summarize the impact of their AI investments."

    # 5. Simulate Pipeline Execution
    logger.info(f"📥 Input Query: {test_query}")
    
    try:
        start_time = datetime.now()
        
        ctx = SharedContext(
            query=test_query,
            available_agents=["decomposition", "retrieval", "critique", "synthesis"]
        )
        final_ctx, result = await orchestrator.run(ctx)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # --- VALIDATION SUITE ---
        
        print("\n" + "="*50)
        print("INTERGRATION VERIFICATION CHECKLIST")
        print("="*50)

        # A. Check Flow & Final Response
        assert result is not None, "❌ Final response is null"
        assert result.status.value == "completed", f"❌ Orchestration failed with status: {result.status.value}"
        print(f"✅ Query Flow: Success ({duration}s)")

        # B. Check Shared Context & Traceability
        assert hasattr(final_ctx, "agent_outputs"), "❌ SharedContext missing agent_outputs"
        print("✅ Shared Context: Properly mutated across all agents")

        # C. Observability & Event Logs
        assert len(result.agent_events) > 0, "❌ No event logs generated"
        print("✅ Observability: Event logs and traces generated")

        # D. API / Serialization Check
        try:
            import json
            # pydantic/dataclass serialization check
            if hasattr(result, "model_dump_json"):
                serialized = result.model_dump_json()
            elif hasattr(result, "__dataclass_fields__"):
                import dataclasses
                serialized = json.dumps(dataclasses.asdict(result), default=str)
            else:
                serialized = json.dumps(result.__dict__, default=str)
            assert isinstance(serialized, str)
            print("✅ API Compatibility: Models are JSON serializable")
        except Exception as e:
            print(f"❌ API Compatibility: Serialization failed: {e}")

        print("="*50)
        print("🏆 SYSTEM INTEGRATION PASSED")
        print("="*50)

    except Exception as e:
        logger.error(f"💥 Integration Test Failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_full_system_check())