import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contracts.shared_context import SharedContext
from agents.retrieval_agent import RetrievalAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

async def test_retrieval():
    print("\n" + "="*70)
    print("Testing RetrievalAgent with Local Documents")
    print("="*70)
    
    agent = RetrievalAgent()
    
    # Test case 1: Query matching Q3 earnings report
    ctx1 = SharedContext(
        task_id="test-retrieval-01",
        query="Summarise the Q3 earnings report and identify risks",
    )
    print(f"\nQuery 1: {ctx1.query}")
    result1 = agent.run(ctx1)
    
    print(f"\nResults for Query 1:")
    print(f"Chunks retrieved: {len(result1.chunks)}")
    for i, chunk in enumerate(result1.chunks):
        print(f"  [{i+1}] Source: {chunk.source}")
        print(f"      Score: {chunk.score}")
        print(f"      Content: {chunk.content[:200]}...")
    
    # Test case 2: Query matching RAG vs Fine-tuning
    ctx2 = SharedContext(
        task_id="test-retrieval-02",
        query="What are the pros and cons of RAG compared to fine-tuning?",
    )
    print(f"\nQuery 2: {ctx2.query}")
    result2 = agent.run(ctx2)
    
    print(f"\nResults for Query 2:")
    print(f"Chunks retrieved: {len(result2.chunks)}")
    for i, chunk in enumerate(result2.chunks):
        print(f"  [{i+1}] Source: {chunk.source}")
        print(f"      Score: {chunk.score}")
        print(f"      Content: {chunk.content[:200]}...")

    # Test case 3: Query with no match (should fallback)
    ctx3 = SharedContext(
        task_id="test-retrieval-03",
        query="How to cook a pizza?",
        metadata={"min_score": 0.5} # Higher threshold to force fallback
    )
    print(f"\nQuery 3: {ctx3.query} (with min_score=0.5)")
    result3 = agent.run(ctx3)
    
    print(f"\nResults for Query 3:")
    print(f"Chunks retrieved: {len(result3.chunks)}")
    for i, chunk in enumerate(result3.chunks):
        print(f"  [{i+1}] Source: {chunk.source} | Score: {chunk.score}")
    if result3.metadata.get("stub"):
        print("  Verified: Fallback synthetic chunk triggered.")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
