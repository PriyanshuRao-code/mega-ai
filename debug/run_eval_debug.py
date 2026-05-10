"""
IMPORTS: sys, os, evals.* modules
INPUTS: None (Run via CLI)
OUTPUTS: Standard Output (Console logs)
DEPENDENCIES: evals.harness, evals.datasets, evals.adversarial_cases, evals.scorers, evals.regression_tracker
EXCEPTIONS: SystemExit (if regressions are found)
"""

import sys
import os

# Ensure the parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.datasets import DatasetLoader
from evals.adversarial_cases import AdversarialLoader
from evals.scorers import (
    ScoreAggregator, CorrectnessScorer, CitationAccuracyScorer,
    ContradictionResolutionScorer, ToolEfficiencyScorer, ContextComplianceScorer
)
from evals.harness import EvalHarness
from evals.regression_tracker import RegressionTracker

# Mock agent executor for debugging purposes
def mock_agent_executor(case) -> dict:
    return {
        "status": "success",
        "citations": ["doc_1"],
        "contradiction_handled": True,
        "tool_calls_count": 1,
        "used_context": True
    }

def main():
    print("Initializing Multi-Agent Evaluation Harness...")
    
    # 1. Load Datasets
    cases = []
    cases.extend(DatasetLoader.get_baseline_cases())
    cases.extend(DatasetLoader.get_ambiguous_cases())
    cases.extend(AdversarialLoader.get_adversarial_cases())
    
    print(f"Loaded {len(cases)} total evaluation cases.")

    # 2. Setup Scorers (Dependency Injection)
    scorers = [
        CorrectnessScorer(),
        CitationAccuracyScorer(),
        ContradictionResolutionScorer(),
        ToolEfficiencyScorer(),
        ContextComplianceScorer()
    ]
    aggregator = ScoreAggregator(scorers)

    # 3. Setup Harness
    harness = EvalHarness(aggregator=aggregator, agent_executor=mock_agent_executor)

    # 4. Execute Suite
    print("Executing tests...")
    results = harness.run_suite(cases)

    # 5. Validate Scoring
    print("\n--- EVALUATION RESULTS ---")
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        print(f"[{status}] Case ID: {res.case_id} | Total Score: {res.total_score:.2f}")
        for dim, score in res.scores.items():
            print(f"    - {dim}: {score:.2f}")

    # 6. Check Regressions
    tracker = RegressionTracker()
    regressions = tracker.detect_regressions(results)

    if regressions:
        print("\n!!! REGRESSIONS DETECTED !!!")
        for reg in regressions:
            print(f"Case {reg['case_id']} dropped from {reg['previous_score']} to {reg['current_score']:.2f} (Drop: {reg['drop']:.2f})")
        sys.exit(1)
    else:
        print("\nNo regressions detected. Ready for deployment.")
        sys.exit(0)

if __name__ == "__main__":
    main()