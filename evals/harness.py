"""
IMPORTS: typing, evals.datasets, evals.scorers, evals.regression_tracker
INPUTS: List[EvalCase], System Callables/Mock Agent
OUTPUTS: List[EvalResult]
DEPENDENCIES: evals.datasets, evals.scorers
EXCEPTIONS: RuntimeError (if agent execution fails)
"""

from typing import List, Dict, Any, Callable
from evals.datasets import EvalCase, EvalResult
from evals.scorers import ScoreAggregator

class EvalHarness:
    def __init__(self, aggregator: ScoreAggregator, agent_executor: Callable[[EvalCase], Dict[str, Any]]):
        self.aggregator = aggregator
        self.agent_executor = agent_executor

    def run_suite(self, cases: List[EvalCase]) -> List[EvalResult]:
        results = []
        for case in cases:
            try:
                # Execute the agent
                agent_output = self.agent_executor(case)
                
                # Apply scorers
                scores = self.aggregator.evaluate(case, agent_output)
                total_score = sum(scores.values()) / len(scores) if scores else 0.0
                
                passed = total_score >= 0.8  # Threshold for passing

                results.append(EvalResult(
                    case_id=case.id,
                    scores=scores,
                    total_score=total_score,
                    agent_output=agent_output,
                    passed=passed
                ))
            except Exception as e:
                raise RuntimeError(f"Harness failed on case {case.id}: {str(e)}")
        
        return results