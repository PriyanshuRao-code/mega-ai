"""
IMPORTS: abc, typing, evals.datasets (EvalCase, EvalResult)
INPUTS: EvalCase, Agent Output (Dict)
OUTPUTS: float (score 0.0 to 1.0)
DEPENDENCIES: evals.datasets
EXCEPTIONS: KeyError (if agent output misses required scoring fields)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from evals.datasets import EvalCase

class IScorer(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def score(self, case: EvalCase, agent_output: Dict[str, Any]) -> float:
        pass

class CorrectnessScorer(IScorer):
    def name(self) -> str: return "correctness"
    def score(self, case: EvalCase, agent_output: Dict[str, Any]) -> float:
        # Mock logic: Compare agent_output with case.expected_output
        return 1.0 if agent_output.get("status") == "success" else 0.0

class CitationAccuracyScorer(IScorer):
    def name(self) -> str: return "citation_accuracy"
    def score(self, case: EvalCase, agent_output: Dict[str, Any]) -> float:
        return 1.0 if len(agent_output.get("citations", [])) > 0 else 0.5

class ContradictionResolutionScorer(IScorer):
    def name(self) -> str: return "contradiction_resolution"
    def score(self, case: EvalCase, agent_output: Dict[str, Any]) -> float:
        if case.category.value == "adversarial":
            return 1.0 if agent_output.get("contradiction_handled") else 0.0
        return 1.0 # N/A for non-contradictory cases

class ToolEfficiencyScorer(IScorer):
    def name(self) -> str: return "tool_efficiency"
    def score(self, case: EvalCase, agent_output: Dict[str, Any]) -> float:
        tool_calls = agent_output.get("tool_calls_count", 0)
        return max(0.0, 1.0 - (tool_calls * 0.1))

class ContextComplianceScorer(IScorer):
    def name(self) -> str: return "context_compliance"
    def score(self, case: EvalCase, agent_output: Dict[str, Any]) -> float:
        return 1.0 if agent_output.get("used_context") else 0.0

class ScoreAggregator:
    def __init__(self, scorers: List[IScorer]):
        self.scorers = scorers

    def evaluate(self, case: EvalCase, agent_output: Dict[str, Any]) -> Dict[str, float]:
        return {scorer.name(): scorer.score(case, agent_output) for scorer in self.scorers}