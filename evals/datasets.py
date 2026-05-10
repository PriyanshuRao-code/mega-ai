"""
IMPORTS: dataclasses, enum, typing
INPUTS: None (for data generation methods)
OUTPUTS: List[EvalCase], EvalResult definitions
DEPENDENCIES: Standard library only
EXCEPTIONS: ValueError (if case validation fails)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

class TestCategory(Enum):
    BASELINE = "baseline"
    AMBIGUOUS = "ambiguous"
    ADVERSARIAL = "adversarial"

@dataclass
class EvalCase:
    id: str
    category: TestCategory
    input_prompt: str
    expected_output: Dict[str, Any]
    context: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id or not self.input_prompt:
            raise ValueError("EvalCase must have an id and input_prompt")

@dataclass
class EvalResult:
    case_id: str
    scores: Dict[str, float]  # e.g. {"correctness": 0.9, ...}
    total_score: float
    agent_output: Dict[str, Any]
    passed: bool

class DatasetLoader:
    """Follows Single Responsibility Principle for loading non-adversarial data."""
    
    @staticmethod
    def get_baseline_cases() -> List[EvalCase]:
        return [
            EvalCase(
                id="base_001",
                category=TestCategory.BASELINE,
                input_prompt="Summarize the company policy.",
                expected_output={"summary_length": "short", "includes_core_values": True},
                context={"policy_text": "Our core values are integrity and speed..."}
            )
        ]

    @staticmethod
    def get_ambiguous_cases() -> List[EvalCase]:
        return [
            EvalCase(
                id="ambig_001",
                category=TestCategory.AMBIGUOUS,
                input_prompt="Fix the system.",
                expected_output={"asks_clarifying_questions": True},
                context={"system_state": "unknown"}
            )
        ]