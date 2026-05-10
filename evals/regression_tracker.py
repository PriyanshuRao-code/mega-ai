"""
IMPORTS: typing, evals.datasets (EvalResult)
INPUTS: List[EvalResult]
OUTPUTS: List[Dict] (regression reports)
DEPENDENCIES: evals.datasets
EXCEPTIONS: ValueError (if historical data is malformed)
"""

from typing import List, Dict, Optional
from evals.datasets import EvalResult

class RegressionTracker:
    def __init__(self):
        # In production, this would load from a DB or JSON file
        self.history: Dict[str, float] = {
            "base_001": 0.95,
            "ambig_001": 0.80,
            "adv_001": 0.90,
            "adv_002": 0.85
        }

    def detect_regressions(self, current_results: List[EvalResult], threshold: float = 0.05) -> List[Dict]:
        regressions = []
        for result in current_results:
            historical_score = self.history.get(result.case_id)
            if historical_score is not None:
                drop = historical_score - result.total_score
                if drop > threshold:
                    regressions.append({
                        "case_id": result.case_id,
                        "previous_score": historical_score,
                        "current_score": result.total_score,
                        "drop": drop
                    })
        return regressions