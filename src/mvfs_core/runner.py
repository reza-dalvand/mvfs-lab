from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class ProjectRunResult:
    """
    Standard result object returned by every project adapter.

    Important:
    --------
    This does NOT force a common data contract.
    It only standardizes the execution output.
    """

    project: str
    dataset: str
    ranking: np.ndarray
    metrics_by_ratio: List[Dict[str, Any]]
    best_metrics: Dict[str, float]
    runtime_seconds: float
    config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def metric_names(self) -> List[str]:
        if not self.metrics_by_ratio:
            return []
        return [k for k in self.metrics_by_ratio[0].keys() if k != "prop"]