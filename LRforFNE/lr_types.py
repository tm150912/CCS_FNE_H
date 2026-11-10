"""Sparse records; full flow witnesses are retained for current/center intervals."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from heuristic_stages.common import Plan


@dataclass
class OracleResult:
    evaluation_id: int
    multipliers: np.ndarray
    plan: Optional[Plan]
    residual: Optional[np.ndarray]
    original_objective: Optional[float]
    u: Optional[float]
    ell: Optional[float]
    epsilon: Optional[float]
    accurate: bool
    blocks: dict
    status: str
    runtime: float
    bound_evaluation_id: Optional[int] = None  # merged ell source; None means evaluation_id


@dataclass
class LRResult:
    method: str
    status: str
    termination_reason: str
    initial_objective: Optional[float]
    objective: Optional[float]
    lower_bound: Optional[float]
    gap: Optional[float]
    solution: Plan
    history: list
    warm_start_runtime: float
    lr_runtime: float
    total_runtime: float
    best_multipliers: Optional[np.ndarray]
    multiplier_bound: Optional[float]
    multiplier_evaluation: Optional[int]
    lower_bound_origin: str
    audit: Optional[dict]
    configuration: dict
    instance: dict
    timings: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    initial_multipliers: Optional[np.ndarray] = None
    initialization_history: list = field(default_factory=list)

