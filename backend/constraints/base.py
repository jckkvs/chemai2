"""
Base Constraint Definitions - chemai2/backend/constraints/base.py
Common data structures for monotonicity and linearity constraints
"""
from dataclasses import dataclass
from typing import Literal, Optional, Dict, List

@dataclass
class ConstraintSpec:
    """Specification for a single feature constraint"""
    feature_name: str
    monotonic: Optional[Literal['increasing', 'decreasing']] = None
    linearity: Literal['none', 'weak', 'strong'] = 'none'
    sigma_range: float = 3.0
    strength: Literal['weak', 'strong'] = 'weak'
    direction: Optional[Literal['increasing', 'decreasing']] = None  # Alias for monotonic

@dataclass
class ConstraintEvaluation:
    """Result of a constraint validation check"""
    feature_name: str
    violation_ratio: float
    is_satisfied: bool
    details: Dict[str, float]
