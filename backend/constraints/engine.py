"""
Constraint Enforcement Engine - chemai2/backend/constraints/engine.py

Re-exports from backend.ml.constraints for backward compatibility.
"""

from backend.ml.constraints import (
    ConstraintSpec,
    ConstraintEvaluation,
    ConstraintEngine,
)

__all__ = [
    'ConstraintSpec',
    'ConstraintEvaluation',
    'ConstraintEngine',
]
