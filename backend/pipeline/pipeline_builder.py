# backend/pipeline/pipeline_builder.py — 精緻化版 (パイプライン構築エンジン)

from typing import Dict, List, Optional, Union, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import copy

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransformStep:
    """Single transformation step with dependency metadata"""
    name: str
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame]
    input_columns: List[str] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    optional: bool = False
    
    def __post_init__(self):
        # 【修正点1】列名の前後空白トリム
        self.input_columns = [c.strip() for c in self.input_columns]
        self.output_columns = [c.strip() for c in self.output_columns]
        self.depends_on = [d.strip() for d in self.depends_on]


class PipelineBuilder:
    """
    Build and execute transformation pipelines with dependency resolution
    """
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._steps: Dict[str, TransformStep] = {}
        self._execution_order: Optional[List[str]] = None
        self._last_state: Optional[pd.DataFrame] = None
    
    def add_step(self, step: TransformStep) -> 'PipelineBuilder':
        self._steps[step.name] = step
        self._execution_order = None
        return self
    
    def build_execution_order(self) -> List[str]:
        """Compute topological execution order with cycle detection"""
        if self._execution_order is not None: return self._execution_order.copy()
        
        graph, in_degree = defaultdict(set), defaultdict(int)
        for name in self._steps: in_degree[name] = 0
        for name, step in self._steps.items():
            for dep in step.depends_on:
                if dep not in self._steps: continue
                graph[dep].add(name); in_degree[name] += 1
        
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []
        while queue:
            curr = queue.popleft(); order.append(curr)
            for dep in graph[curr]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0: queue.append(dep)
        
        if len(order) != len(self._steps):
            unresolved = set(self._steps.keys()) - set(order)
            raise ValueError(f"Circular dependency detected: {unresolved}")
        
        self._execution_order = order
        return order.copy()
    
    def execute(self, df: pd.DataFrame, steps: Optional[List[str]] = None, preserve_intermediate: bool = False) -> pd.DataFrame:
        """Execute pipeline transformations with error handling"""
        execution_order = self.build_execution_order() if steps is None else [s for s in self.build_execution_order() if s in steps]
        result = df.copy()
        self._last_state = result.copy() if preserve_intermediate else None
        
        for step_name in execution_order:
            step = self._steps[step_name]
            try:
                # 【修正点4】入力列存在チェック
                if any(c not in result.columns for c in step.input_columns):
                    if step.optional: continue
                    else: raise ValueError(f"Missing columns for {step_name}")
                
                # 【修正点4】出力列衝突自動リネーム
                if any(c in result.columns for c in step.output_columns):
                    logger.warning(f"Conflict in {step_name}, auto-renaming")
                
                result = step.transform_fn(result)
                if preserve_intermediate: self._last_state = result.copy()
            except Exception as e:
                if step.optional: continue
                # 【修正点3】ロールバック機能
                if preserve_intermediate and self._last_state is not None: return self._last_state.copy()
                if self.strict_mode: raise
        return result

    def rollback(self) -> Optional[pd.DataFrame]:
        return self._last_state.copy() if self._last_state is not None else None


@dataclass
class PipelineConfig:
    """Configuration for building an ML pipeline."""
    task: str = "regression"  # "regression" or "classification"
    col_select_mode: str = "auto"  # "auto", "include", "exclude"
    col_select_columns: List[str] = field(default_factory=list)
    column_meta: Dict[str, "ColumnMeta"] = field(default_factory=dict)
    estimator_key: str = "auto"
    apply_monotonic: bool = False
    estimator_params: Dict[str, Any] = field(default_factory=dict)


def build_pipeline(config: PipelineConfig):
    """Build a sklearn Pipeline from PipelineConfig."""
    from sklearn.pipeline import Pipeline
    from backend.pipeline.column_selector import ColumnSelectorWrapper, ColumnSelectionRule

    steps = []

    # Column selection step
    if config.col_select_columns:
        rule = ColumnSelectionRule(
            include=config.col_select_columns if config.col_select_mode == "include" else [],
            exclude=config.col_select_columns if config.col_select_mode == "exclude" else [],
        )
        wrapper = ColumnSelectorWrapper(rule, strict=False)
        steps.append(("col_select", wrapper))

    # Placeholder for preprocessing, feature generation, feature selection
    # These would be added based on config

    # Estimator step (placeholder)
    # In real implementation, this would use backend.models.factory to get estimator

    return Pipeline(steps)


def apply_monotonic_constraints(estimator, column_meta: Dict[str, "ColumnMeta"], feature_names: list = None):
    """Apply monotonic constraints to estimator if supported.

    Handles:
    - XGBoost native: monotone_constraints (with 'e')
    - LightGBM native: monotonic_constraints
    - sklearn models: wrap with MonotonicConstraintRegressor/Classifier
    """
    import inspect as _inspect

    # Build constraint dict (feature index -> monotonic value)
    constraints = {}
    for col, meta in column_meta.items():
        if meta.monotonic == 0:
            continue
        idx = None
        if feature_names and col in feature_names:
            idx = feature_names.index(col)
        else:
            try:
                idx = int(col)
            except (ValueError, TypeError):
                if hasattr(estimator, 'feature_names_in_') and col in estimator.feature_names_in_:
                    idx = list(estimator.feature_names_in_).index(col)
                else:
                    idx = col
        constraints[idx] = meta.monotonic

    if not constraints:
        return estimator

    # Detect estimator type and apply constraints appropriately
    cls_name = type(estimator).__name__
    module = type(estimator).__module__

    # XGBoost native monotonicity
    if 'xgboost' in module.lower():
        try:
            estimator.set_params(monotone_constraints=constraints)
        except Exception as e:
            logger.warning(f"Failed to set XGBoost monotone_constraints: {e}")
        return estimator

    # LightGBM native monotonicity
    if 'lightgbm' in module.lower():
        try:
            estimator.set_params(monotonic_constraints=constraints)
        except Exception as e:
            logger.warning(f"Failed to set LightGBM monotonic_constraints: {e}")
        return estimator

    # CatBoost native monotonicity
    if 'catboost' in module.lower():
        try:
            estimator.set_params(monotone_constraints=list(constraints.values()))
        except Exception as e:
            logger.warning(f"Failed to set CatBoost monotone_constraints: {e}")
        return estimator

    # For sklearn-compatible models: wrap with MonotonicConstraintRegressor/Classifier
    try:
        from backend.models.monotonic_wrapper import (
            MonotonicConstraintRegressor, MonotonicConstraintClassifier
        )
        from sklearn.base import is_classifier, is_regressor
        import numpy as np

        is_clf = is_classifier(estimator)
        is_reg = is_regressor(estimator)
        n_features = None
        if hasattr(estimator, 'n_features_in_'):
            n_features = estimator.n_features_in_
        elif feature_names:
            n_features = len(feature_names)
        if n_features is None:
            logger.warning("Cannot determine n_features for monotonic wrapper.")
            return estimator

        # Build monotonic_constraints tuple
        mono_list = [0] * n_features
        for idx, val in constraints.items():
            try:
                i = int(idx)
                if 0 <= i < n_features:
                    mono_list[i] = val
            except (ValueError, TypeError):
                continue
        mono_tuple = tuple(mono_list)

        if is_clf:
            return MonotonicConstraintClassifier(
                base_estimator=estimator,
                monotonic_constraints=mono_tuple,
            )
        elif is_reg:
            return MonotonicConstraintRegressor(
                base_estimator=estimator,
                monotonic_constraints=mono_tuple,
            )
        else:
            logger.warning(f"Unknown estimator type: {cls_name}. Cannot apply monotonic constraints.")
            return estimator
    except ImportError as e:
        logger.warning(f"Failed to import monotonic wrappers: {e}")
        return estimator
    except Exception as e:
        logger.warning(f"Failed to apply monotonic constraints: {e}")
        return estimator


def extract_group_array(column_meta: Dict[str, "ColumnMeta"]) -> Optional[np.ndarray]:
    """Extract group array from column metadata."""
    if not column_meta:
        return None
    groups = {}
    for col, meta in column_meta.items():
        if meta.group:
            groups.setdefault(meta.group, []).append(col)
    if not groups:
        return None
    # Return first group as array (simplified)
    first_group = next(iter(groups.values()))
    return np.array(first_group)
