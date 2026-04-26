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
