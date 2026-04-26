# backend/pipeline/column_selector.py — 精緻化版 (列選択エンジン)

from typing import Union, List, Dict, Optional, Set, Tuple, Pattern
from dataclasses import dataclass, field
import re
import logging
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ColumnSelectionRule:
    """
    Rule for selecting columns with pattern matching and dependencies
    """
    include: Union[str, List[str]] = field(default_factory=list)
    exclude: Union[str, List[str]] = field(default_factory=list)
    required: Union[str, List[str]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    regex: bool = False
    
    def __post_init__(self):
        # 【修正点1】文字列をリストに正規化
        if isinstance(self.include, str): self.include = [self.include]
        if isinstance(self.exclude, str): self.exclude = [self.exclude]
        if isinstance(self.required, str): self.required = [self.required]
        
        # 【修正点1】列名の前後空白トリム
        self.include = [c.strip() for c in self.include if c.strip()]
        self.exclude = [c.strip() for c in self.exclude if c.strip()]
        self.required = [c.strip() for c in self.required if c.strip()]
        
        # 【修正点2】依存関係のキーもトリム
        self.dependencies = {
            k.strip(): [v.strip() for v in vals if v.strip()]
            for k, vals in self.dependencies.items() if k.strip()
        }


def select_columns(
    df: pd.DataFrame,
    rule: ColumnSelectionRule,
    strict: bool = True,
    warn_missing: bool = True
) -> pd.DataFrame:
    """
    Select columns from DataFrame based on selection rules with dependency resolution
    """
    available_cols = set(df.columns)
    
    # 【修正点3】必須列の存在チェック
    missing_required = [c for c in rule.required if c not in available_cols]
    if missing_required:
        msg = f"Required columns not found: {missing_required}"
        if strict: raise ValueError(msg)
        else:
            logger.warning(msg)
            rule.required = [c for c in rule.required if c not in missing_required]
    
    # 【修正点1】列名の正規化とパターン展開
    include_set = _expand_patterns(rule.include, available_cols, rule.regex)
    exclude_set = _expand_patterns(rule.exclude, available_cols, rule.regex)
    
    if not include_set: include_set = available_cols.copy()
    selected = include_set - exclude_set
    
    # 【修正点3】存在しない列の警告
    all_referenced = set(rule.include) | set(rule.exclude) | set(rule.required)
    for col in all_referenced:
        if col not in available_cols and not _is_pattern(col, rule.regex):
            if warn_missing: logger.warning(f"Column '{col}' referenced but not found")
    
    # 【修正点2】依存関係の解決
    if rule.dependencies:
        selected = _resolve_dependencies(selected, rule.dependencies, available_cols)
    
    final_cols = [c for c in df.columns if c in selected]
    return df[final_cols]


def _expand_patterns(patterns: List[str], available: Set[str], use_regex: bool) -> Set[str]:
    """Expand glob-style or regex patterns to matching column names"""
    result = set()
    for pattern in patterns:
        if not pattern: continue
        if pattern in available: result.add(pattern); continue
        
        if use_regex:
            try:
                compiled = re.compile(pattern)
                result.update([c for c in available if compiled.search(c)])
            except re.error as e: logger.warning(f"Invalid regex '{pattern}': {e}")
        else:
            import fnmatch
            result.update([c for c in available if fnmatch.fnmatch(c, pattern)])
    return result


def _is_pattern(s: str, use_regex: bool) -> bool:
    """Check if string contains pattern wildcards"""
    return bool(re.search(r'[\.\^\$\*\+\?\{\}\[\]\(\)\|\\]', s)) if use_regex else any(c in s for c in '*?[')


def _resolve_dependencies(selected: Set[str], dependencies: Dict[str, List[str]], available: Set[str]) -> Set[str]:
    """Resolve column dependencies using topological sort"""
    graph, in_degree = defaultdict(set), defaultdict(int)
    for col in selected: in_degree[col] = 0
    
    for col, deps in dependencies.items():
        if col not in selected: continue
        for dep in deps:
            if dep in selected and dep in available:
                graph[dep].add(col); in_degree[col] += 1
    
    from collections import deque
    queue = deque([c for c in selected if in_degree[c] == 0])
    resolved = []
    while queue:
        curr = queue.popleft(); resolved.append(curr)
        for neighbor in graph[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0: queue.append(neighbor)
            
    if len(resolved) != len(selected):
        unresolved = set(selected) - set(resolved)
        cycle = _find_cycle_example(unresolved, dependencies)
        raise ValueError(f"Circular dependency detected: {cycle}")
    return set(resolved)


def _find_cycle_example(unresolved: Set[str], dependencies: Dict[str, List[str]]) -> str:
    """Find an example cycle path for error message"""
    def dfs(node: str, path: List[str], visited: Set[str]) -> Optional[List[str]]:
        if node in path: return path[path.index(node):] + [node]
        if node in visited or node not in dependencies: return None
        visited.add(node); path.append(node)
        for dep in dependencies.get(node, []):
            if dep in unresolved:
                res = dfs(dep, path.copy(), visited)
                if res: return res
        return None
    for start in unresolved:
        cycle = dfs(start, [], set())
        if cycle: return ' → '.join(cycle)
    return "Circular dependency found"


def get_column_groups(df: pd.DataFrame, grouping_rules: Dict[str, Union[str, List[str], Dict]]) -> Dict[str, pd.DataFrame]:
    """Group columns into logical categories"""
    groups = {}
    for name, spec in grouping_rules.items():
        if isinstance(spec, ColumnSelectionRule): groups[name] = select_columns(df, spec)
        elif isinstance(spec, (str, list)): groups[name] = select_columns(df, ColumnSelectionRule(include=spec))
        elif isinstance(spec, dict) and 'include' in spec: groups[name] = select_columns(df, ColumnSelectionRule(**spec))
    return groups
