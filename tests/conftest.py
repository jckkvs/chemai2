"""
tests/conftest.py

pytest共通フィクスチャと設定。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def random_seed() -> int:
    return 42


@pytest.fixture(scope="session")
def small_regression_df() -> pd.DataFrame:
    """セッションスコープの小さい回帰DataFrameフィクスチャ。"""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "numeric_a": np.random.randn(n),
        "numeric_b": np.random.exponential(5, n),
        "cat_a": np.random.choice(["X", "Y", "Z"], n),
        "binary": np.random.randint(0, 2, n).astype(float),
        "target": np.random.randn(n),
    })


@pytest.fixture(scope="session")
def small_classification_df() -> pd.DataFrame:
    """セッションスコープの分類DataFrameフィクスチャ。"""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.choice(["A", "B", "C"], n),
        "target": np.random.randint(0, 2, n),
    })
