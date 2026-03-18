"""
tests/test_automl_extra.py

automl.py のカバレッジ改善テスト。
AutoMLResult, AutoMLEngine を網羅（外部依存なしで動作する範囲）。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.models.automl import AutoMLResult, AutoMLEngine


# ============================================================
# AutoMLResult
# ============================================================

class TestAutoMLResult:
    def test_construction(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        pipe = Pipeline([("model", Ridge())])
        dr = None  # detection_result はテスト時は省略可
        result = AutoMLResult(
            task="regression",
            best_model_key="ridge",
            best_pipeline=pipe,
            best_score=0.95,
            scoring="r2",
            model_scores={"ridge": 0.95},
            model_details={"ridge": {"mean": 0.95, "std": 0.01}},
            detection_result=dr,
            elapsed_seconds=1.5,
        )
        assert result.best_model_key == "ridge"
        assert result.best_score == 0.95
        assert result.elapsed_seconds == 1.5

    def test_with_oof(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        pipe = Pipeline([("model", Ridge())])
        result = AutoMLResult(
            task="regression",
            best_model_key="ridge",
            best_pipeline=pipe,
            best_score=0.9,
            scoring="r2",
            model_scores={"ridge": 0.9},
            model_details={},
            detection_result=None,
            elapsed_seconds=1.0,
            oof_predictions=np.array([1.0, 2.0, 3.0]),
            oof_true=np.array([1.1, 2.1, 2.9]),
        )
        assert result.oof_predictions is not None
        assert len(result.oof_predictions) == 3


# ============================================================
# AutoMLEngine
# ============================================================

class TestAutoMLEngine:
    def test_init_defaults(self):
        engine = AutoMLEngine()
        assert engine.task == "auto"

    def test_init_custom(self):
        engine = AutoMLEngine(task="regression", cv_folds=3)
        assert engine.task == "regression"

    def test_detect_task_regression(self):
        """連続目的変数の場合 → regression 判定"""
        engine = AutoMLEngine(task="auto")
        rng = np.random.RandomState(42)
        y = rng.randn(50)
        task = engine._detect_task(y)
        assert task == "regression"

    def test_detect_task_classification(self):
        """離散値（少数ユニーク）の場合 → classification 判定"""
        engine = AutoMLEngine(task="auto")
        y = np.array([0, 1, 2, 0, 1, 2] * 10)
        task = engine._detect_task(y)
        assert task == "classification"

    def test_detect_task_explicit(self):
        """明示的にtask指定"""
        engine = AutoMLEngine(task="regression")
        y = np.array([0, 1, 0, 1])  # would be classified, but explicit override
        task = engine._detect_task(y)
        assert task == "regression"

    def test_run_minimal(self):
        """最小データでの run() テスト。"""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "f1": rng.randn(60),
            "f2": rng.randn(60),
            "target": rng.randn(60) * 2,
        })
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge"],
        )
        result = engine.run(df, target_col="target")
        assert result.task == "regression"
        assert result.best_model_key == "ridge"
        assert result.best_score is not None
