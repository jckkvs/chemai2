"""
tests/test_inverse_optimizer.py

逆解析エンジン(inverse_optimizer.py)のユニットテスト。
4手法(ランダム/グリッド/ベイズ/GA)と3目標モード(range/maximize/minimize)を検証。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.optim.inverse_optimizer import (
    InverseConfig,
    InverseResult,
    run_inverse_optimization,
    _build_full_df,
    _score_predictions,
    _sbx_crossover,
)


# ─── テスト用の単純なpredict関数 ─────────────────────────
def _linear_predict(X_df: pd.DataFrame) -> np.ndarray:
    """y = x1 + 2*x2 の線形モデルを模擬。"""
    x1 = X_df.iloc[:, 0].values if X_df.shape[1] > 0 else np.zeros(len(X_df))
    x2 = X_df.iloc[:, 1].values if X_df.shape[1] > 1 else np.zeros(len(X_df))
    return x1 + 2 * x2


def _quadratic_predict(X_df: pd.DataFrame) -> np.ndarray:
    """y = -(x1-5)^2 -(x2-3)^2 + 34  (最大値34 at x1=5, x2=3)"""
    x1 = X_df["x1"].values
    x2 = X_df["x2"].values
    return -(x1 - 5.0) ** 2 - (x2 - 3.0) ** 2 + 34.0


# ─── ヘルパーテスト ───────────────────────────────────────
class TestBuildFullDf:
    """_build_full_df のテスト。"""

    def test_basic(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        df = _build_full_df(X, ["a", "b"], {"c": 99.0}, ["a", "b", "c"])
        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 2
        assert df["c"].iloc[0] == 99.0
        assert df["a"].iloc[0] == 1.0

    def test_missing_col_fills_zero(self):
        X = np.array([[1.0]])
        df = _build_full_df(X, ["a"], {}, ["a", "b"])
        assert df["b"].iloc[0] == 0.0


class TestScorePredictions:
    """_score_predictions のテスト。"""

    def test_maximize(self):
        config = InverseConfig(target_mode="maximize")
        preds = np.array([1, 5, 3])
        scores = _score_predictions(preds, config)
        assert scores[1] > scores[0]  # 5 > 1

    def test_minimize(self):
        config = InverseConfig(target_mode="minimize")
        preds = np.array([1, 5, 3])
        scores = _score_predictions(preds, config)
        assert scores[0] > scores[1]  # -1 > -5

    def test_range(self):
        config = InverseConfig(target_mode="range", target_min=4.0, target_max=6.0)
        preds = np.array([5.0, 0.0, 10.0])
        scores = _score_predictions(preds, config)
        # 5.0は中心(5.0)にちょうど一致するのでスコア最大
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]


class TestSBXCrossover:
    """SBX交叉のテスト。"""

    def test_produces_valid_offspring(self):
        rng = np.random.RandomState(42)
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([10.0, 10.0, 10.0])
        c1, c2 = _sbx_crossover(p1, p2, lo, hi, rng)
        assert c1.shape == (3,)
        assert c2.shape == (3,)
        assert np.all(c1 >= lo) and np.all(c1 <= hi)
        assert np.all(c2 >= lo) and np.all(c2 <= hi)

    def test_identical_parents(self):
        rng = np.random.RandomState(42)
        p = np.array([3.0, 3.0])
        c1, c2 = _sbx_crossover(p, p, np.zeros(2), np.ones(2) * 10, rng)
        np.testing.assert_array_equal(c1, p)
        np.testing.assert_array_equal(c2, p)


# ─── 手法別テスト ─────────────────────────────────────────
class TestRandomMethod:
    """ランダムサンプリング手法。"""

    def test_basic(self):
        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={
                "x1": {"min": 0, "max": 10, "fixed": False, "active": True},
                "x2": {"min": 0, "max": 10, "fixed": False, "active": True},
            },
            method_params={"n_samples": 100, "seed": 42},
        )
        result = run_inverse_optimization(
            _linear_predict, ["x1", "x2"], config,
        )
        assert isinstance(result, InverseResult)
        assert len(result.candidates) > 0
        assert "predicted" in result.candidates.columns
        assert result.n_evaluated == 100

    def test_with_fixed(self):
        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={
                "x1": {"min": 0, "max": 10, "fixed": False, "active": True},
                "x2": {"min": 0, "max": 10, "fixed": True, "fixed_val": 5.0, "active": True},
            },
            method_params={"n_samples": 50, "seed": 42},
        )
        result = run_inverse_optimization(
            _linear_predict, ["x1", "x2"], config,
        )
        assert len(result.candidates) > 0
        # x2は固定なので結果に含まれない（search_colsから除外）
        assert result.n_evaluated > 0


class TestGridMethod:
    """グリッドサーチ手法。"""

    def test_basic(self):
        config = InverseConfig(
            method="grid",
            target_mode="maximize",
            constraints={
                "x1": {"min": 0, "max": 10, "fixed": False, "active": True},
                "x2": {"min": 0, "max": 10, "fixed": False, "active": True},
            },
            method_params={"n_points": 5},
        )
        result = run_inverse_optimization(
            _linear_predict, ["x1", "x2"], config,
        )
        assert result.n_evaluated == 25  # 5^2
        # 最大化: x1=10, x2=10 → y=30 が最良に近いはず
        assert result.best_predicted > 20

    def test_high_dim_caps(self):
        """高次元の場合の自動制限。"""
        constraints = {}
        for i in range(10):
            constraints[f"x{i}"] = {"min": 0, "max": 1, "fixed": False, "active": True}
        config = InverseConfig(
            method="grid",
            target_mode="maximize",
            constraints=constraints,
            method_params={"n_points": 50},
        )

        def _sum_predict(X_df):
            return X_df.sum(axis=1).values

        result = run_inverse_optimization(
            _sum_predict, [f"x{i}" for i in range(10)], config,
        )
        # 500000制限により分割数が縮小されている
        assert result.n_evaluated <= 500_001


class TestBayesianMethod:
    """ベイズ最適化手法。"""

    def test_finds_optimum(self):
        config = InverseConfig(
            method="bayesian",
            target_mode="maximize",
            constraints={
                "x1": {"min": 0, "max": 10, "fixed": False, "active": True},
                "x2": {"min": 0, "max": 6, "fixed": False, "active": True},
            },
            method_params={"n_trials": 30, "seed": 42, "acq_func": "EI"},
        )
        result = run_inverse_optimization(
            _quadratic_predict, ["x1", "x2"], config,
        )
        assert len(result.candidates) > 0
        # 最適は x1≈5, x2≈3 → 34 に近いはず
        assert result.best_predicted > 25


class TestGAMethod:
    """遺伝的アルゴリズム手法。"""

    def test_finds_optimum(self):
        config = InverseConfig(
            method="ga",
            target_mode="maximize",
            constraints={
                "x1": {"min": 0, "max": 10, "fixed": False, "active": True},
                "x2": {"min": 0, "max": 6, "fixed": False, "active": True},
            },
            method_params={
                "pop_size": 20, "n_generations": 30,
                "mutation_rate": 0.1, "crossover_rate": 0.8, "seed": 42,
            },
        )
        result = run_inverse_optimization(
            _quadratic_predict, ["x1", "x2"], config,
        )
        assert len(result.candidates) > 0
        assert result.best_predicted > 25


class TestRangeMode:
    """範囲指定モード。"""

    def test_range_target(self):
        config = InverseConfig(
            method="random",
            target_mode="range",
            target_min=14.0,
            target_max=16.0,
            constraints={
                "x1": {"min": 0, "max": 10, "fixed": False, "active": True},
                "x2": {"min": 0, "max": 10, "fixed": False, "active": True},
            },
            method_params={"n_samples": 500, "seed": 42},
        )
        result = run_inverse_optimization(
            _linear_predict, ["x1", "x2"], config,
        )
        # 上位候補の predicted は 14~16 付近のはず
        top_pred = result.candidates["predicted"].iloc[0]
        assert 12 < top_pred < 18


class TestEdgeCases:
    """エッジケース。"""

    def test_no_active_vars_raises(self):
        config = InverseConfig(
            method="random",
            constraints={"x1": {"active": False}},
        )
        with pytest.raises(ValueError, match="探索対象"):
            run_inverse_optimization(
                _linear_predict, ["x1"], config,
            )

    def test_invalid_method_raises(self):
        config = InverseConfig(
            method="nonexistent",
            constraints={"x1": {"min": 0, "max": 1, "fixed": False, "active": True}},
        )
        with pytest.raises(ValueError, match="未対応"):
            run_inverse_optimization(
                _linear_predict, ["x1"], config,
            )

    def test_min_equals_max(self):
        """min == max の場合も動作する。"""
        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={
                "x1": {"min": 5, "max": 5, "fixed": False, "active": True},
                "x2": {"min": 0, "max": 10, "fixed": False, "active": True},
            },
            method_params={"n_samples": 10, "seed": 42},
        )
        result = run_inverse_optimization(
            _linear_predict, ["x1", "x2"], config,
        )
        assert len(result.candidates) > 0
