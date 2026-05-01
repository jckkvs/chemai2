"""
tests/test_optim_comprehensive.py

optim/constraints.py + optim/search_space.py の包括テスト。
全制約型（Range/Sum/Inequality/AtLeastN/Custom）+ apply_constraints と
Variable/SearchSpace/全生成メソッドを網羅。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.optim.constraints import (
    RangeConstraint,
    SumConstraint,
    InequalityConstraint,
    AtLeastNConstraint,
    CustomConstraint,
    apply_constraints,
)
from backend.optim.search_space import (
    VarType,
    Variable,
    SearchSpace,
)


# ============================================================
# Constraint テスト
# ============================================================

class TestRangeConstraint:
    def test_in_range(self):
        c = RangeConstraint(column="x", lo=0, hi=10)
        row = pd.Series({"x": 5})
        assert c.is_satisfied(row)

    def test_below(self):
        c = RangeConstraint(column="x", lo=0, hi=10)
        row = pd.Series({"x": -1})
        assert not c.is_satisfied(row)

    def test_above(self):
        c = RangeConstraint(column="x", lo=0, hi=10)
        row = pd.Series({"x": 11})
        assert not c.is_satisfied(row)

    def test_mask(self):
        c = RangeConstraint(column="x", lo=0, hi=5)
        df = pd.DataFrame({"x": [-1, 0, 3, 5, 6]})
        m = c.mask(df)
        assert m.tolist() == [False, True, True, True, False]

    def test_describe(self):
        c = RangeConstraint(column="x", lo=0, hi=10)
        assert "x" in c.describe()

    def test_lo_only(self):
        c = RangeConstraint(column="x", lo=5)
        assert c.is_satisfied(pd.Series({"x": 10}))
        assert not c.is_satisfied(pd.Series({"x": 3}))

    def test_hi_only(self):
        c = RangeConstraint(column="x", hi=5)
        assert c.is_satisfied(pd.Series({"x": 3}))
        assert not c.is_satisfied(pd.Series({"x": 10}))


class TestSumConstraint:
    def test_exact(self):
        c = SumConstraint(columns=["a", "b", "c"], target=100, tolerance=1e-6)
        row = pd.Series({"a": 30, "b": 30, "c": 40})
        assert c.is_satisfied(row)

    def test_not_exact(self):
        c = SumConstraint(columns=["a", "b"], target=100)
        row = pd.Series({"a": 30, "b": 30})
        assert not c.is_satisfied(row)

    def test_mask(self):
        c = SumConstraint(columns=["a", "b"], target=10, tolerance=0.5)
        df = pd.DataFrame({"a": [5, 9, 3], "b": [5, 1, 2]})
        m = c.mask(df)
        assert m.tolist() == [True, True, False]

    def test_describe(self):
        c = SumConstraint(columns=["a", "b"], target=100)
        assert "a" in c.describe()


class TestInequalityConstraint:
    def test_le(self):
        c = InequalityConstraint(coefficients={"x": 1, "y": 1}, rhs=10, operator="le")
        assert c.is_satisfied(pd.Series({"x": 3, "y": 5}))
        assert not c.is_satisfied(pd.Series({"x": 8, "y": 5}))

    def test_ge(self):
        c = InequalityConstraint(coefficients={"x": 1}, rhs=5, operator="ge")
        assert c.is_satisfied(pd.Series({"x": 10}))
        assert not c.is_satisfied(pd.Series({"x": 3}))

    def test_lt(self):
        c = InequalityConstraint(coefficients={"x": 1}, rhs=5, operator="lt")
        assert c.is_satisfied(pd.Series({"x": 4}))
        assert not c.is_satisfied(pd.Series({"x": 5}))

    def test_gt(self):
        c = InequalityConstraint(coefficients={"x": 1}, rhs=5, operator="gt")
        assert c.is_satisfied(pd.Series({"x": 6}))

    def test_mask(self):
        c = InequalityConstraint(coefficients={"x": 2, "y": 1}, rhs=10, operator="le")
        df = pd.DataFrame({"x": [1, 5, 3], "y": [1, 5, 5]})
        m = c.mask(df)
        assert m.tolist() == [True, False, False]

    def test_describe(self):
        c = InequalityConstraint(coefficients={"x": 1.0, "y": -1.0}, rhs=5, operator="le")
        desc = c.describe()
        assert "x" in desc


class TestAtLeastNConstraint:
    def test_satisfied(self):
        c = AtLeastNConstraint(columns=["a", "b", "c"], min_count=2, threshold=0)
        row = pd.Series({"a": 1, "b": 2, "c": 0})
        assert c.is_satisfied(row)

    def test_not_satisfied(self):
        c = AtLeastNConstraint(columns=["a", "b", "c"], min_count=3, threshold=0)
        row = pd.Series({"a": 1, "b": 0, "c": 0})
        assert not c.is_satisfied(row)

    def test_mask(self):
        c = AtLeastNConstraint(columns=["a", "b"], min_count=1, threshold=5)
        df = pd.DataFrame({"a": [0, 10, 3], "b": [0, 0, 6]})
        m = c.mask(df)
        assert m.tolist() == [False, True, True]


class TestCustomConstraint:
    def test_eval(self):
        c = CustomConstraint(expression="A * B <= 50")
        assert c.is_satisfied(pd.Series({"A": 5, "B": 5}))
        assert not c.is_satisfied(pd.Series({"A": 10, "B": 10}))

    def test_mask(self):
        c = CustomConstraint(expression="x > 3")
        df = pd.DataFrame({"x": [1, 5, 10]})
        m = c.mask(df)
        assert m.tolist() == [False, True, True]

    def test_describe(self):
        c = CustomConstraint(expression="A + B == 100")
        assert "カスタム" in c.describe()


class TestApplyConstraints:
    def test_multiple(self):
        df = pd.DataFrame({"x": [1, 5, 10, 15], "y": [4, 5, 0, 5]})
        constraints = [
            RangeConstraint(column="x", lo=0, hi=12),
            RangeConstraint(column="y", lo=1, hi=10),
        ]
        filtered, report = apply_constraints(df, constraints)
        assert report["before"] == 4
        assert len(filtered) == report["after"]
        assert report["removed"] == report["before"] - report["after"]

    def test_no_constraints(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        filtered, report = apply_constraints(df, [])
        assert len(filtered) == 3


# ============================================================
# Variable テスト
# ============================================================

class TestVariable:
    def test_continuous(self):
        v = Variable("x", VarType.CONTINUOUS, lo=0, hi=10)
        assert v.n_levels == 20
        vals = v.grid_values(10)
        assert len(vals) == 10

    def test_discrete(self):
        v = Variable("d", VarType.DISCRETE, lo=0, hi=10, step=2)
        vals = v.grid_values()
        assert all(v_i in [0, 2, 4, 6, 8, 10] for v_i in vals)

    def test_categorical(self):
        v = Variable("c", VarType.CATEGORICAL, categories=["a", "b", "c"])
        assert v.n_levels == 3
        vals = v.grid_values()
        assert list(vals) == ["a", "b", "c"]

    def test_validation_no_lo(self):
        with pytest.raises(ValueError, match="lo/hi"):
            Variable("x", VarType.CONTINUOUS, lo=None, hi=10)

    def test_validation_lo_gt_hi(self):
        with pytest.raises(ValueError, match="lo"):
            Variable("x", VarType.CONTINUOUS, lo=10, hi=0)

    def test_validation_discrete_no_step(self):
        with pytest.raises(ValueError, match="step"):
            Variable("x", VarType.DISCRETE, lo=0, hi=10)

    def test_validation_categorical_no_categories(self):
        with pytest.raises(ValueError, match="categories"):
            Variable("x", VarType.CATEGORICAL)


# ============================================================
# SearchSpace テスト
# ============================================================

class TestSearchSpace:
    @pytest.fixture
    def simple_space(self):
        return SearchSpace([
            Variable("x", VarType.CONTINUOUS, lo=0, hi=1),
            Variable("y", VarType.DISCRETE, lo=0, hi=10, step=5),
        ])

    def test_dim(self, simple_space):
        assert simple_space.dim == 2

    def test_names(self, simple_space):
        assert simple_space.names == ["x", "y"]

    def test_grid(self, simple_space):
        df = simple_space.generate_candidates(method="grid", n_per_dim=5)
        assert len(df) == 5 * 3  # 5 continuous x 3 discrete (0,5,10)

    def test_random(self, simple_space):
        df = simple_space.generate_candidates(method="random", n_max=100)
        assert len(df) == 100
        assert df["x"].min() >= 0
        assert df["x"].max() <= 1

    def test_lhs(self, simple_space):
        df = simple_space.generate_candidates(method="lhs", n_max=50)
        assert len(df) == 50

    def test_auto(self, simple_space):
        df = simple_space.generate_candidates(method="auto", n_per_dim=5)
        assert len(df) > 0

    def test_grid_downsample(self, simple_space):
        df = simple_space.generate_candidates(method="grid_downsample", n_per_dim=5, n_max=10)
        assert len(df) <= 10

    def test_random_lhs(self, simple_space):
        df = simple_space.generate_candidates(method="random_lhs", n_max=100)
        assert len(df) == 100

    def test_invalid_method(self, simple_space):
        with pytest.raises(ValueError, match="不明な"):
            simple_space.generate_candidates(method="invalid")

    def test_no_variables(self):
        with pytest.raises(ValueError, match="変数"):
            SearchSpace().generate_candidates()

    def test_estimate_grid_size(self, simple_space):
        size = simple_space.estimate_grid_size(n_per_dim=10)
        assert size == 10 * 3

    def test_auto_recommend(self, simple_space):
        method = simple_space.auto_recommend_method(n_per_dim=5)
        assert method in ("grid", "grid_downsample", "random_lhs")

    def test_from_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})
        space = SearchSpace.from_dataframe(df, margin=0.1)
        assert space.dim == 2

    def test_add_variable(self):
        space = SearchSpace()
        space.add(Variable("z", VarType.CONTINUOUS, lo=0, hi=1))
        assert space.dim == 1

    def test_with_categorical(self):
        space = SearchSpace([
            Variable("x", VarType.CONTINUOUS, lo=0, hi=1),
            Variable("c", VarType.CATEGORICAL, categories=["a", "b"]),
        ])
        df = space.generate_candidates(method="grid", n_per_dim=5)
        assert "c" in df.columns
        assert set(df["c"].unique()) == {"a", "b"}


# ============================================================
# T-008: tabular_50_safe.csv を使った実験計画法(DOE)統合テスト
# ============================================================

class TestTabular50SafeDOE:
    """T-008: tabular_50_safe.csv を使った実験計画法のテスト。"""

    @pytest.fixture
    def tabular_df(self) -> pd.DataFrame:
        """tabular_50_safe.csv を読み込む。"""
        from pathlib import Path
        from backend.data.loader import load_file
        return load_file(Path("data/samples/tabular_50_safe.csv"))

    @pytest.fixture
    def factors(self) -> list:
        """tabular_50_safe.csv の特徴量を因子として定義する。"""
        from backend.doe.factor import Factor

        # Feature_1〜8 を連続値因子として定義
        factors = []
        for i in range(1, 9):
            col = f"Feature_{i}"
            factors.append(Factor.continuous(name=col, low=-3.0, high=3.0, n_levels=5))
        return factors

    def test_doe_d_optimality(self, factors) -> None:
        """D最適化（D-optimality）が正しく動作すること。(T-008-01)"""
        from backend.doe.design import DoEOptimizer

        optimizer = DoEOptimizer(
            factors=factors,
            n_new=10,
            criterion="D",
            max_candidates=1000,
            random_seed=42,
            n_starts=3,
            max_iter=100,
        )
        result = optimizer.optimize()

        assert result is not None
        assert hasattr(result, 'design_df')
        assert len(result.design_df) > 0
        assert result.criterion_name == "D"

    def test_doe_maximin(self, factors) -> None:
        """Maximin 基準が正しく動作すること。(T-008-02)"""
        from backend.doe.design import DoEOptimizer

        optimizer = DoEOptimizer(
            factors=factors,
            n_new=10,
            criterion="MAXIMIN",
            max_candidates=1000,
            random_seed=42,
            n_starts=3,
            max_iter=100,
        )
        result = optimizer.optimize()

        assert result is not None
        assert hasattr(result, 'design_df')
        assert len(result.design_df) > 0
        assert result.criterion_name == "MAXIMIN"

    def test_doe_minimax(self, factors) -> None:
        """Minimax 基準が正しく動作すること。(T-008-03)"""
        from backend.doe.design import DoEOptimizer

        optimizer = DoEOptimizer(
            factors=factors,
            n_new=10,
            criterion="MINIMAX",
            max_candidates=1000,
            random_seed=42,
            n_starts=3,
            max_iter=100,
        )
        result = optimizer.optimize()

        assert result is not None
        assert hasattr(result, 'design_df')
        assert len(result.design_df) > 0
        assert result.criterion_name == "MINIMAX"

    def test_doe_with_existing_data(self, factors, tabular_df) -> None:
        """既存データを含めて最適化が正しく動作すること。(T-008-04)"""
        from backend.doe.design import DoEOptimizer

        optimizer = DoEOptimizer(
            factors=factors,
            n_new=5,
            criterion="D",
            existing_df=tabular_df,
            max_candidates=1000,
            random_seed=42,
            n_starts=3,
            max_iter=100,
        )
        result = optimizer.optimize()

        assert result is not None
        assert hasattr(result, 'design_df')
        # 既存データ + 新規5点
        assert len(result.design_df) >= 5
        assert hasattr(result, 'is_new')
        assert sum(result.is_new) == 5

    def test_doe_e_optimality(self, factors) -> None:
        """E最適化（E-optimality）が正しく動作すること。(T-008-05)"""
        from backend.doe.design import DoEOptimizer

        optimizer = DoEOptimizer(
            factors=factors,
            n_new=10,
            criterion="E",
            max_candidates=1000,
            random_seed=42,
            n_starts=3,
            max_iter=100,
        )
        result = optimizer.optimize()

        assert result is not None
        assert result.criterion_name == "E"

    def test_doe_i_optimality(self, factors) -> None:
        """I最適化（I-optimality）が正しく動作すること。(T-008-06)"""
        from backend.doe.design import DoEOptimizer

        optimizer = DoEOptimizer(
            factors=factors,
            n_new=10,
            criterion="I",
            max_candidates=1000,
            random_seed=42,
            n_starts=3,
            max_iter=100,
        )
        result = optimizer.optimize()

        assert result is not None
        assert result.criterion_name == "I"
