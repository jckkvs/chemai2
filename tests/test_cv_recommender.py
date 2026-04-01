"""
backend/utils/cv_recommender テスト.

テストID対応:
    T-CVR01: recommend_cv_strategy — 通常回帰データ
    T-CVR02: recommend_cv_strategy — 時系列データ（列名パターン）
    T-CVR03: recommend_cv_strategy — 時系列データ（単調増加列）
    T-CVR04: recommend_cv_strategy — グループ指定あり
    T-CVR05: recommend_cv_strategy — クラス不均衡
    T-CVR06: recommend_cv_strategy — 小サンプル（LOO）
    T-CVR07: recommend_cv_strategy — 小サンプル（RepeatedKFold）
    T-CVR08: _detect_timeseries — 列名パターン網羅
    T-CVR09: _detect_imbalance — 比率境界テスト
    T-CVR10: _assess_sample_size — カテゴリ境界テスト
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.utils.cv_recommender import (
    CVRecommendation,
    recommend_cv_strategy,
    _detect_timeseries,
    _detect_groups,
    _detect_imbalance,
    _assess_sample_size,
)


# ============================================================
# フィクスチャ
# ============================================================

@pytest.fixture
def normal_regression_data():
    """T-CVR01: 標準的な回帰データ (100サンプル, 5特徴量)"""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = pd.Series(rng.randn(100), name="target")
    return X, y


@pytest.fixture
def timeseries_data_by_name():
    """T-CVR02: 列名に 'date' を含む時系列データ"""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
    })
    y = pd.Series(rng.randn(n))
    return X, y


@pytest.fixture
def timeseries_data_monotonic():
    """T-CVR03: 等間隔単調増加列（列名に時系列キーワードなし）"""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame({
        "index_val": np.arange(n),  # 完全な等間隔単調増加
        "sensor_a": rng.randn(n),
        "sensor_b": rng.randn(n),
    })
    y = pd.Series(rng.randn(n))
    return X, y


@pytest.fixture
def grouped_data():
    """T-CVR04: グループ構造のあるデータ"""
    rng = np.random.RandomState(42)
    n = 100
    groups = np.repeat(["A", "B", "C", "D", "E"], 20)
    X = pd.DataFrame({
        "group_id": groups,
        "feat_1": rng.randn(n),
        "feat_2": rng.randn(n),
    })
    y = pd.Series(rng.randn(n))
    return X, y, groups


@pytest.fixture
def imbalanced_classification_data():
    """T-CVR05: クラス不均衡の分類データ"""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
    # 90% がクラス 0, 10% がクラス 1 → 比率 9:1
    y = pd.Series([0] * 180 + [1] * 20)
    return X, y


@pytest.fixture
def very_small_data():
    """T-CVR06: 非常に小さなデータ (15サンプル)"""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(15, 3), columns=["a", "b", "c"])
    y = pd.Series(rng.randn(15))
    return X, y


@pytest.fixture
def small_data():
    """T-CVR07: 小さなデータ (40サンプル)"""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(40, 3), columns=["a", "b", "c"])
    y = pd.Series(rng.randn(40))
    return X, y


# ============================================================
# T-CVR01: 通常回帰データ → KFold推奨
# ============================================================

class TestNormalRegression:
    def test_recommends_kfold(self, normal_regression_data):
        X, y = normal_regression_data
        rec = recommend_cv_strategy(X, y)
        assert isinstance(rec, CVRecommendation)
        assert rec.recommended_cv == "kfold"
        assert rec.confidence > 0
        assert "n_splits" in rec.recommended_params
        assert rec.recommended_params["n_splits"] in (3, 5, 10)

    def test_result_has_all_fields(self, normal_regression_data):
        X, y = normal_regression_data
        rec = recommend_cv_strategy(X, y)
        assert rec.reason != ""
        assert isinstance(rec.alternative_cvs, list)
        assert isinstance(rec.detected_features, dict)
        assert "n_samples" in rec.detected_features


# ============================================================
# T-CVR02: 時系列データ（列名パターン）
# ============================================================

class TestTimeseriesByName:
    def test_detects_timeseries_column(self, timeseries_data_by_name):
        X, y = timeseries_data_by_name
        rec = recommend_cv_strategy(X, y)
        assert rec.recommended_cv == "timeseries"
        assert rec.confidence >= 0.70
        assert "時系列" in rec.reason or "TimeSeriesSplit" in rec.reason


# ============================================================
# T-CVR03: 時系列データ（単調増加列）
# ============================================================

class TestTimeseriesMonotonic:
    def test_detects_monotonic_column(self, timeseries_data_monotonic):
        X, y = timeseries_data_monotonic
        rec = recommend_cv_strategy(X, y)
        assert rec.recommended_cv == "timeseries"
        assert rec.confidence >= 0.60


# ============================================================
# T-CVR04: グループ指定あり → GroupKFold/LOGO推奨
# ============================================================

class TestGroupedData:
    def test_recommends_group_cv(self, grouped_data):
        X, y, groups = grouped_data
        rec = recommend_cv_strategy(X, y, metadata={"group_col": "group_id"})
        assert rec.recommended_cv in ("group_kfold", "logo")
        assert rec.confidence >= 0.80
        assert "グループ" in rec.reason

    def test_logo_for_few_groups(self, grouped_data):
        """5グループ以下 → LOGO推奨"""
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "group_id": np.repeat(["A", "B", "C"], 10),
            "feat": rng.randn(30),
        })
        y = pd.Series(rng.randn(30))
        rec = recommend_cv_strategy(X, y, metadata={"group_col": "group_id"})
        assert rec.recommended_cv == "logo"


# ============================================================
# T-CVR05: クラス不均衡 → StratifiedKFold推奨
# ============================================================

class TestImbalancedClassification:
    def test_recommends_stratified(self, imbalanced_classification_data):
        X, y = imbalanced_classification_data
        rec = recommend_cv_strategy(X, y, metadata={"task_type": "classification"})
        assert rec.recommended_cv == "stratified_kfold"
        assert rec.confidence >= 0.70
        assert "不均衡" in rec.reason


# ============================================================
# T-CVR06: 非常に小さなデータ → LOO推奨
# ============================================================

class TestVerySmallData:
    def test_recommends_loo(self, very_small_data):
        X, y = very_small_data
        rec = recommend_cv_strategy(X, y)
        assert rec.recommended_cv == "loo"
        assert len(rec.warnings) > 0  # 計算コスト警告


# ============================================================
# T-CVR07: 小さなデータ → RepeatedKFold推奨
# ============================================================

class TestSmallData:
    def test_recommends_repeated_kfold(self, small_data):
        X, y = small_data
        rec = recommend_cv_strategy(X, y)
        assert rec.recommended_cv == "repeated_kfold"
        assert rec.recommended_params.get("n_repeats", 0) > 1


# ============================================================
# T-CVR08: _detect_timeseries — 列名パターン網羅
# ============================================================

class TestDetectTimeseries:
    @pytest.mark.parametrize("col_name", [
        "date", "Date", "DATE",
        "timestamp", "datetime",
        "year", "month",
        "created_at", "updated_at",
        "日付", "日時", "年月日",
    ])
    def test_detects_various_patterns(self, col_name):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            col_name: range(50),
            "feature": rng.randn(50),
        })
        result = _detect_timeseries(X, {})
        assert result["is_timeseries"], f"'{col_name}' should be detected as timeseries"

    def test_no_false_positive(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "feature_a": rng.randn(50),
            "feature_b": rng.randn(50),
        })
        result = _detect_timeseries(X, {})
        assert not result["is_timeseries"]


# ============================================================
# T-CVR09: _detect_imbalance — 比率境界テスト
# ============================================================

class TestDetectImbalance:
    def test_balanced(self):
        y = pd.Series([0] * 50 + [1] * 50)
        result = _detect_imbalance(y)
        assert not result["is_imbalanced"]

    def test_moderate_imbalance(self):
        """3:1超 → 不均衡判定"""
        y = pd.Series([0] * 80 + [1] * 20)
        result = _detect_imbalance(y)
        assert result["is_imbalanced"]
        assert result["imbalance_ratio"] == 4.0

    def test_severe_imbalance(self):
        """10:1超 → 高度不均衡"""
        y = pd.Series([0] * 110 + [1] * 10)
        result = _detect_imbalance(y)
        assert result["is_imbalanced"]
        assert result["confidence"] >= 0.90


# ============================================================
# T-CVR10: _assess_sample_size — カテゴリ境界テスト
# ============================================================

class TestAssessSampleSize:
    def test_very_small(self):
        result = _assess_sample_size(15, 3)
        assert result["is_small"]
        assert result["category"] == "very_small"

    def test_small(self):
        result = _assess_sample_size(40, 3)
        assert result["is_small"]
        assert result["category"] == "small"

    def test_high_dimensional(self):
        result = _assess_sample_size(100, 200)
        assert result["is_small"]
        assert result["category"] == "high_dim"

    def test_normal(self):
        result = _assess_sample_size(500, 10)
        assert not result["is_small"]
        assert result["category"] == "normal"


# ============================================================
# numpy配列入力の互換性テスト
# ============================================================

class TestNumpyInput:
    def test_numpy_arrays(self):
        rng = np.random.RandomState(42)
        X = rng.randn(80, 4)
        y = rng.randn(80)
        rec = recommend_cv_strategy(X, y)
        assert isinstance(rec, CVRecommendation)
        assert rec.recommended_cv in ("kfold", "repeated_kfold")
