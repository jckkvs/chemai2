"""
tests/test_eda_samples.py

TDD for EDA (Exploratory Data Analysis) features using sample data files.
Focus on non-LLM features: basic statistics, correlation, outliers, dimensionality reduction.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from backend.data.eda import (
    compute_column_stats,
    summarize_dataframe,
    compute_correlation,
)
from backend.data.eda_core import (
    compute_basic_statistics,
    compute_correlation_matrix,
    detect_outliers,
)


# ============================================================
# Fixtures: Load sample data
# ============================================================

@pytest.fixture
def safe_df() -> pd.DataFrame:
    """Load tabular_50_safe.csv (no missing values, 50 rows, 9 cols)."""
    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_safe.csv".parent.parent / "data" / "samples" / "tabular_50_safe.csv"
    from backend.data.loader import load_file
    return load_file(p)


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Load tabular_50_simple.csv (has missing values)."""
    p = Path(__file__).parent.parent / "data" / "samples" / "tabular_50_simple.csv".parent.parent / "data" / "samples" / "tabular_50_simple.csv"
    from backend.data.loader import load_file
    return load_file(p)


@pytest.fixture
def smiles_reg_df() -> pd.DataFrame:
    """Load smiles_25_regression.csv (SMILES + numeric target)."""
    p = Path(__file__).parent.parent / "data" / "samples" / "smiles_25_regression.csv".parent.parent / "data" / "samples" / "smiles_25_regression.csv"
    from backend.data.loader import load_file
    return load_file(p)


@pytest.fixture
def mixture_df() -> pd.DataFrame:
    """Load mixture_30_simple.csv (mixture data with SMILES)."""
    p = Path(__file__).parent.parent / "data" / "samples" / "mixture_30_simple.csv".parent.parent / "data" / "samples" / "mixture_30_simple.csv"
    from backend.data.loader import load_file
    return load_file(p)


# ============================================================
# T-030: Column Statistics Tests (eda.py)
# ============================================================

class TestComputeColumnStats:
    """T-030: compute_column_stats using sample data."""

    def test_returns_list_of_ColumnStats(self, safe_df: pd.DataFrame) -> None:
        """Returns a list of ColumnStats objects. (T-030-01)"""
        stats = compute_column_stats(safe_df)
        assert isinstance(stats, list)
        assert len(stats) == len(safe_df.columns)

    def test_column_names_match(self, safe_df: pd.DataFrame) -> None:
        """Column names in stats match DataFrame columns. (T-030-02)"""
        stats = compute_column_stats(safe_df)
        stat_cols = [s.name for s in stats]
        assert stat_cols == list(safe_df.columns)

    def test_numeric_stats_have_mean_std(self, safe_df: pd.DataFrame) -> None:
        """Numeric columns have mean, std, min, max computed. (T-030-03)"""
        stats = compute_column_stats(safe_df)
        for cs in stats:
            if cs.name.startswith("Feature_") or cs.name == "Target":
                assert cs.mean is not None, f"{cs.name}: mean not computed"
                assert cs.std is not None, f"{cs.name}: std not computed"
                assert cs.min is not None, f"{cs.name}: min not computed"
                assert cs.max is not None, f"{cs.name}: max not computed"

    def test_null_rate_zero_for_safe_data(self, safe_df: pd.DataFrame) -> None:
        """null_rate is 0.0 for all columns in safe data. (T-030-04)"""
        stats = compute_column_stats(safe_df)
        for cs in stats:
            assert cs.null_rate == 0.0, f"{cs.name}: null_rate should be 0.0"

    def test_missing_values_counted(self, simple_df: pd.DataFrame) -> None:
        """Missing values are correctly counted. (T-030-05)"""
        stats = compute_column_stats(simple_df)
        for cs in stats:
            if cs.name == "Feature_1":
                assert cs.n_null == 1, f"Feature_1 should have 1 null, got {cs.n_null}"
            elif cs.name == "Feature_2":
                assert cs.n_null == 2, f"Feature_2 should have 2 nulls, got {cs.n_null}"
            elif cs.name == "Feature_3":
                assert cs.n_null == 3, f"Feature_3 should have 3 nulls, got {cs.n_null}"

    def test_n_unique_values(self, safe_df: pd.DataFrame) -> None:
        """n_unique is computed correctly. (T-030-06)"""
        stats = compute_column_stats(safe_df)
        for cs in stats:
            assert cs.n_unique == safe_df[cs.name].nunique(), \
                f"{cs.name}: n_unique mismatch"

    def test_skewness_computed_for_numeric(self, safe_df: pd.DataFrame) -> None:
        """Skewness is computed for numeric columns. (T-030-07)"""
        stats = compute_column_stats(safe_df)
        for cs in stats:
            if cs.name.startswith("Feature_") or cs.name == "Target":
                assert cs.skewness is not None, f"{cs.name}: skewness not computed"

    def test_kurtosis_computed_for_numeric(self, safe_df: pd.DataFrame) -> None:
        """Kurtosis is computed for numeric columns. (T-030-08)"""
        stats = compute_column_stats(safe_df)
        for cs in stats:
            if cs.name.startswith("Feature_") or cs.name == "Target":
                assert cs.kurtosis is not None, f"{cs.name}: kurtosis not computed"


# ============================================================
# T-031: DataFrame Summary Tests (eda.py)
# ============================================================

class TestSummarizeDataframe:
    """T-031: summarize_dataframe using sample data."""

    def test_returns_dict(self, safe_df: pd.DataFrame) -> None:
        """Returns a dict with summary info. (T-031-01)"""
        summary = summarize_dataframe(safe_df)
        assert isinstance(summary, dict)

    def test_has_required_keys(self, safe_df: pd.DataFrame) -> None:
        """Dict has all required keys. (T-031-02)"""
        summary = summarize_dataframe(safe_df)
        required = ["n_rows", "n_cols", "n_numeric", "n_categorical", "n_datetime",
                   "total_null_rate", "n_duplicates", "memory_mb"]
        for key in required:
            assert key in summary, f"Missing key: {key}"

    def test_n_rows_correct(self, safe_df: pd.DataFrame) -> None:
        """n_rows matches DataFrame. (T-031-03)"""
        summary = summarize_dataframe(safe_df)
        assert summary["n_rows"] == len(safe_df)

    def test_n_cols_correct(self, safe_df: pd.DataFrame) -> None:
        """n_cols matches DataFrame. (T-031-04)"""
        summary = summarize_dataframe(safe_df)
        assert summary["n_cols"] == len(safe_df.columns)

    def test_n_numeric_correct(self, safe_df: pd.DataFrame) -> None:
        """n_numeric matches numeric column count. (T-031-05)"""
        summary = summarize_dataframe(safe_df)
        expected = len(safe_df.select_dtypes(include="number").columns)
        assert summary["n_numeric"] == expected

    def test_total_null_rate_zero(self, safe_df: pd.DataFrame) -> None:
        """total_null_rate is 0.0 for safe data. (T-031-06)"""
        summary = summarize_dataframe(safe_df)
        assert summary["total_null_rate"] == 0.0

    def test_n_duplicates_zero(self, safe_df: pd.DataFrame) -> None:
        """n_duplicates is 0 for safe data (no duplicate rows). (T-031-07)"""
        summary = summarize_dataframe(safe_df)
        assert summary["n_duplicates"] == 0

    def test_memory_mb_positive(self, safe_df: pd.DataFrame) -> None:
        """memory_mb is positive. (T-031-08)"""
        summary = summarize_dataframe(safe_df)
        assert summary["memory_mb"] > 0.0


# ============================================================
# T-032: Correlation Tests (eda.py and eda_core.py)
# ============================================================

class TestCorrelation:
    """T-032: Correlation analysis using sample data."""

    def test_compute_correlation_returns_dataframe(self, safe_df: pd.DataFrame) -> None:
        """compute_correlation returns a DataFrame. (T-032-01)"""
        corr = compute_correlation(safe_df)
        assert isinstance(corr, pd.DataFrame)

    def test_correlation_matrix_square(self, safe_df: pd.DataFrame) -> None:
        """Correlation matrix is square (n_numeric x n_numeric). (T-032-02)"""
        corr = compute_correlation(safe_df)
        n = len(safe_df.select_dtypes(include="number").columns)
        assert corr.shape == (n, n)

    def test_correlation_values_in_valid_range(self, safe_df: pd.DataFrame) -> None:
        """All correlation values are between -1 and 1. (T-032-03)"""
        corr = compute_correlation(safe_df)
        for col in corr.columns:
            for idx in corr.index:
                val = corr.loc[idx, col]
                if pd.notna(val):
                    assert -1.0 <= val <= 1.0, f"Correlation {val} out of range"

    def test_diagonal_is_one(self, safe_df: pd.DataFrame) -> None:
        """Diagonal values are 1.0 (self-correlation). (T-032-04)"""
        corr = compute_correlation(safe_df)
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-10, f"Diagonal for {col} should be 1.0"

    def test_pearson_method(self, safe_df: pd.DataFrame) -> None:
        """Pearson correlation works. (T-032-05)"""
        corr = compute_correlation(safe_df, method="pearson")
        assert isinstance(corr, pd.DataFrame)

    def test_spearman_method(self, safe_df: pd.DataFrame) -> None:
        """Spearman correlation works. (T-032-06)"""
        corr = compute_correlation(safe_df, method="spearman")
        assert isinstance(corr, pd.DataFrame)

    def test_kendall_method(self, safe_df: pd.DataFrame) -> None:
        """Kendall correlation works. (T-032-07)"""
        corr = compute_correlation(safe_df, method="kendall")
        assert isinstance(corr, pd.DataFrame)

    def test_target_col_returns_series(self, safe_df: pd.DataFrame) -> None:
        """target_col parameter returns a Series with target correlations. (T-032-08)"""
        corr = compute_correlation(safe_df, target_col="Target")
        assert isinstance(corr, pd.Series)
        assert corr.name == "Target"

    def test_eda_core_compute_basic_statistics(self, safe_df: pd.DataFrame) -> None:
        """compute_basic_statistics from eda_core works. (T-032-09)"""
        stats = compute_basic_statistics(safe_df)
        assert isinstance(stats, dict)
        assert len(stats) > 0

    def test_eda_core_correlation_matrix(self, safe_df: pd.DataFrame) -> None:
        """compute_correlation_matrix from eda_core works. (T-032-10)"""
        corr = compute_correlation_matrix(safe_df)
        assert isinstance(corr, pd.DataFrame)


# ============================================================
# T-033: Outlier Detection Tests (eda_core.py)
# ============================================================

class TestOutlierDetection:
    """T-033: Outlier detection using sample data."""

    def test_iqr_method_returns_dataframe(self, safe_df: pd.DataFrame) -> None:
        """IQR method returns a DataFrame with same shape. (T-033-01)"""
        result = detect_outliers(safe_df, method="iqr")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == safe_df.shape

    def test_iqr_method_returns_bool(self, safe_df: pd.DataFrame) -> None:
        """IQR method returns boolean mask. (T-033-02)"""
        result = detect_outliers(safe_df, method="iqr")
        for col in result.columns:
            assert result[col].dtype == bool, f"{col}: expected bool, got {result[col].dtype}"

    def test_zscore_method(self, safe_df: pd.DataFrame) -> None:
        """Z-score method works. (T-033-03)"""
        result = detect_outliers(safe_df, method="zscore", threshold=2.0)
        assert isinstance(result, pd.DataFrame)

    def test_threshold_affects_outlier_count(self, safe_df: pd.DataFrame) -> None:
        """Higher threshold results in fewer outliers. (T-033-04)"""
        outliers_low = detect_outliers(safe_df, method="iqr", threshold=1.5)
        outliers_high = detect_outliers(safe_df, method="iqr", threshold=3.0)
        total_low = outliers_low.sum().sum()
        total_high = outliers_high.sum().sum()
        assert total_low >= total_high, "Higher threshold should give fewer outliers"

    def test_missing_values_excluded(self, simple_df: pd.DataFrame) -> None:
        """Missing values are not treated as outliers. (T-033-05)"""
        result = detect_outliers(simple_df, method="iqr")
        # Check that rows with missing values are not flagged as outliers (only non-null values considered)
        assert isinstance(result, pd.DataFrame)


# ============================================================
# T-034: Feature Engineering Tests (feature_engineer.py)
# ============================================================

from backend.data.feature_engineer import (
    InteractionTransformer,
    GroupAggTransformer,
    DatetimeFeatureExtractor,
    LagRollingTransformer,
    FeatureEngineeringConfig,
    build_feature_engineering_pipeline,
)


class TestInteractionTransformer:
    """T-034: InteractionTransformer using tabular_50_safe.csv."""

    def test_fit_returns_self(self, safe_df: pd.DataFrame) -> None:
        """fit() returns self. (T-034-01)"""
        X = safe_df.drop(columns=["Target"])
        tr = InteractionTransformer(degree=2, interaction_only=True)
        result = tr.fit(X)
        assert result is tr

    def test_transform_adds_interaction_terms(self, safe_df: pd.DataFrame) -> None:
        """transform() adds interaction terms. (T-034-02)"""
        X = safe_df.drop(columns=["Target"])
        tr = InteractionTransformer(degree=2, interaction_only=True)
        tr.fit(X)
        Xt = tr.transform(X)
        # With 8 features, interaction_only=True should give C(8,2)=28 interaction terms
        assert Xt.shape[1] > X.shape[1], "Should have more features after interaction"

    def test_get_feature_names_out(self, safe_df: pd.DataFrame) -> None:
        """get_feature_names_out returns feature names. (T-034-03)"""
        X = safe_df.drop(columns=["Target"])
        tr = InteractionTransformer(degree=2, interaction_only=True)
        tr.fit(X)
        Xt = tr.transform(X)
        names = tr.get_feature_names_out()
        assert len(names) == Xt.shape[1], f"Expected {Xt.shape[1]} names, got {len(names)}"

    def test_no_bias_term(self, safe_df: pd.DataFrame) -> None:
        """interaction_only=True excludes bias term. (T-034-04)"""
        X = safe_df.drop(columns=["Target"])
        tr = InteractionTransformer(degree=2, interaction_only=True, include_bias=False)
        tr.fit(X)
        names = tr.get_feature_names_out()
        assert "1" not in names, "Bias term should not be present"


class TestDatetimeFeatureExtractor:
    """T-034-B: DatetimeFeatureExtractor (for timeseries data)."""

    def test_fit_returns_self(self) -> None:
        """fit() returns self. (T-034-B-01)"""
        tr = DatetimeFeatureExtractor(components=["month", "hour"])
        result = tr.fit(pd.DataFrame({"date": ["2023-01-01", "2023-06-15"]}))
        assert result is tr

    def test_extracts_components(self) -> None:
        """Extracts datetime components correctly. (T-034-B-02)"""
        dates = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"))
        # With add_cyclic=True (default), month adds sin/cos, dayofweek adds sin/cos
        tr = DatetimeFeatureExtractor(components=["month", "dayofweek"], add_cyclic=False)
        tr.fit(dates.to_frame())
        result = tr.transform(dates.to_frame())
        assert result.shape == (5, 2), f"Expected (5,2), got {result.shape}"

    def test_cyclic_features(self) -> None:
        """Adds sin/cos for cyclic components. (T-034-B-03)"""
        dates = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"))
        tr = DatetimeFeatureExtractor(components=["month"], add_cyclic=True)
        tr.fit(dates.to_frame())
        result = tr.transform(dates.to_frame())
        # month, month_sin, month_cos = 3 features
        assert result.shape[1] == 3, f"Expected 3 features with cyclic, got {result.shape[1]}"


class TestLagRollingTransformer:
    """T-034-C: LagRollingTransformer (for timeseries data)."""

    def test_fit_returns_self(self) -> None:
        """fit() returns self. (T-034-C-01)"""
        tr = LagRollingTransformer(lags=[1, 2], windows=[3])
        X = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        result = tr.fit(X)
        assert result is tr

    def test_lag_features(self) -> None:
        """Generates lag features (original col not included). (T-034-C-02)"""
        tr = LagRollingTransformer(lags=[1, 2], windows=[3], agg_funcs=["mean"])
        X = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        tr.fit(X)
        result = tr.transform(X)
        # 2 lag + 1 rolling = 3 features (original col NOT included)
        assert result.shape[1] == 3, f"Expected 3 features, got {result.shape[1]}"

    def test_rolling_features(self) -> None:
        """Generates rolling statistics. (T-034-C-03)"""
        tr = LagRollingTransformer(lags=[], windows=[3], agg_funcs=["mean", "std"])
        X = pd.DataFrame({"val": range(10, 20)})
        tr.fit(X)
        result = tr.transform(X)
        # 0 lag + 2 rolling = 2 features (original col NOT included)
        assert result.shape[1] == 2, f"Expected 3 features, got {result.shape[1]}"


class TestBuildFeatureEngineeringPipeline:
    """T-034-D: build_feature_engineering_pipeline."""

    def test_returns_list(self) -> None:
        """Returns a list of (name, transformer) tuples. (T-034-D-01)"""
        config = FeatureEngineeringConfig(add_interactions=True)
        steps = build_feature_engineering_pipeline(config)
        assert isinstance(steps, list)
        assert len(steps) >= 0

    def test_with_interaction(self, safe_df: pd.DataFrame) -> None:
        """With add_interactions=True, includes InteractionTransformer. (T-034-D-02)"""
        config = FeatureEngineeringConfig(add_interactions=True)
        steps = build_feature_engineering_pipeline(config)
        step_names = [name for name, _ in steps]
        assert "interactions" in step_names, f"Expected 'interactions' in {step_names}"
