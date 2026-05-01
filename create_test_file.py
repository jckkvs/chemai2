"""Create a clean test file for feature selection."""
import textwrap

content = textwrap.dedent("""\
\"\"\"
Tests for correlation-based feature selection and LLM-assisted feature recommendation.

Run:  pytest backend/chem/test_feature_selection.py -v
\"\"\"
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from backend.chem.adaptive_feature_selector import (
    AdaptiveFeatureSelector, FeatureSelectionResult,
)
from backend.chem.correlation_selector import (
    CorrelationBasedSelector,
    CorrelationMethod,
    CorrelationResult,
    FeatureRanking,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_regression_df():
    \"\"\"Sample regression dataset similar to smiles_25_regression.csv.\"\"\"
    np.random.seed(42)
    n = 25
    # Simulate 10 feature columns with known correlations to target
    data = {
        "Compound_Name": [f"Compound_{i:03d}" for i in range(n)],
        "SMILES": ["CCO", "CCCCO", "c1ccccc1", "CC(=O)O"] * (n // 4 + 1),
    }
    # Feature 1: strong positive correlation (0.9)
    f1 = np.linspace(1.0, 10.0, n) + np.random.normal(0, 0.5, n)
    # Feature 2: moderate positive correlation (0.6)
    f2 = np.linspace(2.0, 8.0, n) + np.random.normal(0, 1.0, n)
    # Feature 3: weak correlation (0.2)
    f3 = np.random.normal(5.0, 1.0, n)
    # Feature 4: negative correlation (-0.7)
    f4 = np.linspace(10.0, 1.0, n) + np.random.normal(0, 1.0, n)
    # Feature 5: no correlation (0.0)
    f5 = np.random.normal(0.0, 1.0, n)
    # Feature 6-10: random noise features
    for i in range(6, 11):
        data[f"feature_{i}"] = np.random.normal(0, 1, n)

    data["feature_1"] = f1
    data["feature_2"] = f2
    data["feature_3"] = f3
    data["feature_4"] = f4
    data["feature_5"] = f5
    data["logS"] = -1.5 - 0.3 * f1 + 0.2 * f4 + np.random.normal(0, 0.3, n)

    return pd.DataFrame(data)


@pytest.fixture
def selector():
    return CorrelationBasedSelector()


# ============================================================
# Tests: CorrelationBasedSelector
# ============================================================

class TestCorrelationBasedSelectorInit:
    def test_init_default(self):
        sel = CorrelationBasedSelector()
        assert sel.method == CorrelationMethod.PEARSON
        assert sel.min_correlation == 0.0
        assert sel.max_features is None

    def test_init_custom(self):
        sel = CorrelationBasedSelector(
            method=CorrelationMethod.SPEARMAN,
            min_correlation=0.3,
            max_features=10,
        )
        assert sel.method == CorrelationMethod.SPEARMAN
        assert sel.min_correlation == 0.3
        assert sel.max_features == 10


class TestComputeCorrelations:
    def test_compute_returns_correlation_results(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        assert isinstance(result, CorrelationResult)
        assert len(result.rankings) > 0

    def test_strongest_feature_has_highest_correlation(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        rankings = result.rankings
        # feature_1 was designed with strong correlation
        top = rankings[0]
        assert abs(top.correlation) >= abs(rankings[-1].correlation)

    def test_excludes_non_numeric_columns(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        feature_names = [r.feature_name for r in result.rankings]
        assert "Compound_Name" not in feature_names
        assert "SMILES" not in feature_names

    def test_excludes_target_column_from_rankings(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        feature_names = [r.feature_name for r in result.rankings]
        assert "logS" not in feature_names

    def test_correlation_sign_preserved(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        # feature_4 was designed with negative correlation to logS
        f4_rank = next(r for r in result.rankings if r.feature_name == "feature_4")
        assert f4_rank.correlation < 0

    def test_p_values_are_computed(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        for r in result.rankings:
            assert 0.0 <= r.p_value <= 1.0

    def test_missing_target_raises(self, selector, sample_regression_df):
        df = sample_regression_df
        with pytest.raises(ValueError, match="Target column.*not found"):
            selector.compute_correlations(df, target_column="nonexistent")


class TestSelectTopFeatures:
    def test_select_returns_feature_list(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        selected = selector.select_top_features(result, n_features=3)
        assert len(selected) == 3
        assert all(isinstance(f, str) for f in selected)

    def test_select_respects_max_features(self, selector, sample_regression_df):
        df = sample_regression_df
        selector.max_features = 2
        result = selector.compute_correlations(df, target_column="logS")
        selected = selector.select_top_features(result, n_features=10)
        assert len(selected) <= 2

    def test_select_by_min_correlation(self, selector, sample_regression_df):
        df = sample_regression_df
        selector.min_correlation = 0.5
        result = selector.compute_correlations(df, target_column="logS")
        selected = selector.select_top_features(result, n_features=10)
        for f_name in selected:
            rank = next(r for r in result.rankings if r.feature_name == f_name)
            assert abs(rank.correlation) >= 0.5

    def test_select_returns_highest_correlation_first(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        selected = selector.select_top_features(result, n_features=3)
        # feature_1 should be in top features
        assert "feature_1" in selected


class TestSelectByCorrelationThreshold:
    def test_select_by_threshold_returns_all_above_threshold(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        selected = selector.select_by_threshold(result, threshold=0.4)
        for f_name in selected:
            rank = next(r for r in result.rankings if r.feature_name == f_name)
            assert abs(rank.correlation) >= 0.4

    def test_threshold_zero_returns_all(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        selected = selector.select_by_threshold(result, threshold=0.0)
        assert len(selected) == len(result.rankings)


class TestCorrelationMethod:
    def test_pearson_computation(self, sample_regression_df):
        sel = CorrelationBasedSelector(method=CorrelationMethod.PEARSON)
        result = sel.compute_correlations(sample_regression_df, target_column="logS")
        assert len(result.rankings) > 0

    def test_spearman_computation(self, sample_regression_df):
        sel = CorrelationBasedSelector(method=CorrelationMethod.SPEARMAN)
        result = sel.compute_correlations(sample_regression_df, target_column="logS")
        assert len(result.rankings) > 0

    def test_kendall_computation(self, sample_regression_df):
        sel = CorrelationBasedSelector(method=CorrelationMethod.KENDALL)
        result = sel.compute_correlations(sample_regression_df, target_column="logS")
        assert len(result.rankings) > 0


class TestCorrelationResult:
    def test_to_dataframe(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        df_out = result.to_dataframe()
        assert "feature_name" in df_out.columns
        assert "correlation" in df_out.columns
        assert "abs_correlation" in df_out.columns
        assert "p_value" in df_out.columns
        assert len(df_out) == len(result.rankings)

    def test_get_top_n(self, selector, sample_regression_df):
        df = sample_regression_df
        result = selector.compute_correlations(df, target_column="logS")
        top3 = result.get_top_n(3)
        assert len(top3) == 3


# ============================================================
# Tests: Integration with AdaptiveFeatureSelector
# ============================================================

class TestAdaptiveFeatureSelectorIntegration:
    def test_select_with_correlation_ranking(self, sample_regression_df):
        selector = AdaptiveFeatureSelector()
        result = selector.select_with_correlation(
            df=sample_regression_df,
            target_column="logS",
            task_type="solubility",
            n_molecules=25,
            max_time_per_mol_s=30,
            top_n_features=3,
        )
        assert isinstance(result, FeatureSelectionResult)
        assert len(result.selected_features) <= 3

    def test_correlation_ranking_includes_notes(self, sample_regression_df):
        selector = AdaptiveFeatureSelector()
        result = selector.select_with_correlation(
            df=sample_regression_df,
            target_column="logS",
            task_type="solubility",
            n_molecules=25,
            max_time_per_mol_s=30,
            top_n_features=5,
        )
        assert any("correlation" in note.lower() for note in result.notes)


# ============================================================
# Tests: LLM Feature Advisor (mocked)
# ============================================================

@pytest.fixture
def mock_llm_provider():
    provider = Mock()
    provider.query.return_value = (
        "Based on the target 'logS' (aqueous solubility), "
        "I recommend using: rdkit_2d, morgan_fp, xtb_sp, xtb_ml_derived. "
        "These capture molecular size, polarity, and electronic effects relevant to solubility."
    )
    return provider


class TestLLMFeatureAdvisor:
    def test_llm_advisor_recommends_features(self, mock_llm_provider, sample_regression_df):
        from backend.chem.llm_feature_advisor import LLMFeatureAdvisor

        advisor = LLMFeatureAdvisor(provider=mock_llm_provider)
        recommendations = advisor.recommend(
            df=sample_regression_df,
            target_column="logS",
            task_type="solubility",
        )
        assert len(recommendations.feature_names) > 0
        assert recommendations.confidence > 0

    def test_llm_advisor_parses_feature_list(self, mock_llm_provider):
        from backend.chem.llm_feature_advisor import LLMFeatureAdvisor

        advisor = LLMFeatureAdvisor(provider=mock_llm_provider)
        text = "I recommend: rdkit_2d, morgan_fp, xtb_sp"
        parsed = advisor._parse_feature_names(text)
        assert "rdkit_2d" in parsed
        assert "morgan_fp" in parsed
        assert "xtb_sp" in parsed

    def test_llm_advisor_handles_empty_response(self, mock_llm_provider):
        from backend.chem.llm_feature_advisor import LLMFeatureAdvisor

        mock_llm_provider.query.return_value = ""
        advisor = LLMFeatureAdvisor(provider=mock_llm_provider)
        recommendations = advisor.recommend(
            df=sample_regression_df,
            target_column="logS",
        )
        assert len(recommendations.feature_names) == 0
        assert recommendations.confidence == 0.0


# ============================================================
# Tests: Real data file (smiles_25_regression.csv)
# ============================================================

class TestWithRealDataFile:
    def test_load_and_analyze(self):
        df = pd.read_csv("C:/Users/horie/chemai2_cc/data/samples/smiles_25_regression.csv")
        assert len(df) == 25
        assert "SMILES" in df.columns
        assert "logS" in df.columns

    def test_correlation_on_real_data(self, selector):
        df = pd.read_csv("C:/Users/horie/chemai2_cc/data/samples/smiles_25_regression.csv")
        # After computing descriptors, we can test correlation
        # For now just check the file loads correctly
        assert len(df) > 0
""")

with open("C:/Users/horie/chemai2_cc/backend/chem/test_feature_selection.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Test file created successfully")
