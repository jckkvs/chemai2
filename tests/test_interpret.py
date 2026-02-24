"""
tests/test_interpret.py

backend/interpret モジュールのユニットテスト。
ShapExplainer (モック版) と SRIDecomposer をテストする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.interpret.sri import (
    SRIDecomposer,
    SRIResult,
    select_features_by_independence,
)
from backend.interpret.shap_explainer import ShapResult


# ============================================================
# テスト用フィクスチャ
# ============================================================

@pytest.fixture
def dummy_shap_result() -> ShapResult:
    """テスト用のダミーShapResultを生成する。"""
    np.random.seed(42)
    n, d = 100, 5
    feature_names = [f"feat_{i}" for i in range(d)]
    shap_values = np.random.randn(n, d)
    return ShapResult(
        shap_values=shap_values,
        expected_value=0.5,
        feature_names=feature_names,
        X_transformed=np.random.randn(n, d),
        explainer_type="tree",
        is_multiclass=False,
    )


@pytest.fixture
def multiclass_shap_result() -> ShapResult:
    """マルチクラス用のダミーShapResultを生成する。"""
    np.random.seed(42)
    n, d, c = 80, 4, 3  # 3クラス
    feature_names = [f"f{i}" for i in range(d)]
    shap_values = np.random.randn(n, d, c)
    return ShapResult(
        shap_values=shap_values,
        expected_value=np.array([0.3, 0.3, 0.4]),
        feature_names=feature_names,
        X_transformed=np.random.randn(n, d),
        explainer_type="tree",
        is_multiclass=True,
    )


# ============================================================
# T-008: SRIDecomposer テスト
# ============================================================

class TestSRIDecomposer:
    """
    T-008: SRI分解のテスト。
    論文: Ittner et al. arXiv 2021 (§3.1 SRI Decomposition)
    """

    def test_decompose_returns_sriresult(self, dummy_shap_result: ShapResult) -> None:
        """decompose() が SRIResult を返すこと。(T-008-01)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        assert isinstance(result, SRIResult)

    def test_synergy_matrix_shape(self, dummy_shap_result: ShapResult) -> None:
        """Synergy行列が (d, d) の形状であること。(T-008-02)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        d = len(dummy_shap_result.feature_names)
        assert result.synergy_matrix.shape == (d, d)

    def test_redundancy_matrix_shape(self, dummy_shap_result: ShapResult) -> None:
        """Redundancy行列が (d, d) の形状であること。(T-008-03)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        d = len(dummy_shap_result.feature_names)
        assert result.redundancy_matrix.shape == (d, d)

    def test_independence_vec_shape(self, dummy_shap_result: ShapResult) -> None:
        """Independence ベクトルが (d,) の形状であること。(T-008-04)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        d = len(dummy_shap_result.feature_names)
        assert result.independence_vec.shape == (d,)

    def test_synergy_matrix_symmetric(self, dummy_shap_result: ShapResult) -> None:
        """Synergy行列が対称行列であること。(T-008-05)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        np.testing.assert_allclose(result.synergy_matrix, result.synergy_matrix.T, atol=1e-10)

    def test_redundancy_matrix_symmetric(self, dummy_shap_result: ShapResult) -> None:
        """Redundancy行列が対称行列であること。(T-008-06)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        np.testing.assert_allclose(
            result.redundancy_matrix, result.redundancy_matrix.T, atol=1e-10
        )

    def test_independence_vec_nonnegative(self, dummy_shap_result: ShapResult) -> None:
        """Independence スコアが非負であること。(T-008-07)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        assert np.all(result.independence_vec >= 0)

    def test_total_sri_tuple(self, dummy_shap_result: ShapResult) -> None:
        """total_sri が (float, float, float) のタプルであること。(T-008-08)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        syn, red, ind = result.total_sri
        assert isinstance(syn, float)
        assert isinstance(red, float)
        assert isinstance(ind, float)
        assert syn >= 0 and red >= 0 and ind >= 0

    def test_feature_names_preserved(self, dummy_shap_result: ShapResult) -> None:
        """特徴量名がSRIResultに保持されること。(T-008-09)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        assert result.feature_names == dummy_shap_result.feature_names

    def test_summary_df_has_required_columns(self, dummy_shap_result: ShapResult) -> None:
        """summary_df() が必要な列を持つこと。(T-008-10)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        df = result.summary_df()
        assert isinstance(df, pd.DataFrame)
        for col in ["feature", "synergy", "redundancy", "independence"]:
            assert col in df.columns

    def test_pairwise_df_has_required_columns(self, dummy_shap_result: ShapResult) -> None:
        """pairwise_df() が必要な列を持つこと。(T-008-11)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        df = result.pairwise_df()
        for col in ["feature_i", "feature_j", "synergy", "redundancy"]:
            assert col in df.columns

    def test_pairwise_df_row_count(self, dummy_shap_result: ShapResult) -> None:
        """pairwise_df() の行数が C(d,2) であること。(T-008-12)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        d = len(dummy_shap_result.feature_names)
        expected_pairs = d * (d - 1) // 2
        assert len(result.pairwise_df()) == expected_pairs

    def test_multiclass_decompose_uses_class0(self, multiclass_shap_result: ShapResult) -> None:
        """マルチクラスSHAPでもクラス0で分解されること。(T-008-13)"""
        decomposer = SRIDecomposer()
        # 警告ログが出るが例外は出ないこと
        result = decomposer.decompose(multiclass_shap_result)
        assert isinstance(result, SRIResult)
        d = len(multiclass_shap_result.feature_names)
        assert result.synergy_matrix.shape == (d, d)

    def test_center_option_no_error(self, dummy_shap_result: ShapResult) -> None:
        """center=False でもエラーなく実行できること。(T-008-14)"""
        decomposer = SRIDecomposer(center=False)
        result = decomposer.decompose(dummy_shap_result)
        assert isinstance(result, SRIResult)


class TestSelectFeaturesByIndependence:
    """T-009: Independence基準の特徴量選択のテスト。"""

    def test_select_top_n(self, dummy_shap_result: ShapResult) -> None:
        """top_n=3 で3特徴量が返ること。(T-009-01)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        selected = select_features_by_independence(result, top_n=3)
        assert len(selected) == 3

    def test_select_with_threshold(self, dummy_shap_result: ShapResult) -> None:
        """threshold を指定した場合、条件を満たす特徴量が返ること。(T-009-02)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        threshold = 0.0
        selected = select_features_by_independence(result, threshold=threshold)
        for feat in selected:
            idx = result.feature_names.index(feat)
            assert result.independence_vec[idx] >= threshold

    def test_select_all_if_no_args(self, dummy_shap_result: ShapResult) -> None:
        """引数なしで全特徴量が返ること。(T-009-03)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        selected = select_features_by_independence(result)
        assert len(selected) == len(dummy_shap_result.feature_names)

    def test_select_returns_independence_sorted(self, dummy_shap_result: ShapResult) -> None:
        """返るリストがIndependence降順であること。(T-009-04)"""
        decomposer = SRIDecomposer()
        result = decomposer.decompose(dummy_shap_result)
        selected = select_features_by_independence(result)
        scores = [result.independence_vec[result.feature_names.index(f)] for f in selected]
        assert scores == sorted(scores, reverse=True)
