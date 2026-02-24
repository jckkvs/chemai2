"""
tests/test_dim_reduction.py

backend/data/dim_reduction.py のユニットテスト。
DimReducer (PCA / t-SNE) と run_pca / run_tsne をテストする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.data.dim_reduction import (
    DimReductionConfig,
    DimReducer,
    run_pca,
    run_tsne,
)


# ============================================================
# フィクスチャ
# ============================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 80
    return pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n) + 2,
        "f3": np.random.randn(n) * 3,
        "f4": np.abs(np.random.randn(n)),
        "target": np.random.choice([0, 1], n),
    })


# ============================================================
# DimReducer - PCA
# ============================================================

class TestDimReducerPCA:
    """T-DR-001: DimReducer PCAモードのテスト。"""

    def test_output_shape_2d(self, sample_df: pd.DataFrame) -> None:
        """PCA 2D出力の形状が正しいこと。(T-DR-001-01)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="pca", n_components=2)
        reducer = DimReducer(cfg)
        out = reducer.fit_transform(features)
        assert out.shape == (80, 2)

    def test_output_shape_3d(self, sample_df: pd.DataFrame) -> None:
        """PCA 3D出力の形状が正しいこと。(T-DR-001-02)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="pca", n_components=3)
        reducer = DimReducer(cfg)
        out = reducer.fit_transform(features)
        assert out.shape == (80, 3)

    def test_explained_variance_ratio(self, sample_df: pd.DataFrame) -> None:
        """PCAの寄与率が(0, 1]の範囲であること。(T-DR-001-03)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="pca", n_components=2)
        reducer = DimReducer(cfg)
        reducer.fit(features)
        evr = reducer.explained_variance_ratio_
        assert evr is not None
        assert len(evr) == 2
        assert all(0 < v <= 1 for v in evr)

    def test_cumulative_variance_le1(self, sample_df: pd.DataFrame) -> None:
        """寄与率の合計が1以下であること。(T-DR-001-04)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="pca", n_components=3)
        reducer = DimReducer(cfg)
        reducer.fit(features)
        evr = reducer.explained_variance_ratio_
        assert evr is not None
        assert evr.sum() <= 1.0 + 1e-8

    def test_transform_on_new_data(self, sample_df: pd.DataFrame) -> None:
        """fit後に新規データをtransformできること。(T-DR-001-05)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="pca", n_components=2)
        reducer = DimReducer(cfg)
        reducer.fit(features)
        new_data = features.iloc[:10]
        out = reducer.transform(new_data)
        assert out.shape == (10, 2)

    def test_scale_false(self, sample_df: pd.DataFrame) -> None:
        """scale=FalseでもPCAが実行できること。(T-DR-001-06)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="pca", n_components=2, scale=False)
        reducer = DimReducer(cfg)
        out = reducer.fit_transform(features)
        assert out.shape == (80, 2)

    def test_feature_names_out(self, sample_df: pd.DataFrame) -> None:
        """get_feature_names_out() が正しいラベルを返すこと。(T-DR-001-07)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="pca", n_components=2)
        reducer = DimReducer(cfg)
        reducer.fit(features)
        names = reducer.get_feature_names_out()
        assert list(names) == ["pca_0", "pca_1"]

    def test_numpy_input(self) -> None:
        """numpy配列入力でも動作すること。(T-DR-001-08)"""
        X = np.random.randn(50, 5)
        cfg = DimReductionConfig(method="pca", n_components=2)
        reducer = DimReducer(cfg)
        out = reducer.fit_transform(X)
        assert out.shape == (50, 2)


# ============================================================
# DimReducer - t-SNE
# ============================================================

class TestDimReducerTSNE:
    """T-DR-002: DimReducer t-SNEモードのテスト。"""

    def test_output_shape(self, sample_df: pd.DataFrame) -> None:
        """t-SNE 2D出力の形状が正しいこと。(T-DR-002-01)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="tsne", n_components=2,
                                 perplexity=5.0, tsne_max_iter=250)
        reducer = DimReducer(cfg)
        out = reducer.fit_transform(features)
        assert out.shape == (80, 2)

    def test_transform_returns_cached_embedding(self, sample_df: pd.DataFrame) -> None:
        """fit後のtransformがキャッシュされた埋め込みを返すこと。(T-DR-002-02)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="tsne", n_components=2,
                                 perplexity=5.0, tsne_max_iter=250)
        reducer = DimReducer(cfg)
        out1 = reducer.fit_transform(features)
        out2 = reducer.transform(features)
        np.testing.assert_array_equal(out1, out2)

    def test_explained_variance_none_for_tsne(self, sample_df: pd.DataFrame) -> None:
        """t-SNEではexplained_variance_ratio_がNoneであること。(T-DR-002-03)"""
        features = sample_df.drop(columns=["target"])
        cfg = DimReductionConfig(method="tsne", n_components=2,
                                 perplexity=5.0, tsne_max_iter=250)
        reducer = DimReducer(cfg)
        reducer.fit(features)
        assert reducer.explained_variance_ratio_ is None


# ============================================================
# DimReducer - UMAP (条件付き)
# ============================================================

class TestDimReducerUMAP:
    """T-DR-003: DimReducer UMAPモードのテスト。"""

    def test_umap_raises_without_lib(self) -> None:
        """umap-learnが未インストールの場合はImportErrorが上がること。(T-DR-003-01)"""
        from backend.data.dim_reduction import _UMAP_CLASS

        if _UMAP_CLASS is not None:
            pytest.skip("umap-learnがインストールされているためスキップ")

        features = pd.DataFrame(np.random.randn(30, 4), columns=["a", "b", "c", "d"])
        cfg = DimReductionConfig(method="umap", n_components=2)
        reducer = DimReducer(cfg)
        with pytest.raises(ImportError, match="umap-learn"):
            reducer.fit(features)

    def test_umap_output_shape_if_available(self) -> None:
        """umap-learnが利用可能な場合は正しい形状で出力されること。(T-DR-003-02)"""
        from backend.data.dim_reduction import _UMAP_CLASS

        if _UMAP_CLASS is None:
            pytest.skip("umap-learnが未インストールのためスキップ")

        features = pd.DataFrame(np.random.randn(50, 4), columns=["a", "b", "c", "d"])
        cfg = DimReductionConfig(method="umap", n_components=2, n_neighbors=5)
        reducer = DimReducer(cfg)
        out = reducer.fit_transform(features)
        assert out.shape == (50, 2)

    def test_unknown_method_raises(self) -> None:
        """未知の手法指定でValueErrorが上がること。(T-DR-003-03)"""
        cfg = DimReductionConfig(method="unknown")
        reducer = DimReducer(cfg)
        X = np.random.randn(20, 3)
        with pytest.raises(ValueError, match="未知の次元削減手法"):
            reducer.fit(X)


# ============================================================
# run_pca
# ============================================================

class TestRunPCA:
    """T-DR-004: run_pca ヘルパー関数のテスト。"""

    def test_returns_dataframe_and_evr(self, sample_df: pd.DataFrame) -> None:
        """DataFrameとEVRのタプルが返ること。(T-DR-004-01)"""
        emb_df, evr = run_pca(sample_df, n_components=2, target_col="target")
        assert isinstance(emb_df, pd.DataFrame)
        assert emb_df.shape == (80, 2)
        assert len(evr) == 2

    def test_column_names(self, sample_df: pd.DataFrame) -> None:
        """列名がPC1, PC2となること。(T-DR-004-02)"""
        emb_df, _ = run_pca(sample_df, n_components=2)
        assert list(emb_df.columns) == ["PC1", "PC2"]

    def test_target_col_excluded(self, sample_df: pd.DataFrame) -> None:
        """target_col指定時に目的変数が除外されること。(T-DR-004-03)"""
        emb_df1, evr1 = run_pca(sample_df, n_components=2, target_col="target")
        emb_df2, evr2 = run_pca(sample_df, n_components=2, target_col=None)
        # 異なるEVR → target列の有無による差
        assert not np.allclose(evr1, evr2, atol=1e-2)

    def test_index_preserved(self, sample_df: pd.DataFrame) -> None:
        """出力のインデックスが入力と一致すること。(T-DR-004-04)"""
        emb_df, _ = run_pca(sample_df)
        pd.testing.assert_index_equal(emb_df.index, sample_df.index)


# ============================================================
# run_tsne
# ============================================================

class TestRunTSNE:
    """T-DR-005: run_tsne ヘルパー関数のテスト。"""

    def test_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        """DataFrameが返ること。(T-DR-005-01)"""
        emb_df = run_tsne(sample_df, n_components=2,
                          perplexity=5.0, target_col="target",
                          random_state=0)
        assert isinstance(emb_df, pd.DataFrame)
        assert emb_df.shape == (80, 2)

    def test_column_names(self, sample_df: pd.DataFrame) -> None:
        """列名がtSNE1, tSNE2となること。(T-DR-005-02)"""
        emb_df = run_tsne(sample_df, n_components=2, perplexity=5.0)
        assert list(emb_df.columns) == ["tSNE1", "tSNE2"]

    def test_perplexity_auto_clamp(self) -> None:
        """少ないサンプルでもperplexityが自動クランプされてエラーにならないこと。(T-DR-005-03)"""
        df = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
        # perplexity=30 > (20-1)/3 = 6.33 → 自動クランプ
        emb_df = run_tsne(df, perplexity=30.0)
        assert emb_df.shape == (20, 2)
