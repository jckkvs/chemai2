"""
backend/data/dim_reduction.py

次元削減モジュール。PCA / t-SNE / UMAP をsklearn互換Transformerとして提供する。
オプション依存（umap-learn）が未インストールの場合はPCA/t-SNEのみ使用可能。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

# 任意依存
umap_module = safe_import("umap", alias="umap-learn")
# UMAPクラスを取得
_UMAP_CLASS = getattr(umap_module, "UMAP", None) if umap_module is not None else None


# ============================================================
# 設定
# ============================================================

@dataclass
class DimReductionConfig:
    """次元削減の設定。"""
    method: str = "pca"            # "pca" | "tsne" | "umap"
    n_components: int = 2
    # PCA
    whiten: bool = False
    # t-SNE
    perplexity: float = 30.0
    tsne_max_iter: int = 1000
    # UMAP
    n_neighbors: int = 15
    min_dist: float = 0.1
    # 共通
    random_state: int = 42
    scale: bool = True             # 事前にStandardScalerを適用するか


# ============================================================
# Transformer
# ============================================================

class DimReducer(BaseEstimator, TransformerMixin):
    """
    次元削減Transformer (PCA / t-SNE / UMAP)。

    Implements: 要件定義書 §3.6 次元削減

    Args:
        config: DimReductionConfig

    Note:
        t-SNEはfit/transformが分離できないため（バージョン制約）、
        fit_transform() を推奨。新規データのtransformはサポートしない
        (sklearn 1.4+では PCA-then-TSNE の2段階方式を推奨)。
    """

    def __init__(self, config: DimReductionConfig | None = None) -> None:
        self.config = config or DimReductionConfig()
        self._scaler: StandardScaler | None = None
        self._reducer: Any = None
        self._embedding: np.ndarray | None = None
        self._feature_names_in: list[str] = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "DimReducer":
        """
        次元削減モデルをfitする。
        t-SNEの場合はfit_transform()を内部で実行。
        """
        X_arr = self._prepare(X)
        cfg = self.config

        if cfg.method == "pca":
            self._reducer = PCA(
                n_components=cfg.n_components,
                whiten=cfg.whiten,
                random_state=cfg.random_state,
            )
            self._reducer.fit(X_arr)

        elif cfg.method == "tsne":
            # t-SNEは内部でfit_transformのみ対応
            self._reducer = TSNE(
                n_components=cfg.n_components,
                perplexity=cfg.perplexity,
                max_iter=cfg.tsne_max_iter,
                random_state=cfg.random_state,
            )
            self._embedding = self._reducer.fit_transform(X_arr)

        elif cfg.method == "umap":
            if _UMAP_CLASS is None:
                raise ImportError(
                    "UMAPを使用するには 'umap-learn' のインストールが必要です: "
                    "pip install umap-learn"
                )
            self._reducer = _UMAP_CLASS(
                n_components=cfg.n_components,
                n_neighbors=cfg.n_neighbors,
                min_dist=cfg.min_dist,
                random_state=cfg.random_state,
            )
            self._embedding = self._reducer.fit_transform(X_arr)

        else:
            raise ValueError(f"未知の次元削減手法: {cfg.method}。'pca'/'tsne'/'umap'から選択。")

        return self

    def transform(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> np.ndarray:
        """
        次元削減を適用する。

        Note:
            t-SNE / UMAPはfit時のデータを返す（新規データ非対応）。
        """
        assert self._reducer is not None, "fit()を先に呼んでください。"
        cfg = self.config

        if cfg.method == "pca":
            X_arr = self._prepare(X, fit=False)
            return self._reducer.transform(X_arr)

        elif cfg.method in ("tsne", "umap"):
            # 既存埋め込みを返す（新規データ変換は非対応）
            if self._embedding is not None:
                return self._embedding
            X_arr = self._prepare(X, fit=False)
            return self._reducer.fit_transform(X_arr)

        raise ValueError(f"未知の手法: {cfg.method}")

    def fit_transform(  # type: ignore[override]
        self, X: np.ndarray | pd.DataFrame, y: Any = None, **params: Any
    ) -> np.ndarray:
        """fit + transform を1ステップで実行（t-SNE/UMAP推奨）。"""
        return self.fit(X, y).transform(X)

    def _prepare(
        self,
        X: np.ndarray | pd.DataFrame,
        fit: bool = True,
    ) -> np.ndarray:
        """前処理（スケーリング）を適用して numpy 配列を返す。"""
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            X_arr = X.values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)

        if self.config.scale:
            if fit:
                self._scaler = StandardScaler()
                X_arr = self._scaler.fit_transform(X_arr)
            elif self._scaler is not None:
                X_arr = self._scaler.transform(X_arr)

        return X_arr

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        cfg = self.config
        return np.array([f"{cfg.method}_{i}" for i in range(cfg.n_components)])

    @property
    def explained_variance_ratio_(self) -> np.ndarray | None:
        """PCAの寄与率（PCA以外はNone）。"""
        if self.config.method == "pca" and self._reducer is not None:
            return self._reducer.explained_variance_ratio_
        return None


# ============================================================
# 便利関数
# ============================================================

def run_pca(
    df: pd.DataFrame,
    n_components: int = 2,
    scale: bool = True,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    DataFrameにPCAを適用して2D埋め込み + 寄与率を返す。

    Args:
        df: 入力DataFrame（数値列のみ使用）
        n_components: 次元数
        scale: スケーリングするか
        target_col: 除外する目的変数列

    Returns:
        (embedding_df, explained_variance_ratio)
    """
    numeric = df.select_dtypes(include="number")
    if target_col and target_col in numeric.columns:
        numeric = numeric.drop(columns=[target_col])

    cfg = DimReductionConfig(method="pca", n_components=n_components, scale=scale)
    reducer = DimReducer(cfg)
    embedding = reducer.fit_transform(numeric)

    col_names = [f"PC{i+1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, columns=col_names, index=df.index)
    evr = reducer.explained_variance_ratio_
    return emb_df, evr if evr is not None else np.array([])


def run_tsne(
    df: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    scale: bool = True,
    target_col: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    DataFrameにt-SNEを適用して2D埋め込みを返す。

    Args:
        df: 入力DataFrame
        n_components: 次元数
        perplexity: パープレキシティ（サンプル数の1/3以下を推奨）
        scale: スケーリングするか
        target_col: 除外する目的変数列
        random_state: 乱数シード

    Returns:
        embedding_df (index=元DataFrameのindex)
    """
    numeric = df.select_dtypes(include="number")
    if target_col and target_col in numeric.columns:
        numeric = numeric.drop(columns=[target_col])

    # perplexityはサンプル数-1未満でないといけない
    n = len(numeric)
    perplexity = min(perplexity, (n - 1) / 3)

    cfg = DimReductionConfig(
        method="tsne",
        n_components=n_components,
        perplexity=perplexity,
        scale=scale,
        random_state=random_state,
    )
    reducer = DimReducer(cfg)
    embedding = reducer.fit_transform(numeric)

    col_names = [f"tSNE{i+1}" for i in range(n_components)]
    return pd.DataFrame(embedding, columns=col_names, index=df.index)


def run_umap(
    df: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    scale: bool = True,
    target_col: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    DataFrameにUMAPを適用して埋め込みを返す。
    umap-learnが未インストールの場合はImportErrorを送出する。

    Args:
        df: 入力DataFrame
        n_components: 次元数
        n_neighbors: 近傍数
        min_dist: 最小距離
        scale: スケーリングするか
        target_col: 除外する目的変数列
        random_state: 乱数シード

    Returns:
        embedding_df
    """
    numeric = df.select_dtypes(include="number")
    if target_col and target_col in numeric.columns:
        numeric = numeric.drop(columns=[target_col])

    cfg = DimReductionConfig(
        method="umap",
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        scale=scale,
        random_state=random_state,
    )
    reducer = DimReducer(cfg)
    embedding = reducer.fit_transform(numeric)

    col_names = [f"UMAP{i+1}" for i in range(n_components)]
    return pd.DataFrame(embedding, columns=col_names, index=df.index)
