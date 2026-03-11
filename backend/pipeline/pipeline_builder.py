"""
backend/pipeline/pipeline_builder.py

統合パイプラインビルダー。
PipelineConfig に基づき 5 段階（列選択 → 前処理 → 特徴量生成 → 特徴量選択 → 推定器）の
sklearn Pipeline を構築する。

主要機能:
  - ColumnMeta の monotonic 情報を XGBoost/LightGBM/HistGB の
    monotonic_constraints に自動反映（ネイティブ対応モデル）
  - SVR/GPR/KernelRidge/SVC 等カーネル系モデルには MonotonicKernelWrapper を適用
  - ColumnMeta の group 情報を FeatureSelector（GroupLasso）に連携
  - 各ステップを独立して有効/無効化できる
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.pipeline.column_selector import ColumnMeta, ColumnSelectorWrapper
from backend.pipeline.col_preprocessor import ColPreprocessConfig, ColPreprocessor
from backend.pipeline.feature_generator import FeatureGenConfig, FeatureGenerator
from backend.pipeline.feature_selector import FeatureSelectorConfig, FeatureSelector
from backend.models.factory import get_model

logger = logging.getLogger(__name__)



# ============================================================
# 設定クラス
# ============================================================

@dataclass
class PipelineConfig:
    """
    5 段階 ML パイプラインの設定。

    Attributes:
        task: "regression" | "classification"

        # Step 1: 入力列制御
        col_select_mode: "all" | "include" | "exclude"
        col_select_columns: 対象列名リスト（include/exclude 時）
        col_select_range: (start, end) インデックス範囲（include 時）
        column_meta: 列名 → ColumnMeta（単調性・線形性・グループ情報）

        # Step 2: 列別前処理
        preprocessor_config: ColPreprocessConfig（None でデフォルト）

        # Step 3: 特徴量生成
        feature_gen_config: FeatureGenConfig（None で none=スキップ）

        # Step 4: 特徴量選択
        feature_sel_config: FeatureSelectorConfig（None で none=スキップ）

        # Step 5: 推定器
        estimator_key: factory.py のモデルキー（例: "rf", "xgb"）
        estimator_params: estimator のパラメータ上書き辞書
        apply_monotonic: True なら ColumnMeta から monotonic_constraints を自動設定
    """
    task: str = "regression"

    # Step 1
    col_select_mode: str = "all"
    col_select_columns: list[str] | None = None
    col_select_range: tuple[int, int] | None = None
    column_meta: dict[str, ColumnMeta] = field(default_factory=dict)

    # Step 2
    preprocessor_config: ColPreprocessConfig | None = None

    # Step 3
    feature_gen_config: FeatureGenConfig | None = None

    # Step 4
    feature_sel_config: FeatureSelectorConfig | None = None

    # Step 5
    estimator_key: str = "rf"
    estimator_params: dict[str, Any] = field(default_factory=dict)
    apply_monotonic: bool = True


# ============================================================
# ビルダー関数
# ============================================================

def build_pipeline(config: PipelineConfig) -> Pipeline:
    """
    PipelineConfig から sklearn Pipeline を構築して返す。

    Pipeline の各ステップ:
      1. col_select  : ColumnSelectorWrapper
      2. preprocess  : ColPreprocessor
      3. feature_gen : FeatureGenerator
      4. feature_sel : FeatureSelector
      5. estimator   : sklearn 互換モデル

    Args:
        config: PipelineConfig

    Returns:
        fit 前の sklearn Pipeline

    Notes:
        - ColumnMeta の monotonic 情報が apply_monotonic=True のとき
          XGBoost/LightGBM/HistGB の monotonic_constraints に反映。
        - feature_sel_config が group_lasso を使う場合、column_meta が
          FeatureSelector に自動連携される。
    """
    steps: list[tuple[str, Any]] = []

    # ---- Step 1: 列選択 ----
    col_selector = ColumnSelectorWrapper(
        mode=config.col_select_mode,
        columns=config.col_select_columns or None,    # 空リスト→None（clone()互換性）
        col_range=config.col_select_range,
        column_meta=config.column_meta or None,       # 空辞書→None（clone()互換性）
    )
    steps.append(("col_select", col_selector))

    # ---- Step 2: 列別前処理 ----
    preprocessor = ColPreprocessor(config=config.preprocessor_config)
    steps.append(("preprocess", preprocessor))

    # ---- Step 3: 特徴量生成 ----
    feature_gen = FeatureGenerator(config=config.feature_gen_config)
    steps.append(("feature_gen", feature_gen))

    # ---- Step 4: 特徴量選択 ----
    sel_config = config.feature_sel_config
    if sel_config is not None:
        # task を PipelineConfig から同期
        sel_config.task = config.task
    feature_sel = FeatureSelector(
        config=sel_config,
        column_meta=config.column_meta,
    )
    steps.append(("feature_sel", feature_sel))

    # ---- Step 5: 推定器 ----
    estimator = get_model(
        config.estimator_key,
        task=config.task,
        **config.estimator_params,
    )

    # monotonic_constraints の自動反映
    if config.apply_monotonic and config.column_meta:
        estimator = apply_monotonic_constraints(
            estimator=estimator,
            column_meta=config.column_meta,
        )

    steps.append(("estimator", estimator))

    pipe = Pipeline(steps)
    logger.info(
        f"build_pipeline() 完了: task={config.task}, "
        f"estimator={config.estimator_key}, "
        f"steps={[s for s, _ in steps]}"
    )
    return pipe


# ============================================================
# 単調性制約ヘルパー
# ============================================================

def apply_monotonic_constraints(
    estimator: Any,
    column_meta: dict[str, ColumnMeta],
    feature_names: list[str] | None = None,
    *,
    soft_monotonic_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    estimatorの種類に応じて単調性制約を適用する。

    - **ネイティブ対応**（XGBoost/LightGBM/HistGB）:
        get_params() のキーチェックで判定し、monotonic_constraints等に直接設定。

    - **ソフト対応**（SVR/KernelRidge/GPR/SVC等）:
        MonotonicKernelWrapper / MonotonicKernelClassifierWrapper でラップ。
        学習データ範囲 ± 1.5σ のグリッドでペナルティ反復フィット。

    Args:
        estimator: sklearn 互換の推定器
        column_meta: 列名 → ColumnMeta の辞書
        feature_names: 特徴量名リスト。None の場合は column_meta のキー順。
        soft_monotonic_kwargs: MonotonicKernelWrapper に渡す追加引数
            (n_grid, sigma_factor, penalty_weight, max_iter)

    Returns:
        設定済み estimator（サポート外の場合はそのまま返す）
    """
    names = feature_names or list(column_meta.keys())
    n_features = len(names)
    constraints = tuple(
        column_meta.get(n, ColumnMeta()).monotonic for n in names
    )

    if not any(c != 0 for c in constraints):
        logger.debug("monotonic_constraints: 全て 0 のためスキップ")
        return estimator

    # ── ネイティブ対応: get_params() に "monoton" を含むキーがある場合 ──
    try:
        params = estimator.get_params()
    except Exception:
        return estimator

    monotonic_keys = [
        k for k in params
        if "monoton" in k.lower()
    ]

    if monotonic_keys:
        # XGBoost / LightGBM / HistGB 等 → ネイティブ設定
        cls_name = type(estimator).__name__
        for key in monotonic_keys:
            try:
                val: Any = list(constraints) if "lgbm" in cls_name.lower() else constraints
                estimator.set_params(**{key: val})
                logger.info(
                    f"monotonic 制約設定(ネイティブ): {cls_name}.{key}, "
                    f"制約あり={sum(1 for c in constraints if c != 0)}/{len(constraints)} 列"
                )
                break
            except Exception as e:
                logger.warning(f"set_params({key}=...) 失敗: {e}")
        return estimator

    # ── ソフト対応: カーネル系モデルの判定とラップ ──
    try:
        from backend.models.monotonic_kernel import (
            is_soft_monotonic_candidate,
            wrap_with_soft_monotonic,
        )
        if is_soft_monotonic_candidate(estimator):
            kwargs = soft_monotonic_kwargs or {}
            wrapped = wrap_with_soft_monotonic(
                estimator,
                constraints,
                **kwargs,
            )
            logger.info(
                f"MonotonicKernelWrapper 適用: {type(estimator).__name__}, "
                f"制約={constraints}"
            )
            return wrapped
    except ImportError:
        logger.warning("monotonic_kernel モジュールが見つかりません")

    # ネイティブ・ソフトどちらも非対応の場合
    logger.debug(
        f"'{type(estimator).__name__}' は monotonic パラメータを持ちません。制約を無視します。"
    )
    return estimator


# ============================================================
# グループ情報ユーティリティ
# ============================================================

def extract_group_array(
    column_meta: dict[str, ColumnMeta],
    feature_names: list[str],
) -> np.ndarray | None:
    """
    ColumnMeta のグループ情報から GroupCV 等が使用できる整数配列を返す。

    同じ group 文字列の列が同一グループに割り当てられる。
    group=None の列はグループ -1 となる。

    Args:
        column_meta: 列名 → ColumnMeta の辞書
        feature_names: 特徴量名リスト

    Returns:
        グループ ID 整数配列（shape: [n_features]）、または全列 group=None なら None
    """
    groups = [
        column_meta.get(n, ColumnMeta()).group for n in feature_names
    ]

    if all(g is None for g in groups):
        return None

    # 文字列グループラベルを整数に変換
    label_map: dict[str, int] = {}
    next_id = 0
    result = []
    for g in groups:
        if g is None:
            result.append(-1)
        else:
            if g not in label_map:
                label_map[g] = next_id
                next_id += 1
            result.append(label_map[g])

    return np.array(result, dtype=int)
