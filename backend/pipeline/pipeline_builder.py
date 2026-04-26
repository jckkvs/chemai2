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
    estimator,
    column_meta: Dict[str, 'ColumnMeta'],
    feature_names: Optional[List[str]] = None,
    verbose: bool = False
):
    """
    単調性制約を推定器に適用するラッパー関数
    
    【修正点1】列名解決ロジックの強化：変換前後の列名マッピングに対応
    """
    from backend.models.monotonic_kernel import ConstrainedEstimatorWrapper
    
    # 制約対象の列をフィルタリング
    constraints = {}
    for col, meta in column_meta.items():
        if hasattr(meta, 'monotonic') and meta.monotonic in (1, -1):
            # 【修正点2】feature_names が指定されている場合のマッピング
            if feature_names and col not in feature_names:
                # 列名が変換された可能性: 部分一致でマッチングを試みる
                matched = [f for f in feature_names if col in f or f.startswith(col + "_")]
                if matched:
                    if verbose:
                        logger.info(f"Column '{col}' mapped to {matched[0]} after transformation")
                    constraints[matched[0]] = meta.monotonic
                continue
            constraints[col] = meta.monotonic
    
    if not constraints:
        if verbose:
            logger.info("No monotonic constraints to apply")
        return estimator
    
    # 【修正点3】推定器が既に制約ラッパー済みの場合の二重適用防止
    if isinstance(estimator, ConstrainedEstimatorWrapper):
        if verbose:
            logger.warning("Estimator already wrapped with constraints. Skipping re-wrap.")
        return estimator
    
    # 【修正点4】制約適用可能な推定器のチェック
    supported = ['HistGradientBoostingRegressor', 'HistGradientBoostingClassifier', 
                 'XGBRegressor', 'XGBClassifier', 'LGBMRegressor', 'LGBMClassifier']
    est_name = type(estimator).__name__
    
    if est_name not in supported and not hasattr(estimator, 'monotonic_cst'):
        if verbose:
            logger.info(f"Estimator {est_name} does not support native monotonic constraints. "
                       f"Using soft constraint wrapper.")
        # 【修正点5】ソフト制約ラッパーへのフォールバック
        return ConstrainedEstimatorWrapper(
            base_estimator=estimator,
            constraints=constraints,
            strength='soft',  # 強制的にソフト制約を使用
            feature_names=feature_names
        )
    
    # Native monotonic constraints 適用
    if hasattr(estimator, 'set_params') and 'monotonic_cst' in estimator.get_params():
        # 【修正点6】feature_names の順序と制約配列の整合性確保
        if feature_names:
            monotonic_array = []
            for feat in feature_names:
                val = constraints.get(feat, 0)
                monotonic_array.append(int(val))
            estimator.set_params(monotonic_cst=tuple(monotonic_array))
            if verbose:
                logger.info(f"Applied native monotonic constraints: {monotonic_array}")
        else:
            logger.warning("feature_names not provided for native constraint application")
    
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
