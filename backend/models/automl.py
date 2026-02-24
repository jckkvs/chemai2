"""
backend/models/automl.py

AutoML エンジン。非専門家がワンボタンで機械学習を実行できるエンジン。
データ型自動判定 → 前処理 → 複数モデル学習 → 自動選択 → 結果返却。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from backend.data.type_detector import TypeDetector, DetectionResult
from backend.data.preprocessor import Preprocessor, PreprocessConfig, build_full_pipeline
from backend.models.factory import get_model, get_default_automl_models
from backend.models.cv_manager import CVConfig, run_cross_validation
from backend.utils.config import RANDOM_STATE, AUTOML_CV_FOLDS, AUTOML_MAX_MODELS

logger = logging.getLogger(__name__)


@dataclass
class AutoMLResult:
    """AutoML実行結果を保持するデータクラス。"""
    task: str                          # "regression" | "classification"
    best_model_key: str
    best_pipeline: Pipeline
    best_score: float
    scoring: str
    model_scores: dict[str, float]     # {model_key: cv_mean_score}
    model_details: dict[str, dict]     # {model_key: {mean, std, fit_time}}
    detection_result: DetectionResult
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)


class AutoMLEngine:
    """
    AutoMLエンジン。

    Implements: 要件定義書 §3.11 AutoMLモード

    Args:
        task: "auto" | "regression" | "classification"
        cv_folds: CV分割数
        max_models: 試すモデルの最大数
        timeout_seconds: 全体のタイムアウト（秒）
        progress_callback: 進捗コールバック (step, total, message) -> None
    """

    def __init__(
        self,
        task: str = "auto",
        cv_folds: int = AUTOML_CV_FOLDS,
        max_models: int = AUTOML_MAX_MODELS,
        timeout_seconds: int = 600,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        self.task = task
        self.cv_folds = cv_folds
        self.max_models = max_models
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback or (lambda s, t, m: None)

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        smiles_col: str | None = None,
        group_col: str | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ) -> AutoMLResult:
        """
        AutoML全フローを実行する。

        Args:
            df: 入力DataFrame
            target_col: 目的変数の列名
            smiles_col: SMILES列名（化合物データの場合）
            group_col: グループ列名（GroupKFold等で使用）
            preprocess_config: 前処理設定（省略時はデフォルト）

        Returns:
            AutoMLResult インスタンス

        Raises:
            ValueError: データが少なすぎる・タスク判定失敗等
        """
        start = time.time()
        warnings: list[str] = []
        total_steps = 6

        # Step 1: データ品質チェック
        self.progress_callback(1, total_steps, "データ品質チェック中...")
        self._check_data_quality(df, target_col, warnings)

        # Step 2: 変数型判定
        self.progress_callback(2, total_steps, "変数型を自動判定中...")
        detector = TypeDetector()
        detection_result = detector.detect(df.drop(columns=[target_col]))

        # Step 3: タスク判定
        self.progress_callback(3, total_steps, "タスク種別を判定中...")
        task = self._infer_task(df[target_col]) if self.task == "auto" else self.task
        logger.info(f"タスク: {task}")

        # Step 4: 目的変数・特徴量の準備
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        if smiles_col and smiles_col in X.columns:
            X = X.drop(columns=[smiles_col])  # SMILES列は別途処理

        groups = df[group_col].values if group_col and group_col in df.columns else None

        # Step 5: モデル学習
        self.progress_callback(4, total_steps, "複数モデルで学習中...")
        model_keys = get_default_automl_models(task)[: self.max_models]
        scoring = self._get_scoring(task)
        cv_key = "stratified_kfold" if task == "classification" else "kfold"

        model_scores: dict[str, float] = {}
        model_details: dict[str, dict[str, Any]] = {}
        best_key = ""
        best_score = float("-inf")

        preprocess_cfg = preprocess_config or PreprocessConfig()
        deadline = start + self.timeout_seconds

        for i, mkey in enumerate(model_keys):
            if time.time() > deadline:
                warnings.append(f"タイムアウトにより {mkey} 以降のモデルをスキップしました。")
                break

            self.progress_callback(
                4, total_steps,
                f"学習中... ({i + 1}/{len(model_keys)}: {mkey})"
            )
            try:
                model_inst = get_model(mkey, task=task)
                pipeline = build_full_pipeline(
                    detection_result, model_inst,
                    target_col=target_col,
                    config=preprocess_cfg,
                )
                cv_config = CVConfig(cv_key=cv_key, n_splits=self.cv_folds)
                result = run_cross_validation(
                    pipeline, X, y, cv_config,
                    scoring=scoring,
                    groups=groups,
                    n_jobs=1,  # AutoML内部はシングル（並列はモデルレベルで）
                )
                score_key = f"test_{scoring}"
                if score_key in result:
                    mean_s = float(np.mean(result[score_key]))
                    std_s = float(np.std(result[score_key]))
                else:
                    mean_s = result.get("mean_test_score", 0.0)
                    std_s = result.get("std_test_score", 0.0)

                model_scores[mkey] = mean_s
                model_details[mkey] = {
                    "mean": mean_s,
                    "std": std_s,
                    "fit_time": float(np.mean(result.get("fit_time", [0]))),
                }
                if mean_s > best_score:
                    best_score = mean_s
                    best_key = mkey
                logger.info(f"  {mkey}: {mean_s:.4f} ± {std_s:.4f}")
            except Exception as e:
                msg = f"{mkey} の学習中にエラー: {e}"
                logger.warning(msg)
                warnings.append(msg)

        if not best_key:
            raise RuntimeError("全モデルの学習に失敗しました。データを確認してください。")

        # Step 6: 最良モデルを全データで再学習
        self.progress_callback(5, total_steps, f"最良モデル({best_key})を全データで学習中...")
        best_model = get_model(best_key, task=task)
        best_pipeline = build_full_pipeline(
            detection_result, best_model,
            target_col=target_col,
            config=preprocess_cfg,
        )
        best_pipeline.fit(X, y)

        self.progress_callback(6, total_steps, "完了!")
        elapsed = time.time() - start

        logger.info(
            f"AutoML完了: {elapsed:.1f}秒 / 最良モデル={best_key} / score={best_score:.4f}"
        )

        return AutoMLResult(
            task=task,
            best_model_key=best_key,
            best_pipeline=best_pipeline,
            best_score=best_score,
            scoring=scoring,
            model_scores=model_scores,
            model_details=model_details,
            detection_result=detection_result,
            elapsed_seconds=elapsed,
            warnings=warnings,
        )

    @staticmethod
    def _infer_task(y_series: pd.Series) -> str:
        """目的変数から回帰/分類を自動判定する。"""
        if pd.api.types.is_float_dtype(y_series):
            return "regression"
        if pd.api.types.is_integer_dtype(y_series):
            n_unique = y_series.nunique()
            threshold = max(10, int(0.05 * len(y_series)))
            return "classification" if n_unique <= threshold else "regression"
        # 文字列/カテゴリ → 分類
        return "classification"

    @staticmethod
    def _get_scoring(task: str) -> str:
        """タスク種別に応じたデフォルトscoring文字列を返す。"""
        if task == "regression":
            return "neg_root_mean_squared_error"
        return "f1_weighted"

    @staticmethod
    def _check_data_quality(
        df: pd.DataFrame,
        target_col: str,
        warnings: list[str],
    ) -> None:
        """データ品質の基本チェックを実施して警告リストに追記する。"""
        if len(df) < 10:
            raise ValueError(f"データが少なすぎます（{len(df)}行）。最低10行必要です。")
        if target_col not in df.columns:
            raise ValueError(f"目的変数列 '{target_col}' が存在しません。")

        null_rate = df[target_col].isna().mean()
        if null_rate > 0:
            warnings.append(f"目的変数 '{target_col}' に欠損値が {null_rate:.1%} あります。欠損行を除外します。")

        dup_rate = df.duplicated().mean()
        if dup_rate > 0.05:
            warnings.append(f"重複行が {dup_rate:.1%} あります。")
