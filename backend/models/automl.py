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
from backend.chem.rdkit_adapter import RDKitAdapter
from backend.chem.smiles_transformer import SmilesDescriptorTransformer
from backend.utils.config import RANDOM_STATE, AUTOML_CV_FOLDS

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
    processed_X: pd.DataFrame | None = None
    # SHAP解析・評価用: パイプライン適用前の特徴量と目的変数
    X_train: pd.DataFrame | None = None
    y_train: np.ndarray | None = None
    # CV の Out-Of-Fold 予測 (全データに対するCVの予測値)
    oof_predictions: np.ndarray | None = None
    oof_true: np.ndarray | None = None
    # Holdout (train/test split) の予測
    holdout_true: np.ndarray | None = None
    # SMILES相関係数とTransformerの保持
    smiles_correlations: dict[str, float] = field(default_factory=dict)
    smiles_transformer: Any | None = None


class AutoMLEngine:
    """
    AutoMLエンジン。

    Implements: 要件定義書 §3.11 AutoMLモード

    Args:
        task: "auto" | "regression" | "classification"
        cv_folds: CV分割数
        model_keys: 試すモデルのキーリスト（None時はデフォルトを使用）
        timeout_seconds: 全体のタイムアウト（秒）
        progress_callback: 進捗コールバック (step, total, message) -> None
    """

    def __init__(
        self,
        task: str = "auto",
        cv_folds: int = AUTOML_CV_FOLDS,
        cv_key: str = "auto",  # "auto" = kfold(regression) / stratified_kfold(classification)
        cv_groups_col: str | None = None,  # GroupKFold等で使うグループ列名
        model_keys: list[str] | None = None,
        model_params: dict[str, dict[str, Any]] | None = None,  # {model_key: {param: val}}
        preprocess_params: dict[str, Any] | None = None,  # PreprocessConfigの上書き
        timeout_seconds: int = 600,
        progress_callback: Callable[[int, int, str], None] | None = None,
        selected_descriptors: list[str] | None = None,
        monotonic_constraints_dict: dict[str, int] | None = None,
    ) -> None:
        self.task = task
        self.cv_folds = cv_folds
        self.cv_key = cv_key
        self.cv_groups_col = cv_groups_col
        self.model_keys = model_keys
        self.model_params = model_params or {}
        self.preprocess_params = preprocess_params or {}
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback or (lambda s, t, m: None)
        self.selected_descriptors = selected_descriptors
        self.monotonic_constraints_dict = monotonic_constraints_dict or {}

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        smiles_col: str | None = None,
        group_col: str | None = None,
        preprocess_config: PreprocessConfig | None = None,
        cv_extra_params: dict[str, Any] | None = None,
    ) -> AutoMLResult:
        """
        AutoML全フローを実行する。

        Args:
            df: 入力DataFrame
            target_col: 目的変数の列名
            smiles_col: SMILES列名（化合物データの場合）
            group_col: グループ列名（GroupKFold等で使用）
            preprocess_config: 前処理設定（省略時はデフォルト）
            cv_extra_params: CVスプリッタに渡す追加引数

        Returns:
            AutoMLResult インスタンス
        """
        start = time.time()
        warnings: list[str] = []
        total_steps = 6
        cv_extra_params = cv_extra_params or {}

        # Step 1: データ品質チェック
        self.progress_callback(1, total_steps, "データ品質チェック中...")
        self._check_data_quality(df, target_col, warnings)

        # 目的変数の欠損行を除去
        if df[target_col].isna().any():
            initial_len = len(df)
            df = df.dropna(subset=[target_col]).copy()
            logger.info(f"目的変数の欠損により {initial_len - len(df)} 行を除去しました。")

        # Step 2: 変数型判定（SMILES等を検出）
        self.progress_callback(2, total_steps, "変数型を自動判定中...")
        detector = TypeDetector()
        detection_result = detector.detect(df.drop(columns=[target_col]))

        # Step 3: タスク判定
        self.progress_callback(3, total_steps, "タスク種別を判定中...")
        task = self._infer_task(df[target_col]) if self.task == "auto" else self.task
        logger.info(f"タスク: {task}")

        # Step 4: 目的変数・特徴量の準備
        y = df[target_col].values
        _drop_cols = [target_col]

        groups = df[group_col].values if group_col and group_col in df.columns else None
        # cv_groups_col が指定されている場合はそちらを優先
        if self.cv_groups_col and self.cv_groups_col in df.columns:
            groups = df[self.cv_groups_col].values

        # グループ列を特徴量から除外（_leakage_group等がfeatureに混入するのを防止）
        if group_col and group_col in df.columns:
            _drop_cols.append(group_col)
        if self.cv_groups_col and self.cv_groups_col in df.columns and self.cv_groups_col not in _drop_cols:
            _drop_cols.append(self.cv_groups_col)

        X = df.drop(columns=_drop_cols)

        # 特徴量が1つも残っていない場合のチェック
        if X.shape[1] == 0:
            raise ValueError("学習に使用できる特徴量がありません。目的変数以外の列が存在するか確認してください。")

        # Step 5: モデル学習
        self.progress_callback(4, total_steps, "複数モデルで学習中...")
        model_keys = self.model_keys if self.model_keys else get_default_automl_models(task)
        if not model_keys:
             raise ValueError("学習に使用するモデルが指定されておらず、デフォルトも取得できませんでした。")

        scoring = self._get_scoring(task)
        # cv_key 自動決定: ユーザー指定全優先、"auto"の場合はタスクに応じて自動選択
        if self.cv_key == "auto":
            cv_key = "stratified_kfold" if task == "classification" else "kfold"
        else:
            cv_key = self.cv_key

        # GroupKFold系の場合: グループ数 >= n_splits のバリデーション
        if cv_key in ("group_kfold", "leave_one_group_out") and groups is not None:
            n_unique_groups = len(np.unique(groups))
            if n_unique_groups < self.cv_folds:
                if n_unique_groups >= 2:
                    logger.warning(
                        f"GroupKFold: グループ数({n_unique_groups}) < n_splits({self.cv_folds})。"
                        f"n_splitsを{n_unique_groups}に自動調整します。"
                    )
                    self.cv_folds = n_unique_groups
                else:
                    logger.warning(
                        f"GroupKFold: グループ数({n_unique_groups})が不足。通常KFoldにフォールバックします。"
                    )
                    cv_key = "stratified_kfold" if task == "classification" else "kfold"
                    groups = None

        model_scores: dict[str, float] = {}
        model_details: dict[str, dict[str, Any]] = {}
        best_key = ""
        best_score = float("-inf")
        preprocess_cfg = preprocess_config or PreprocessConfig()
        deadline = start + self.timeout_seconds

        X_train = X

        for i, mkey in enumerate(model_keys):
            if time.time() > deadline:
                warnings.append(f"タイムアウトにより {mkey} 以降のモデルをスキップしました。")
                break

            self.progress_callback(
                4, total_steps,
                f"学習中... ({i + 1}/{len(model_keys)}: {mkey})"
            )
            try:
                model_inst = get_model(mkey, task=task, **self.model_params.get(mkey, {}))
                # 単調性制約を適用（ネイティブ対応 or ソフト制約ラッパー）
                if self.monotonic_constraints_dict:
                    try:
                        from backend.pipeline.column_selector import ColumnMeta
                        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
                        # feature_namesのマッピング（X列名 → monotonic値 → ColumnMeta）
                        _col_meta = {
                            col: ColumnMeta(monotonic=self.monotonic_constraints_dict.get(col, 0))
                            for col in X_train.columns
                        }
                        model_inst = apply_monotonic_constraints(
                            model_inst, _col_meta,
                            feature_names=list(X_train.columns)
                        )
                    except Exception as _e:
                        logger.warning(f"単調性制約適用をスキップ ({mkey}): {_e}")
                pipeline_base = build_full_pipeline(
                    detection_result, model_inst,
                    target_col=target_col,
                    config=preprocess_cfg,
                )
                # SMILES列がある場合、パイプラインの先頭にTransformerを挿入
                if smiles_col and smiles_col in X_train.columns:
                    st_trans = SmilesDescriptorTransformer(
                        smiles_col=smiles_col,
                        selected_descriptors=self.selected_descriptors
                    )
                    pipeline = Pipeline([
                        ("smiles_vars", st_trans),
                        ("main_pipe", pipeline_base)
                    ])
                else:
                    pipeline = pipeline_base
                cv_config = CVConfig(
                    cv_key=cv_key, 
                    n_splits=self.cv_folds,
                    extra_params=cv_extra_params
                )
                result = run_cross_validation(
                    pipeline, X_train, y, cv_config,
                    scoring=scoring,
                    groups=groups,
                    n_jobs=1,
                )
                score_key = f"test_{scoring}"
                if score_key in result:
                    mean_s = float(np.mean(result[score_key]))
                    std_s = float(np.std(result[score_key]))
                else:
                    mean_s = result.get("mean_test_score", 0.0)
                    std_s = result.get("std_test_score", 0.0)

                model_scores[mkey] = mean_s
                fold_scores_list = result[score_key].tolist() if score_key in result else []
                model_details[mkey] = {
                    "mean": mean_s,
                    "std": std_s,
                    "fit_time": float(np.mean(result.get("fit_time", [0]))),
                    "fold_scores": fold_scores_list,
                }
                if mean_s > best_score:
                    best_score = mean_s
                    best_key = mkey
                logger.info(f"  {mkey}: {mean_s:.4f} ± {std_s:.4f}")
            except Exception as e:
                msg = f"{mkey} の学習中にエラー: {str(e)}"
                logger.warning(msg)
                warnings.append(msg)
                import traceback
                logger.debug(traceback.format_exc())

        if not best_key:
            err_details = "\n".join(warnings[-min(len(warnings), 5):])
            raise RuntimeError(f"全モデルの学習に失敗しました（特徴量が全て除去された可能性があります）。詳細:\n{err_details}")

        # Step 6: 最良モデルを全データで再学習
        self.progress_callback(5, total_steps, f"最良モデル({best_key})を全データで学習中...")
        best_model = get_model(best_key, task=task, **self.model_params.get(best_key, {}))
        # 最良モデルにも単調性制約を適用
        if self.monotonic_constraints_dict:
            try:
                from backend.pipeline.column_selector import ColumnMeta
                from backend.pipeline.pipeline_builder import apply_monotonic_constraints
                _col_meta_best = {
                    col: ColumnMeta(monotonic=self.monotonic_constraints_dict.get(col, 0))
                    for col in X_train.columns
                }
                best_model = apply_monotonic_constraints(
                    best_model, _col_meta_best,
                    feature_names=list(X_train.columns)
                )
            except Exception as _e:
                logger.warning(f"最良モデルへの単調性制約適用をスキップ ({best_key}): {_e}")
        best_pipeline_base = build_full_pipeline(
            detection_result, best_model,
            target_col=target_col,
            config=preprocess_cfg,
        )
        if smiles_col and smiles_col in X_train.columns:
            st_trans = SmilesDescriptorTransformer(
                smiles_col=smiles_col,
                selected_descriptors=self.selected_descriptors
            )
            best_pipeline = Pipeline([
                ("smiles_vars", st_trans),
                ("main_pipe", best_pipeline_base)
            ])
        else:
            best_pipeline = best_pipeline_base
            
        best_pipeline.fit(X_train, y)

        # パイプラインの前処理部分(estimator以外)でtransformし、
        # 「実際にモデルに入力された最終データ」を取得する
        processed_X_final: pd.DataFrame | None = None
        try:
            # Pipeline[-1]がestimator。Pipeline[:-1]が前処理ステップ群。
            preprocessor_steps = best_pipeline[:-1]
            X_transformed = preprocessor_steps.transform(X_train)
            # 特徴量名の取得
            try:
                feat_names = preprocessor_steps.get_feature_names_out().tolist()
            except Exception:
                n_cols = X_transformed.shape[1] if hasattr(X_transformed, "shape") else len(X_transformed[0])
                feat_names = [f"feature_{i}" for i in range(n_cols)]
            # sparse → dense変換
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            processed_X_final = pd.DataFrame(
                X_transformed, columns=feat_names, index=X_train.index
            )
        except Exception as e:
            logger.warning(f"前処理後データの取得に失敗: {e}")
            processed_X_final = X_train  # フォールバック: 生データ

        # OOF (Out-Of-Fold) 予測を計算（最良モデルで cross_val_predict）
        oof_preds: np.ndarray | None = None
        try:
            from sklearn.model_selection import cross_val_predict
            from backend.models.cv_manager import get_cv
            _cv_splitter = get_cv(CVConfig(cv_key=cv_key, n_splits=self.cv_folds, extra_params=cv_extra_params))
            _cv_method = "predict_proba" if task == "classification" and hasattr(best_pipeline, "predict_proba") else "predict"
            oof_preds = cross_val_predict(
                best_pipeline, X_train, y,
                cv=_cv_splitter, method=_cv_method, n_jobs=1,
                groups=groups,
            )
            # predict_proba の場合はスコア（クラス1の確率）のみ or argmax
            if _cv_method == "predict_proba" and oof_preds.ndim == 2:
                oof_preds = oof_preds.argmax(axis=1)
        except Exception as e:
            logger.warning(f"OOF予測の計算に失敗: {e}")
            oof_preds = None

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
            processed_X=processed_X_final,
            X_train=X_train,
            y_train=y,
            oof_predictions=oof_preds,
            oof_true=y if oof_preds is not None else None,
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
