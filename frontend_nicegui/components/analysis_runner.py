"""
frontend_nicegui/components/analysis_runner.py

解析実行コンポーネント: AutoMLEngine呼び出しとリアルタイム進捗表示。
"""
from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)


async def run_analysis(state: dict[str, Any], status_container, on_complete=None) -> None:
    """
    AutoML解析を実行し、結果をstateに保存する。

    Args:
        state: 共有ステート辞書
        status_container: 進捗表示を描画するUIコンテナ
        on_complete: 完了時のコールバック
    """
    df = state.get("df")
    target_col = state.get("target_col")

    if df is None or not target_col:
        ui.notify("データと目的変数を設定してください", type="warning")
        return

    # 進捗表示の構築
    status_container.clear()
    with status_container:
        progress_label = ui.label("⏳ 解析を開始しています...").classes("text-lg q-mb-sm")
        progress_bar = ui.linear_progress(value=0, show_value=False).classes("q-mb-sm")
        progress_detail = ui.label("").classes("text-caption text-grey-5")
        log_container = ui.column().classes("full-width q-mt-md")

    def progress_callback(step: int, total: int, msg: str) -> None:
        progress_bar.value = step / total
        progress_label.text = f"⏳ {msg}"
        progress_detail.text = f"ステップ {step}/{total}"
        with log_container:
            ui.label(f"  [{step}/{total}] {msg}").classes("text-caption text-grey-6")

    try:
        from backend.models.automl import AutoMLEngine

        # タスク判定
        task = state.get("task_type", "auto")

        # モデル選択
        model_keys = state.get("selected_models")
        if not model_keys:
            from backend.models.factory import get_default_automl_models
            effective_task = task
            if effective_task == "auto":
                effective_task = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
            model_keys = get_default_automl_models(task=effective_task)

        # SMILES列
        smiles_col = state.get("smiles_col") or None

        # 記述子選択
        selected_desc = state.get("selected_descriptors")

        # エンジンの構築
        engine = AutoMLEngine(
            task=task,
            cv_folds=state.get("cv_folds", 5),
            model_keys=model_keys if model_keys else None,
            timeout_seconds=state.get("timeout", 300),
            progress_callback=progress_callback,
            selected_descriptors=selected_desc,
        )

        # 除外列の処理
        exclude_cols = state.get("exclude_cols", [])
        df_work = df.copy()
        if exclude_cols:
            df_work = df_work.drop(columns=[c for c in exclude_cols if c in df_work.columns], errors="ignore")

        # 実行
        result = engine.run(
            df_work,
            target_col=target_col,
            smiles_col=smiles_col if smiles_col and smiles_col in df_work.columns else None,
            group_col=state.get("group_col"),
        )

        # 結果の保存
        state["automl_result"] = result
        state["pipeline_result"] = type("PipelineResult", (), {"elapsed": result.elapsed_seconds})()

        # 成功表示
        progress_bar.value = 1.0
        progress_label.text = f"✅ 解析完了！ 最良モデル: {result.best_model_key}"
        progress_detail.text = (
            f"スコア: {result.best_score:.4f} | "
            f"所要時間: {result.elapsed_seconds:.1f}秒 | "
            f"タスク: {result.task}"
        )
        ui.notify(
            f"✅ 解析完了！ 最良: {result.best_model_key} (スコア: {result.best_score:.4f})",
            type="positive",
            timeout=5000,
        )

        if on_complete:
            on_complete()

    except Exception as ex:
        progress_label.text = f"❌ エラーが発生しました"
        progress_detail.text = str(ex)
        with log_container:
            ui.label(f"❌ {traceback.format_exc()}").classes("text-caption text-red")
        ui.notify(f"解析エラー: {ex}", type="negative")
        logger.error(f"AutoML実行エラー: {traceback.format_exc()}")
