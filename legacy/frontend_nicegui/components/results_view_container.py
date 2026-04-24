"""
frontend_nicegui/components/results_view_container.py

解析結果、レポート生成、実験ダッシュボードを統合したコンポーネント。
"""
from __future__ import annotations

from typing import Any
import logging
logger = logging.getLogger(__name__)

from nicegui import ui

@ui.refreshable
def render_results_view_container(state: dict[str, Any]) -> None:
    """結果表示コンテナ — stateとstorageの両方から結果を取得"""
    
    # 状態の更新関数を登録
    state["_refresh_results"] = render_results_view_container.refresh

    # stateから結果を取得
    result = state.get("automl_result")
    results = state.get("automl_results", {})
    
    # stateにない場合はstorageから復元（シリアライズ可能な形式のみ）
    if result is None and not results:
        try:
            from nicegui import app
            import pandas as pd
            storage_result = app.storage.user.get('automl_result')
            # bytes や DataFrame などの非JSONセーフなデータが混入していないか確認
            if storage_result and not isinstance(storage_result, (bytes, pd.DataFrame)):
                state["automl_result"] = storage_result
                state["automl_results"] = {"デフォルト": storage_result}
                result = storage_result
                results = {"デフォルト": storage_result}
                logger.info("✓ storageから結果を復元しました")
        except Exception as e:
            logger.warning(f"storageからの復元失敗: {e}")
    
    # 結果がない場合
    if result is None and not results:
        with ui.card().classes('w-full glass-card q-pa-lg'):
            with ui.column().classes('items-center justify-center q-pa-xl'):
                ui.icon('analytics', size='64px', color='grey-5')
                ui.label('解析結果がまだありません').classes('text-h6 text-grey-5 q-mt-md')
                ui.label('「🚀 解析開始」ボタンを押して解析を実行してください。').classes('text-caption text-grey-6')
        return

    with ui.tabs().classes("full-width").props("dense active-color=cyan indicator-color=cyan") as sub_tabs:
        tab_results = ui.tab("results", label="📊 モデル評価", icon="analytics")
        tab_report = ui.tab("report", label="📝 レポート生成", icon="summarize")
        tab_dashboard = ui.tab("dashboard", label="🔬 実験ダッシュ", icon="dashboard")

    with ui.tab_panels(sub_tabs, value=tab_results).classes("full-width bg-transparent"):
        # --- 1. モデル評価詳細 ---
        with ui.tab_panel(tab_results):
            from frontend_nicegui.components.results_tab import render_results_tab
            render_results_tab(state)

        # --- 2. レポート生成 ---
        with ui.tab_panel(tab_report):
            from frontend_nicegui.pages.export_panel import render_export_panel
            render_export_panel(state)

        # --- 3. 実験ダッシュボード ---
        with ui.tab_panel(tab_dashboard):
            from frontend_nicegui.pages.experiment_comparison import render_experiment_comparison
            render_experiment_comparison(state)
