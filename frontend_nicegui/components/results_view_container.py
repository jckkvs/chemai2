"""
frontend_nicegui/components/results_view_container.py

解析結果、レポート生成、実験ダッシュボードを統合したコンポーネント。
"""
from __future__ import annotations

from typing import Any
from nicegui import ui

@ui.refreshable
def render_results_view_container(state: dict[str, Any]) -> None:
    """結果・レポートタブ全体を描画する。"""
    
    # 状態の更新関数を登録（初回描画時に行う）
    state["_refresh_results"] = render_results_view_container.refresh

    # [追加] stateに結果がない場合、storageから復元
    if state.get("automl_result") is None and not state.get("automl_results"):
        from nicegui import app
        saved_result = app.storage.user.get('automl_result')
        if saved_result:
            state["automl_result"] = saved_result
            state["automl_results"] = {"デフォルト": saved_result}
            logger.info("✓ results_view_container: app.storage.user から結果を復元しました")

    if state.get("automl_result") is None and not state.get("automl_results"):
        with ui.card().classes("glass-card q-pa-xl items-center justify-center text-center"):
            ui.icon("analytics", color="grey-7", size="xl").classes("q-mb-md")
            ui.label("解析結果がまだありません").classes("text-h6 text-grey-5")
            ui.label("「🚀 解析開始」ボタンを押して解析を実行してください。").classes("text-grey-6 q-mt-sm")
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
