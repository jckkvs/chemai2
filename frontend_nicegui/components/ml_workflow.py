"""
frontend_nicegui/components/ml_workflow.py

機械学習ワークフロー設定の統合コンポーネント。
パイプライン構成、単調性制約、CV設定などをサブタブで管理。
"""
from __future__ import annotations

from typing import Any
from nicegui import ui

@ui.refreshable
def render_ml_workflow(state: dict[str, Any]) -> None:
    """機械学習ワークフロー設定タブ全体を描画する。"""
    
    # ── 🔍 診断ログ (サーバーコンソール) ──
    df = state.get("df")
    target_col = state.get("target_col")
    print(f"🔍 [ML Workflow Render] df_exists={df is not None}, target_col='{target_col}', "
          f"cv_key='{state.get('cv_key')}', models={len(state.get('selected_models', []))}")

    # 1. データ未読込
    if df is None:
        with ui.card().classes("glass-card q-pa-xl items-center justify-center text-center full-width"):
            ui.icon("info", color="amber", size="xl").classes("q-mb-md")
            ui.label("⚠️ データを読み込んでください").classes("text-h6 text-amber")
            ui.label("機械学習を行うには、まずデータ管理タブでファイルをアップロードしてください。").classes("text-grey-6 q-mb-md")
            
            ui.button(
                "📁 データ管理へ移動", 
                on_click=lambda: state.get("_switch_to_data", lambda: None)()
            ).props("unelevated color=cyan-9 no-caps")
        return

    # 2. 目的変数未設定
    if not target_col:
        with ui.card().classes("glass-card q-pa-xl items-center justify-center text-center full-width"):
            ui.icon("label_important", color="cyan", size="xl").classes("q-mb-md")
            ui.label("🎯 目的変数が未設定です").classes("text-h6 text-cyan")
            ui.label("どの列を予測するか「列の役割」タブで設定してください。").classes("text-grey-6 q-mb-md")
            
            ui.button(
                "🏷️ 列の役割を設定する", 
                on_click=lambda: state.get("_switch_to_column_role", lambda: None)()
            ).props("unelevated color=cyan-9 no-caps")
        return

    with ui.tabs().classes("full-width").props("dense active-color=cyan indicator-color=cyan") as sub_tabs:
        tab_pipeline = ui.tab("pipeline", label="⚙️ パイプライン設定", icon="tune")
        tab_monotonic = ui.tab("monotonic", label="📈 単調性制約", icon="trending_up")
        tab_history = ui.tab("history", label="📜 解析履歴", icon="history")

    with ui.tab_panels(sub_tabs, value=tab_pipeline).classes("full-width bg-transparent"):
        # --- 1. パイプライン設定 ---
        with ui.tab_panel(tab_pipeline):
            with ui.column().classes("full-width q-gutter-y-md"):
                # 設定整合性チェッカー
                from frontend_nicegui.components.settings_checker import render_settings_checker
                render_settings_checker(state)
                
                ui.separator()
                
                # データリークチェック
                from frontend_nicegui.components.leakage_check_ui import render_leakage_check_panel
                render_leakage_check_panel(state)
                
                ui.separator()
                
                # CV設定
                from frontend_nicegui.components.cv_config_ui import render_cv_config
                render_cv_config(state)
                
                ui.separator()
                
                # 詳細なパイプライン構成 (スケーラー、モデル等)
                from frontend_nicegui.components.pipeline_config_ui import render_pipeline_config
                render_pipeline_config(state)
                
                ui.separator()
                
                # 解析後の自動処理
                from frontend_nicegui.components.post_analysis_config import render_post_analysis_config
                render_post_analysis_config(state)

        # --- 2. 単調性制約 ---
        with ui.tab_panel(tab_monotonic):
            from frontend_nicegui.components.monotonicity_constraint_panel import render_monotonicity_constraint_panel
            render_monotonicity_constraint_panel(state)

        # --- 3. 解析履歴 ---
        with ui.tab_panel(tab_history):
            _render_analysis_history(state)

def _render_analysis_history(state: dict[str, Any]) -> None:
    """過去の解析実行結果をリスト表示する（プレースホルダまたは既存ロジック）。"""
    ui.label("🧪 解析履歴").classes("text-h6 q-mb-md")
    # ここに履歴表示ロジックを実装（現在はプレースホルダ）
    ui.label("（履歴機能は今後のアップデートで追加予定です）").classes("text-grey-5")
