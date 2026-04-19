"""
frontend_nicegui/components/eda_advanced_panel.py

Advanced EDA Panel — OFAT/Conflict検出結果表示
"""
from nicegui import ui
import pandas as pd
from backend.data.eda_advanced import detect_quasi_ofat_patterns, detect_conflict_data

def render_eda_advanced_panel(state: dict):
    """Advanced EDA の結果を表示する。"""
    df = state.get("df")
    target_col = state.get("target_col")
    
    if df is None:
        ui.notify("データを読み込んでください", type="warning")
        return

    ui.label("🔬 物理的妥当性・コンフリクト解析").classes("text-h5 font-bold q-mb-md")
    
    with ui.row().classes("full-width q-col-gutter-lg"):
        # --- OFAT検出 ---
        with ui.column().classes("col-12 col-md-6"):
            with ui.card().classes("full-width q-pa-md"):
                ui.label("🏃 Quasi-OFATパターン検出").classes("text-h6 text-amber")
                ui.markdown("""
                一変数のみが変化しているデータペアを抽出します。
                物性変化の感度確認に役立ちますが、数が多すぎるとサンプリングの偏りを示唆します。
                """).classes("text-caption text-grey-5")
                
                def _run_ofat():
                    ofat_results = detect_quasi_ofat_patterns(df)
                    ofat_container.clear()
                    with ofat_container:
                        if not ofat_results:
                            ui.label("OFATパターンは見つかりませんでした").classes("text-grey q-mt-md")
                        else:
                            ui.label(f"{len(ofat_results)}件のペアを検出しました").classes("text-subtitle2 q-mt-md")
                            rows = [
                                {"Pair": f"{r['idx1']} & {r['idx2']}", "Variable": r['variable'], "Amount": f"{r['change_amount']:.4f}"}
                                for r in ofat_results[:20]
                            ]
                            ui.table(
                                columns=[
                                    {"name": "Pair", "label": "ペア(Index)", "field": "Pair"},
                                    {"name": "Variable", "label": "変化変数", "field": "Variable"},
                                    {"name": "Amount", "label": "変化量(scaled)", "field": "Amount"}
                                ],
                                rows=rows
                            ).classes("full-width").props("dense flat bordered")

                ui.button("OFATパターンをスキャン", on_click=_run_ofat).props("outline color=amber no-caps")
                ofat_container = ui.column().classes("full-width")

        # --- Conflict検出 ---
        with ui.column().classes("col-12 col-md-6"):
            with ui.card().classes("full-width q-pa-md"):
                ui.label("⚔️ Conflict（データ矛盾）検出").classes("text-h6 text-red")
                ui.markdown("""
                「特徴量がほぼ同じなのに目的変数が大きく異なる」サンプルを抽出します。
                測定ミス、または重要な特徴量が欠落している可能性があります。
                """).classes("text-caption text-grey-5")
                
                def _run_conflict():
                    if not target_col:
                        ui.notify("目的変数を設定してください", type="warning")
                        return
                    
                    conflict_results = detect_conflict_data(df, target_col)
                    conflict_container.clear()
                    with conflict_container:
                        if not conflict_results:
                            ui.label("Conflictデータは見つかりませんでした").classes("text-green q-mt-md")
                        else:
                            ui.label(f"{len(conflict_results)}件の矛盾ペアを検出しました").classes("text-red text-subtitle2 q-mt-md")
                            rows = [
                                {
                                    "Pair": f"{r['idx1']} & {r['idx2']}", 
                                    "F_Diff": f"{r['feature_diff']:.4f}", 
                                    "T_Diff": f"{r['target_diff']:.4f}"
                                }
                                for r in conflict_results[:20]
                            ]
                            ui.table(
                                columns=[
                                    {"name": "Pair", "label": "ペア(Index)", "field": "Pair"},
                                    {"name": "F_Diff", "label": "特徴量差", "field": "F_Diff"},
                                    {"name": "T_Diff", "label": "目的変数差", "field": "T_Diff"}
                                ],
                                rows=rows
                            ).classes("full-width").props("dense flat bordered")

                ui.button("Conflictデータをスキャン", on_click=_run_conflict).props("outline color=red no-caps")
                conflict_container = ui.column().classes("full-width")
