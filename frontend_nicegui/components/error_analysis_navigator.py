"""
frontend_nicegui/components/error_analysis_navigator.py

Error Analysis Navigator — 予測エラーの詳細航法（結果タブ用）
"""
from nicegui import ui
import pandas as pd
import numpy as np
from backend.ml.error_analysis import ErrorAnalyzer
from backend.data.chemical_sync import ChemicalStructureSync

def render_error_navigator(ar, state: dict):
    """結果オブジェクト(ar)から誤差を抽出し、ナビゲーションUIを表示。"""
    
    y_true = getattr(ar, "oof_true", None)
    y_pred = getattr(ar, "oof_predictions", None)
    df_raw = state.get("df")
    smiles_col = state.get("smiles_col")
    
    if y_true is None or y_pred is None or df_raw is None:
        ui.label("誤差分析に必要なデータが不足しています。").classes("text-grey")
        return

    # データの調整（OOFのインデックスに合わせる）
    # ar.oof_indices がある場合はそれを使う
    oof_indices = getattr(ar, "oof_indices", range(len(y_true)))
    work_df = df_raw.iloc[oof_indices].copy()
    
    analyzer = ErrorAnalyzer(work_df, y_true, y_pred, smiles_col=smiles_col)
    worst_samples = analyzer.get_worst_samples(top_n=10)
    sync = ChemicalStructureSync(df_raw)

    ui.label("🔍 予測エラー詳細解析").classes("text-h6 text-red-4 q-mt-md")
    
    with ui.row().classes("full-width q-col-gutter-lg"):
        # --- 左: ワーストサンプルリスト ---
        with ui.column().classes("col-12 col-md-7"):
            ui.label("誤差の大きいサンプル TOP10").classes("text-subtitle2")
            
            with ui.scroll_area().style("height: 400px;").classes("full-width"):
                for idx, row in worst_samples.iterrows():
                    with ui.card().classes("q-pa-sm q-mb-sm cursor-pointer hover-bounce").on("click", lambda _idx=idx: _show_detail(_idx)):
                        with ui.row().classes("items-center justify-between full-width"):
                            with ui.row().classes("items-center q-gutter-md"):
                                ui.label(f"#{idx}").classes("text-bold text-grey-5")
                                b64 = sync.get_structure_b64(idx, size=(60, 60))
                                if b64:
                                    ui.image(f"data:image/png;base64,{b64}").style("width: 50px; height: 50px;")
                                ui.label(f"Error: {row['abs_error']:.4f}").classes("text-red text-bold")
                            ui.icon("chevron_right", color="grey-3")

        # --- 右: 詳細 & 解決提案 ---
        with ui.column().classes("col-12 col-md-5"):
            detail_container = ui.card().classes("full-width q-pa-md").style("height: 400px; overflow-y: auto;")
            with detail_container:
                ui.label("サンプルを選択して詳細を表示").classes("text-grey text-center full-height flex items-center justify-center")

    def _show_detail(idx: int):
        detail_container.clear()
        row = df_raw.iloc[idx]
        actual = analyzer.y_true[np.where(worst_samples.index == idx)[0][0]]
        pred = analyzer.y_pred[np.where(worst_samples.index == idx)[0][0]]
        
        with detail_container:
            ui.label(f"📄 サンプル詳細 (Index: {idx})").classes("text-h6")
            
            with ui.row().classes("full-width justify-center q-my-md"):
                b64 = sync.get_structure_b64(idx, size=(200, 200))
                if b64: ui.image(f"data:image/png;base64,{b64}").style("width: 180px; height: 180px;")
            
            with ui.row().classes("full-width q-gutter-sm justify-center"):
                ui.badge(f"実測: {actual:.4f}", color="grey-8")
                ui.badge(f"予測: {pred:.4f}", color="red-10")
            
            ui.separator().classes("q-my-md")
            ui.label("🔬 改善の提案").classes("text-subtitle2 text-cyan")
            suggestions = analyzer.suggest_next_steps()
            for s in suggestions:
                ui.markdown(f"- {s}").classes("text-caption")

    # 全体インサイト
    ui.separator().classes("q-my-md")
    with ui.card().classes("full-width bg-red-10 q-pa-md").style("background: rgba(255, 0, 0, 0.05); border: 1px solid rgba(255, 0, 0, 0.1);"):
        ui.label("📊 エラーの構造的傾向").classes("text-subtitle2 text-red-3")
        sim_clusters = analyzer.analyze_chemical_similarity(worst_samples)
        if sim_clusters:
            ui.label(f"特定の化学的部分構造をもつ {len(sim_clusters)} 組のペアで一貫して高い誤差が出ています。").classes("text-caption")
            ui.label("→ この系統のデータを追加、または専用の記述子の導入を検討してください。").classes("text-caption text-bold")
        else:
            ui.label("エラーは化学的に分散しています。全体的な記述子の不足、またはデータのノイズが疑われます。").classes("text-caption")
