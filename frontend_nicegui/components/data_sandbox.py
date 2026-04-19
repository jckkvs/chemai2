"""
frontend_nicegui/components/data_sandbox.py

Data Sandbox UI — 自由探索ワークベンチ
"""
from nicegui import ui
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

from backend.data.flexible_explorer import FlexibleDataView
from backend.data.chemical_sync import ChemicalStructureSync

def render_data_sandbox(state: dict):
    """Data Sandbox のメインレンダラー"""
    df = state.get("df")
    if df is None:
        ui.notify("データを読み込んでください", type="warning")
        return

    # インスタンスの初期化（stateに保持して共有）
    if "sandbox_engine" not in state or state.get("_sandbox_df_id") != id(df):
        state["sandbox_engine"] = FlexibleDataView(df)
        state["sandbox_sync"] = ChemicalStructureSync(df)
        state["_sandbox_df_id"] = id(df)

    engine: FlexibleDataView = state["sandbox_engine"]
    sync: ChemicalStructureSync = state["sandbox_sync"]

    with ui.row().classes("full-width no-wrap q-gutter-md").style("height: 85vh;"):
        # --- 左パネル: パラメータ設定 ---
        with ui.column().classes("col-3 glass-card q-pa-md").style("min-width: 300px; overflow-y: auto;"):
            ui.label("🛠️ 探索パラメータ").classes("text-h6 text-cyan")
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            ui.label("軸の選択").classes("text-caption text-grey q-mt-sm")
            x_ax = ui.select(num_cols, label="X軸", value=num_cols[0]).classes("full-width").props("dense outlined")
            y_ax = ui.select(num_cols, label="Y軸", value=num_cols[1] if len(num_cols) > 1 else num_cols[0]).classes("full-width").props("dense outlined")
            color_ax = ui.select(["None"] + num_cols, label="色分け", value="None").classes("full-width").props("dense outlined")

            ui.separator().classes("q-my-md")
            ui.label("データ加工").classes("text-caption text-grey")
            
            def _apply_transform_ui(col, ttype):
                engine.apply_transform(col, ttype)
                _update_plots()

            with ui.expansion("📈 変換設定", icon="transform").classes("full-width"):
                for col in num_cols[:10]: # 10列まで
                    with ui.row().classes("items-center justify-between full-width q-mb-xs"):
                        ui.label(col).classes("text-caption truncate").style("max-width: 100px;")
                        ui.select(["none", "log10", "zscore", "binning"], value="none",
                                 on_change=lambda e, c=col: _apply_transform_ui(c, e.value)).props("dense outlined style='width:100px;'")

        # --- 中央パネル: プロット・テーブル ---
        with ui.column().classes("col-6 q-gutter-md"):
            # 散布図
            scatter_card = ui.card().classes("full-width q-pa-none").style("height: 60%; position: relative;")
            with scatter_card:
                scatter_plot = ui.plotly({}).classes("full-width").style("height: 100%;")
            
            # テーブル
            with ui.card().classes("full-width").style("height: 35%;"):
                ui.label("📋 選択データの詳細").classes("text-caption text-grey")
                selection_table = ui.table(
                    columns=[], rows=[], pagination={"rowsPerPage": 10}
                ).classes("full-width").props("dense flat bordered")

        # --- 右パネル: 化学構造・インサイト ---
        with ui.column().classes("col-3 q-gutter-md"):
            with ui.card().classes("full-width q-pa-md items-center justify-center").style("height: 300px;"):
                ui.label("⚗️ 化学構造").classes("text-caption text-grey q-mb-sm")
                mol_img = ui.image("").style("width: 250px; height: 250px; border-radius: 8px;")
                smi_label = ui.label("").classes("text-caption text-grey-5 truncate full-width text-center")

            with ui.card().classes("full-width flex-grow q-pa-md"):
                ui.label("💡 インサイト").classes("text-h6 text-purple")
                ui.markdown("""
                - **選択されたクラスタ** の平均目的変数が他より高いようです。
                - **Conflictデータ** がこの領域に集中しています。
                """).classes("text-body2")

    # --- インタラクション・更新 ---
    def _update_plots():
        current_df = engine.get_data()
        
        fig = go.Figure()
        
        # 色分け設定
        color_val = None
        if color_ax.value != "None":
            color_val = current_df[color_ax.value]

        fig.add_trace(go.Scatter(
            x=current_df[x_ax.value],
            y=current_df[y_ax.value],
            mode="markers",
            marker=dict(
                size=8,
                color=color_val,
                colorscale="Viridis",
                showscale=True if color_val is not None else False,
                line=dict(width=1, color="white")
            ),
            customdata=current_df.index,
            hovertemplate="Index: %{customdata}<br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_title=x_ax.value,
            yaxis_title=y_ax.value,
            hovermode="closest",
            clickmode="event+select"
        )
        
        scatter_plot.update_figure(fig)

    # ホバー/クリックイベント
    def _handle_hover(e: Any):
        if not e or "points" not in e: return
        point_idx = e["points"][0].get("customdata")
        if point_idx is None: return
        
        # 構造表示
        b64 = sync.get_structure_b64(point_idx)
        mol_img.set_source(f"data:image/png;base64,{b64}")
        smi_label.text = sync.get_smiles(point_idx)

    def _handle_select(e: Any):
        if not e or "points" not in e: return
        indices = [p["customdata"] for p in e["points"]]
        engine.set_selection(indices)
        
        selected_df = engine.get_selected_data()
        if not selected_df.empty:
            cols = [{"name": c, "label": c, "field": c} for c in selected_df.columns[:5]]
            rows = selected_df.head(50).to_dict("records")
            selection_table.props(f'columns={cols} rows={rows}')

    # Plotlyイベントのバインド
    scatter_plot.on("plotly_hover", _handle_hover)
    scatter_plot.on("plotly_selected", _handle_select)
    
    # 軸変更時に更新
    x_ax.on_value_change(_update_plots)
    y_ax.on_value_change(_update_plots)
    color_ax.on_value_change(_update_plots)

    # 初期描画
    _update_plots()
