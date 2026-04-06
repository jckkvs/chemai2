import uuid
from nicegui import ui
import plotly.graph_objects as go

GLOBAL_PLOTS: dict[str, go.Figure] = {}

def render_plot_with_expand(fig: go.Figure, title: str = "", height: str = "400px"):
    """
    Plotlyグラフと「別ウィンドウで開く」ボタンを含むコンテナを生成する。
    """
    # ユニークIDで状態を保存（外部ウィンドウ用）
    plot_id = f"plot_{uuid.uuid4().hex[:8]}"
    GLOBAL_PLOTS[plot_id] = fig
    
    with ui.card().classes("full-width q-pa-sm shadow-2").style("border: 1px solid rgba(0,188,212,0.3); border-radius: 8px; background: rgba(0,20,40,0.25);"):
        # ヘッダー行（タイトル + ボタン）
        with ui.row().classes("full-width items-center justify-between q-mb-xs"):
            ui.label(title).classes("text-subtitle2 text-bold q-ml-sm")
            
            # 別ウィンドウ用ボタン
            with ui.button("別ウィンドウで開く", icon="open_in_new", on_click=lambda: ui.open(f"/view_plot/{plot_id}", new_tab=True)).props("flat dense color=cyan size=sm no-caps"):
                pass
        
        # グラフ本体
        # height を動的に設定可能にし、レスポンシブ対応を強化
        ui.plotly(fig).classes("full-width").style(f"height: {height};")
