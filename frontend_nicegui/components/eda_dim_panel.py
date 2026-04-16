"""
frontend_nicegui/components/eda_dim_panel.py
【プレミアム版】次元削減パネル
- t-SNE / PCA / UMAP 対応
- 2D / 3D 可視化サポート
- 手動特徴量選択・パラメータ調整機能
"""
import logging
import pandas as pd
import numpy as np
from nicegui import ui
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List

from backend.data.eda_core import compute_dimensionality_reduction

logger = logging.getLogger(__name__)

@ui.refreshable
def dim_reduction_panel(df: pd.DataFrame, target_col: Optional[str] = None, state: Optional[dict] = None, scale: bool = True):
    """次元削減用統合パネル"""
    state = state or {}
    
    with ui.column().classes("w-full q-gutter-md"):
        # ヘッダー領域
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column():
                ui.label("📊 次元削減・高次元可視化").classes("text-h6 text-bold text-cyan")
                ui.label("特徴量を2次元または3次元に圧縮して、データの構造を可視化します。").classes("text-caption text-grey")

        # 数値列の抽出
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            ui.label("⚠️ 次元削減には最低2つの数値列が必要です").classes("text-amber q-pa-md")
            return

        # 設定カード
        with ui.card().classes("w-full p-4 glass-card border-cyan/20"):
            with ui.row().classes("w-full q-gutter-sm items-end"):
                method = ui.select(
                    options=["PCA", "t-SNE", "UMAP"], 
                    label="手法", value="PCA"
                ).classes("w-32")
                
                n_comp = ui.select(
                    options=[2, 3], label="次元", value=2
                ).classes("w-20")
                
                features = ui.select(
                    options=numeric_cols, 
                    label="使用する特徴量", 
                    multiple=True,
                    value=numeric_cols[:min(12, len(numeric_cols))]
                ).classes("flex-grow").props("use-chips")

                with ui.row().classes("items-center mb-2"):
                    scale_toggle = ui.switch("標準化", value=scale).classes("mr-4")

                with ui.row().classes("items-center mb-2").bind_visibility_from(method, 'value', value='t-SNE'):
                    perplexity = ui.number("Perplexity", value=30, min=1).classes("w-24")
                
                ui.button("実行", icon="play_arrow", 
                          on_click=lambda: _run_and_refresh(
                              df, features.value, method.value, n_comp.value, 
                              state, perplexity.value if method.value == 't-SNE' else None,
                              scale_toggle.value
                          )).props("unelevated color=cyan")

        # 結果表示領域
        result_key = "_dim_res"
        if result_key in state:
            _render_results(state[result_key])
        else:
            with ui.column().classes("w-full items-center q-pa-xl opacity-30"):
                ui.icon("insights", size="64px")
                ui.label("「実行」ボタンをクリックして分析を開始してください")

def _run_and_refresh(df, features, method, n_components, state, perplexity=None, scale=True):
    if not features or len(features) < 2:
        ui.notify("2つ以上の特徴量を選択してください", type="warning")
        return

    # 計算中表示
    loading = ui.notification("計算中...", spinner=True, timeout=None)
    
    try:
        X = df[features].dropna()
        if X.empty:
            ui.notify("有効なデータがありません", type="warning")
            return

        params = {'scale': scale}
        if perplexity: params['perplexity'] = perplexity
        
        # Core演算
        res = compute_dimensionality_reduction(
            X.values, method=method.lower(), n_components=n_components, **params
        )
        
        if res is None:
            ui.notify(f"{method} の計算に失敗しました", type="negative")
            return

        coords, explained_var = res
        
        # 結果の保存
        state["_dim_res"] = {
            "method": method,
            "coords": coords,
            "explained_var": explained_var,
            "index": X.index.tolist(),
            "n_components": n_components,
            "features": features
        }
        
        loading.dismiss()
        ui.notify(f"{method} 計算完了", type="positive")
        dim_reduction_panel.refresh()
        
    except Exception as e:
        loading.dismiss()
        logger.error(f"DimReduction UI Error: {e}")
        ui.notify(f"エラー: {e}", type="negative")

def _render_results(res: dict):
    method = res["method"]
    n_comp = res["n_components"]
    coords = np.array(res["coords"])
    explained_var = res["explained_var"]
    
    with ui.card().classes("w-full q-pa-md bg-slate-900/40 border-slate-700"):
        ui.label(f"{method} ({n_comp}次元) 可視化結果").classes("text-subtitle2 text-bold q-mb-sm")
        
        # Plotly Figure 作成
        fig = go.Figure()
        
        if n_comp == 3:
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='markers',
                marker=dict(size=4, color='cyan', opacity=0.7, line=dict(width=0.5, color='white')),
                text=res["index"], hoverinfo='text'
            ))
            fig.update_layout(
                scene=dict(xaxis_title="Comp1", yaxis_title="Comp2", zaxis_title="Comp3"),
                margin=dict(l=0, r=0, b=0, t=0)
            )
        else:
            fig.add_trace(go.Scatter(
                x=coords[:, 0], y=coords[:, 1],
                mode='markers',
                marker=dict(size=8, color='rgba(0, 188, 212, 0.6)', line=dict(width=1, color='rgba(0, 188, 212, 1)')),
                text=res["index"], hoverinfo='text'
            ))
            fig.update_layout(
                xaxis_title="Dimension 1", yaxis_title="Dimension 2",
                hovermode='closest'
            )
        
        # 共通レイアウト
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        if explained_var is not None and method == "PCA":
            var_sum = sum(explained_var[:n_comp])
            fig.update_layout(title=f"累積寄与率: {var_sum:.1%}")

        ui.plotly(fig).classes("w-full")

        # 詳細情報の拡張
        with ui.expansion("📊 使用した特徴量と詳細", icon="list").classes("w-full opacity-60"):
            ui.label(f"特徴量数: {len(res['features'])}").classes("text-caption")
            ui.label(", ".join(res["features"])).classes("text-caption q-mb-md")
            if explained_var is not None:
                ui.label("各主成分の寄与率:").classes("text-caption font-bold")
                for i, v in enumerate(explained_var):
                    ui.label(f"PC{i+1}: {v:.4f}").classes("text-caption ml-4")
