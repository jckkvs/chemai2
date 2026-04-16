"""
EDA - 次元削減パネル（変数別色分け機能・自動最適化・プロフェッショナル可視化完全統合版）
"""
import pandas as pd
import numpy as np
from nicegui import ui
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import time
import traceback
import threading
import logging

from backend.data.eda_core import compute_dimensionality_reduction

logger = logging.getLogger(__name__)

def dim_reduction_panel(state: dict):
    """次元削減パネル（t-SNE, PCA, UMAP）"""
    
    ui.label("📊 次元削減・探索可視化").classes("text-2xl font-bold q-mb-md")
    
    ui.markdown("""
    高次元データを2次元または3次元に圧縮して可視化します。
    **インタラクティブ探索**: 計算完了後、左下のドロップダウンから任意の特徴量を選択して色分けを瞬時に切り替えられます。
    """).classes("text-body2 text-grey-7 q-mb-md")
    
    # データが読み込まれているか確認
    if "df" not in state or state["df"] is None:
        ui.warning("先にデータを読み込んでください")
        return
    
    df = state["df"]
    column_roles = state.get("column_roles", {col: "feature" for col in df.select_dtypes(include=[np.number]).columns})
    target_col = state.get("target_col")
    
    # 特徴量の分類
    feature_cols = [col for col, role in column_roles.items() if role == "feature"]
    numeric_cols = [col for col in feature_cols if col in df.select_dtypes(include=[np.number]).columns]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    
    # 色分け基準の選択肢（説明変数 ＋ 目的変数）
    all_color_options = [target_col] + numeric_cols + categorical_cols if target_col else numeric_cols + categorical_cols
    all_color_options = [opt for opt in all_color_options if opt is not None]

    if len(numeric_cols) < 2:
        ui.error("次元削減には2つ以上の数値説明変数が必要です")
        return
    
    # 設定UI
    with ui.card().classes("w-full q-mb-md"):
        ui.label("⚙️ 実行設定").classes("text-lg font-semibold q-mb-sm text-cyan")
        
        with ui.column().classes("w-full q-gutter-y-md"):
            with ui.row().classes("w-full q-gutter-md items-end"):
                # 手法選択
                method_select = ui.select(
                    options=["PCA", "t-SNE", "UMAP"],
                    label="次元削減手法",
                    value="PCA"
                ).classes("w-40").props("outlined dense")
                
                # 次元数
                n_components_select = ui.select(
                    options=[2, 3],
                    label="出力次元数",
                    value=2
                ).classes("w-32").props("outlined dense")
                
                # 使用する特徴量
                selected_features = ui.select(
                    options=numeric_cols,
                    label="計算に使用する特徴量",
                    multiple=True,
                    value=numeric_cols[:min(10, len(numeric_cols))]
                ).classes("flex-grow").props("use-chips outlined dense")

            # --- 🔥 色分け基準の動的選択 ---
            with ui.row().classes("w-full items-center q-pa-sm bg-grey-9 rounded q-mt-sm border border-cyan-8"):
                ui.icon("palette", color="cyan").classes("q-mr-sm")
                ui.label("🎨 色分け基準 (Color By):").classes("text-sm font-bold")
                
                color_select = ui.select(
                    options=[None] + all_color_options,
                    label="",
                    value=target_col if target_col in all_color_options else (numeric_cols[0] if numeric_cols else None),
                    with_input=True
                ).classes("w-72").props("dense outlined placeholder='変数を検索...'")
                
                ui.label("数値: グラデーション | カテゴリ: 離散色").classes("text-xs text-grey-5 q-ml-sm")

            # t-SNE / UMAP用パラメータ
            with ui.row().classes("q-gutter-sm").bind_visibility_from(method_select, 'value', lambda v: v in ['t-SNE', 'UMAP']):
                perplexity_ui = ui.number(label="Perplexity (t-SNE)", value=5.0, min=1.0, max=50.0).classes("w-32").props("dense outlined").bind_visibility_from(method_select, 'value', lambda v: v == 't-SNE')
                lr_ui = ui.number(label="LR (t-SNE)", value=200.0).classes("w-32").props("dense outlined").bind_visibility_from(method_select, 'value', lambda v: v == 't-SNE')
                neighbors_ui = ui.number(label="Neighbors (UMAP)", value=15).classes("w-32").props("dense outlined").bind_visibility_from(method_select, 'value', lambda v: v == 'UMAP')
        
        # 実行ボタン
        ui.button(
            "📊 次元削減を実行",
            on_click=lambda: _run_dim_reduction_with_progress(
                df,
                selected_features.value,
                method_select.value,
                n_components_select.value,
                state,
                color_select,
                perplexity=perplexity_ui.value if method_select.value == 't-SNE' else None,
                learning_rate=lr_ui.value if method_select.value == 't-SNE' else None,
                n_neighbors=neighbors_ui.value if method_select.value == 'UMAP' else None
            )
        ).props("unelevated color=primary icon=play_arrow").classes("q-mt-md full-width")
    
    # 進捗表示領域
    progress_container = ui.column().classes("w-full q-mt-md")
    state["_dim_reduction_progress"] = progress_container
    
    # 表示用のコンテナ（結果がここに描画される）
    results_container = ui.column().classes("w-full q-mt-lg")
    
    def refresh_results():
        """結果表示エリアを再描画（色分け変更時など）"""
        results_container.clear()
        with results_container:
            _render_multiple_results(state, color_select)
            
    # 色分け基準が変更されたら結果エリアを再描画 (リアクティブ連動)
    color_select.on_value_change(refresh_results)
    
    # 初回描画（キャッシュがあれば）
    if "dim_reduction_results" in state:
        refresh_results()


def _run_dim_reduction_with_progress(df: pd.DataFrame, features: list, method: str, 
                                     n_components: int, state: dict, color_select_ui,
                                     perplexity: Optional[float] = None,
                                     learning_rate: Optional[float] = None,
                                     n_neighbors: Optional[int] = None):
    """次元削減を実行し、結果を更新"""
    if not features or len(features) < 2:
        ui.notify("2つ以上の特徴量を選択してください", type="warning")
        return
    
    progress_container = state.get("_dim_reduction_progress")
    if not progress_container: return
    progress_container.clear()
    
    with progress_container:
        ui.label(f"🔄 {method} 実行中...").classes("text-bold text-cyan")
        progress_bar = ui.linear_progress(value=0.1, show_value=False).classes("w-full")
        status = ui.label("データ前処理中...").classes("text-xs text-grey-5")
        log_area = ui.label("").classes("text-xs text-grey-500 font-mono q-mt-xs bg-grey-9 q-pa-xs full-width border border-grey-8")
    
    # データ形状確認
    X_clean = df[features].dropna()
    n_samples, n_features = X_clean.shape
    log_area.text = f"🔍 Input: {n_samples} samples x {n_features} features"
    
    # 🔧 t-SNE perplexity 自動調整ロジック
    tsne_params = {}
    if method == 't-SNE':
        default_perplexity = perplexity if perplexity is not None else 5.0
        max_perplexity = max(1.0, n_samples / 2 - 1e-5)
        safe_perplexity = max(1.0, min(default_perplexity, max_perplexity))
        tsne_params['perplexity'] = safe_perplexity
        if learning_rate: tsne_params['learning_rate'] = learning_rate
        
        if abs(safe_perplexity - default_perplexity) > 1e-3:
            log_area.text += f"\n⚠️ Perplexity adjusted: {default_perplexity} -> {safe_perplexity:.2f} (n_samples={n_samples})"

    # 🔧 UMAP parameters
    umap_params = {}
    if method == 'UMAP':
        if n_neighbors: umap_params['n_neighbors'] = n_neighbors

    def on_finished(result_df, explained):
        progress_bar.value = 1.0
        status.text = "✅ 計算完了"
        
        if "dim_reduction_results" not in state:
            state["dim_reduction_results"] = {}
        
        res_key = f"{method}_{n_components}D_{int(time.time())}"
        state["dim_reduction_results"][res_key] = {
            'method': method,
            'n_components': n_components,
            'data': result_df,
            'features': features,
            'explained_variance': explained,
            'timestamp': time.time()
        }
        
        # 画面を更新（color_select_ui の変更イベント経由ではなく直接呼ぶ）
        ui.notify(f"{method} の計算が完了しました", type="positive")
        # 循環参照を避けるため、ui.timerなどで遅延実行するか、直接UIコンテナを操作
        ui.timer(0.1, lambda: color_select_ui.set_value(color_select_ui.value), once=True)

    def compute():
        try:
            status.text = f"{method} モデル Fitting 開始..."
            progress_bar.value = 0.3
            
            kwargs = {**tsne_params, **umap_params}
            coords, explained = compute_dimensionality_reduction(
                X_clean.values, 
                method=method.lower(), 
                n_components=n_components,
                **kwargs
            )
            
            if coords is None:
                raise ValueError(f"{method} の計算結果が None です")
                
            # 結果を結合
            coord_cols = [f"{method}_dim{i+1}" for i in range(n_components)]
            result_df = pd.DataFrame(coords, columns=coord_cols, index=X_clean.index)
            full_res_df = pd.concat([df.loc[X_clean.index], result_df], axis=1)
            
            on_finished(full_res_df, explained)
        except Exception as e:
            logger.error(traceback.format_exc())
            ui.notify(f"計算エラー: {str(e)}", type="negative")
            status.text = "❌ エラー発生"

    threading.Thread(target=compute, daemon=True).start()


def _render_multiple_results(state: dict, color_select_ui):
    """すべての次元削減結果をタブで表示"""
    results = state.get("dim_reduction_results", {})
    if not results: return
    
    # 最新の結果をデフォルトにする
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]['timestamp'], reverse=True)
    
    with ui.tabs().classes("w-full bg-grey-9 rounded-t") as tabs:
        tab_objs = []
        for key in sorted_keys:
            res = results[key]
            tab_objs.append((key, ui.tab(key, label=f"{res['method']} ({res['n_components']}D)")))
    
    with ui.tab_panels(tabs, value=tab_objs[0][1] if tab_objs else None).classes("w-full bg-grey-10 q-pa-md rounded-b shadow-2xl"):
        for key, tab in tab_objs:
            with ui.tab_panel(tab):
                _render_single_view(results[key], color_select_ui.value)


def _render_single_view(result: Dict[str, Any], color_var: Optional[str]):
    """単一の結果を描画"""
    method = result['method']
    df = result['data']
    n_comp = result['n_components']
    
    # 統計情報表示
    if method == 'PCA' and result['explained_variance'] is not None:
        ev = result['explained_variance']
        with ui.row().classes("w-full q-mb-sm items-center q-gutter-x-md bg-cyan-9/20 q-pa-sm rounded border border-cyan-8/30"):
            ui.icon("analytics", color="cyan")
            ui.label("PCA 説明分散率:").classes("text-xs font-bold")
            for i, v in enumerate(ev):
                ui.label(f"PC{i+1}: {v:.1%}")
    
    # プロット領域
    with ui.column().classes("w-full h-[600px] flex items-center justify-center"):
        fig = _create_fig(df, method, n_comp, color_var)
        ui.plotly(fig).classes("w-full h-full").style("min-height: 500px;")


def _create_fig(df: pd.DataFrame, method: str, n_components: int, color_var: Optional[str]):
    """Plotly Expressを使用して正方形プロットを作成 (Dark Theme)"""
    dim_cols = [c for c in df.columns if c.startswith(f"{method}_dim")]
    
    fig_args = {
        'data_frame': df,
        'x': dim_cols[0],
        'y': dim_cols[1],
        'template': 'plotly_dark',
        'hover_data': [df.index.name or 'index'],
        'title': f"{method} Analysis"
    }
    
    # 色分け適用
    if color_var and color_var in df.columns:
        fig_args['color'] = color_var
        if pd.api.types.is_numeric_dtype(df[color_var]):
            fig_args['color_continuous_scale'] = px.colors.sequential.Viridis
        else:
            fig_args['color_discrete_sequence'] = px.colors.qualitative.Bold
            
    if n_components == 3 and len(dim_cols) >= 3:
        fig_args['z'] = dim_cols[2]
        fig = px.scatter_3d(**fig_args)
        fig.update_layout(scene=dict(aspectmode='cube'))
    else:
        fig = px.scatter(**fig_args)
        fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
        
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, Roboto, sans-serif"),
        title_font_size=20
    )
    return fig
