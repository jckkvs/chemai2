"""
EDA - 次元削減パネル（変数別色分け機能・特徴量選択・自動カラーマッピング完全実装）
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
    
    ui.label("📊 次元削減・可視化").classes("text-2xl font-bold q-mb-md")
    
    ui.markdown("""
    高次元データを2次元または3次元に圧縮して可視化します。
    **新機能**: 「色分け基準」を選択することで、特定の特徴量（数値・カテゴリ問わず）の分布を重ねて探索できます。
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
        ui.label("⚙️ 設定").classes("text-lg font-semibold q-mb-sm text-cyan")
        
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
                    label="使用する特徴量",
                    multiple=True,
                    value=numeric_cols[:min(10, len(numeric_cols))]
                ).classes("flex-grow").props("use-chips outlined dense")

            # --- 色分け設定 (探索用) ---
            with ui.row().classes("w-full items-center q-pa-sm bg-grey-9 rounded q-mt-sm"):
                ui.icon("palette", color="cyan").classes("q-mr-sm")
                ui.label("🎨 色分け基準 (Color By):").classes("text-sm font-bold")
                
                color_select = ui.select(
                    options=[None] + all_color_options,
                    label="",
                    value=None,
                    with_input=True
                ).classes("w-72").props("dense outlined placeholder='変数を検索...'")
                
                ui.label("数値: グラデーション | カテゴリ: 離散色").classes("text-xs text-grey-5 q-ml-md")

            # t-SNE用パラメータ
            with ui.row().classes("q-gutter-sm").bind_visibility_from(method_select, 'value', lambda v: v == 't-SNE'):
                perp_input = ui.number(label="Perplexity", value=5.0, min=1.0, max=50.0).classes("w-32").props("dense outlined")
                lr_input = ui.number(label="LR", value=200.0).classes("w-32").props("dense outlined")
        
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
                perplexity=perp_input.value if method_select.value == 't-SNE' else None,
                learning_rate=lr_input.value if method_select.value == 't-SNE' else None
            )
        ).props("unelevated color=primary icon=play_arrow").classes("q-mt-md full-width")
    
    # 進捗表示領域
    progress_container = ui.column().classes("w-full q-mt-md")
    state["_dim_reduction_progress"] = progress_container
    
    # キャッシュされた結果を再描画
    if "dim_reduction_results" in state:
        _render_multiple_results(state["dim_reduction_results"], state, color_select)


def _run_dim_reduction_with_progress(df: pd.DataFrame, features: list, method: str, 
                                     n_components: int, state: dict, color_select_ui,
                                     perplexity: Optional[float] = None,
                                     learning_rate: Optional[float] = None):
    """次元削減を実行し、結果を更新"""
    if not features or len(features) < 2:
        ui.notify("2つ以上の特徴量を選択してください", type="warning")
        return
    
    progress_container = state.get("_dim_reduction_progress")
    if not progress_container: return
    progress_container.clear()
    
    with progress_container:
        ui.label(f"🔄 {method} 実行中...").classes("text-bold")
        progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
        status = ui.label("計算準備中...").classes("text-xs text-grey-5")
        log_area = ui.label("").classes("text-xs text-grey-500 font-mono q-mt-xs bg-grey-9 q-pa-xs full-width")
    
    # 🔍 デバッグ情報：データ形状を取得
    X_for_shape = df[features].dropna()
    n_samples, n_features = X_for_shape.shape
    log_area.text = f"🔍 Data Shape: {n_samples} samples × {n_features} features"
    print(f"[DimReduction DEBUG] {method} START. Shape: {X_for_shape.shape}")

    # t-SNE特有のチェック
    if method == 't-SNE' and perplexity is not None:
        if perplexity >= n_samples:
            ui.notify(f"Perplexity ({perplexity}) はサンプル数 ({n_samples}) 未満である必要があります。", type="negative")
            return

    def on_finished(result_df, explained):
        progress_bar.value = 1.0
        status.text = "✅ 完了"
        
        if "dim_reduction_results" not in state:
            state["dim_reduction_results"] = {}
        
        res_key = f"{method}_{n_components}d_{int(time.time())}"
        state["dim_reduction_results"][res_key] = {
            'method': method,
            'n_components': n_components,
            'data': result_df,
            'features': features,
            'explained_variance': explained
        }
        # 再描画
        _render_multiple_results(state["dim_reduction_results"], state, color_select_ui)

    # 非同期実行 (簡略化)
    def compute():
        try:
            from backend.data.eda_core import compute_dimensionality_reduction
            X = df[features].dropna()
            
            # パラメータを渡す
            params = {}
            if method == 't-SNE':
                if perplexity is not None: params['perplexity'] = perplexity
                if learning_rate is not None: params['learning_rate'] = learning_rate

            coords, explained = compute_dimensionality_reduction(X.values, method=method.lower(), n_components=n_components, **params)
            
            # 結果を結合
            coord_cols = [f"{method}_dim{i+1}" for i in range(n_components)]
            result_df = pd.DataFrame(coords, columns=coord_cols, index=X.index)
            # 全列を保持 (色分け用)
            full_res_df = pd.concat([df.loc[X.index], result_df], axis=1)
            
            ui.run_javascript(f"console.log('{method} computed')")
            on_finished(full_res_df, explained)
        except Exception as e:
            logger.error(traceback.format_exc())
            ui.notify(f"エラー: {str(e)}", type="negative")

    threading.Thread(target=compute, daemon=True).start()


def _render_multiple_results(results: Dict[str, Any], state: dict, color_select_ui):
    """すべての次元削減結果をタブで表示"""
    # 既存のコンテナがあればクリア（通常はdim_reduction_panelの呼び出し元が管理するが、ここではパネル内で完結させる）
    if "_results_container" not in state:
        state["_results_container"] = ui.column().classes("w-full q-mt-lg border-t border-grey-8 q-pt-md")
    
    container = state["_results_container"]
    container.clear()
    
    with container:
        ui.label("📈 次元削減・可視化結果").classes("text-xl font-bold q-mb-md text-cyan")
        
        with ui.tabs().classes("w-full") as tabs:
            tab_objs = []
            for key, res in list(results.items())[::-1]: # 最新順
                tab_objs.append((key, ui.tab(key, label=f"{res['method']} ({res['n_components']}D)")))
        
        with ui.tab_panels(tabs, value=tab_objs[0][1] if tab_objs else None).classes("w-full bg-grey-10 q-pa-md rounded shadow-inner"):
            for key, tab in tab_objs:
                with ui.tab_panel(tab):
                    _render_single_plot(results[key], state, color_select_ui)


def _render_single_plot(result: Dict[str, Any], state: dict, color_select_ui):
    """単一のプロットをレンダリング"""
    method = result['method']
    df = result['data']
    n_comp = result['n_components']
    
    # 説明分散率 (PCA)
    if method == 'PCA' and result['explained_variance'] is not None:
        ev = result['explained_variance']
        ui.label(f"説明分散率: {', '.join([f'PC{i+1}: {v:.1%}' for i, v in enumerate(ev)])}").classes("text-xs text-grey-5 q-mb-sm")

    plot_slot = ui.column().classes("w-full h-[600px] flex items-center justify-center")
    
    def update_fig(color_var=None):
        plot_slot.clear()
        with plot_slot:
            fig = _create_fig(df, method, n_comp, color_var)
            ui.plotly(fig).classes("w-full h-full")
    
    # 初回描画
    update_fig(color_select_ui.value)
    
    # 色分け変更の監視 (このタブコンテキスト内でのみ有効)
    def on_color_change(e):
        # アクティブなタブかどうかは問わず、この関数が呼ばれたら更新する（ NiceGUIのイベントバインドは注意が必要だが、ここでは単純化）
        update_fig(e.value)
    
    # イベントリスナーの上書き防止（NiceGUIの仕様に合わせて管理が必要な場合はここにロジックを追加）
    color_select_ui.on_value_change(on_color_change)


def _create_fig(df: pd.DataFrame, method: str, n_components: int, color_var: Optional[str]):
    """Plotly Expressを使用して正方形プロットを作成"""
    dim_cols = [c for c in df.columns if c.startswith(f"{method}_dim")]
    
    fig_args = {
        'data_frame': df,
        'x': dim_cols[0],
        'y': dim_cols[1],
        'template': 'plotly_dark',
        'hover_data': [df.index.name or 'index'] if df.index.name else None,
        'title': f"{method}分析結果 (Color: {color_var if color_var else 'Uniform'})"
    }
    
    if n_components == 3 and len(dim_cols) >= 3:
        fig_args['z'] = dim_cols[2]
        if color_var: fig_args['color'] = color_var
        fig = px.scatter_3d(**fig_args)
        fig.update_layout(scene=dict(aspectmode='cube'))
    else:
        if color_var: fig_args['color'] = color_var
        fig = px.scatter(**fig_args)
        fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
        
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig
