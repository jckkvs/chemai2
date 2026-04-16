"""
EDA - 次元削減パネル（変数別色分け機能・特徴量選択機能完全実装）
"""
import pandas as pd
import numpy as np
from nicegui import ui
import plotly.graph_objects as go
import plotly.express as px
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
    データのクラスタリングや外れ値の検出に役立ちます。
    
    **新機能**:
    - **特徴量選択**: 次元削減に使用する変数を個別に選択可能
    - **動的色分け**: プロット後の結果を各変数（数値・カテゴリ）で瞬時に色分け
    - **正方形プロット**: 物理的距離を正確に保つアスペクト比
    """).classes("text-body2 text-grey-7 q-mb-md")
    
    # データが読み込まれているか確認
    if "df" not in state or state["df"] is None:
        ui.warning("先にデータを読み込んでください")
        return
    
    df = state["df"]
    column_roles = state.get("column_roles", {col: "feature" for col in df.select_dtypes(include=[np.number]).columns})
    
    # 数値列のみを抽出（目的変数を除く説明変数）
    feature_cols = [col for col, role in column_roles.items() if role == "feature"]
    numeric_cols = [col for col in feature_cols if col in df.select_dtypes(include=[np.number]).columns]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    
    if len(numeric_cols) < 2:
        ui.error("次元削減には2つ以上の数値説明変数が必要です")
        return
    
    # 設定UI
    with ui.card().classes("w-full q-mb-md"):
        ui.label("⚙️ 設定").classes("text-lg font-semibold q-mb-sm")
        
        with ui.column().classes("w-full q-gutter-y-md"):
            with ui.row().classes("w-full q-gutter-md"):
                # 手法選択
                method_select = ui.select(
                    options=["PCA", "t-SNE", "UMAP"],
                    label="次元削減手法",
                    value="PCA"
                ).classes("w-48")
                
                # 次元数
                n_components_select = ui.select(
                    options=[2, 3],
                    label="出力次元数",
                    value=2
                ).classes("w-32")
                
                # 使用する特徴量
                selected_features = ui.select(
                    options=numeric_cols,
                    label="使用する特徴量（複数選択可）",
                    multiple=True,
                    value=numeric_cols[:min(10, len(numeric_cols))]
                ).classes("flex-grow").props("use-chips")

            # t-SNE用パラメータ
            with ui.row().classes("q-gutter-sm").bind_visibility_from(method_select, 'value', lambda v: v == 't-SNE'):
                perplexity_input = ui.number(
                    label="t-SNE Perplexity",
                    value=5.0,
                    min=1.0,
                    max=50.0
                ).classes("w-40").props("dense outlined")
                
                learning_rate_input = ui.number(
                    label="学習率",
                    value=200.0,
                    min=10.0,
                    max=1000.0
                ).classes("w-40").props("dense outlined")
        
        # 実行ボタン
        ui.button(
            "📊 次元削減を実行",
            on_click=lambda: _run_dim_reduction_with_progress(
                df,
                selected_features.value,
                method_select.value,
                n_components_select.value,
                perplexity_input.value if method_select.value == 't-SNE' else None,
                learning_rate_input.value if method_select.value == 't-SNE' else None,
                state
            )
        ).props("unelevated color=primary").classes("q-mt-md")
    
    # 進捗表示領域
    progress_container = ui.column().classes("w-full q-mt-md")
    state["_dim_reduction_progress"] = progress_container
    
    # 結果表示領域（キャッシュから復元）
    if "dim_reduction_results" in state:
        _render_multiple_results_with_coloring(state["dim_reduction_results"], numeric_cols, categorical_cols, state)


def _run_dim_reduction_with_progress(df: pd.DataFrame, features: list, method: str, 
                                     n_components: int, perplexity: Optional[float] = None,
                                     learning_rate: Optional[float] = None, state: dict = None):
    """進捗表示付きで次元削減を実行"""
    
    if not features or len(features) < 2:
        ui.error("2つ以上の特徴量を選択してください")
        return
    
    progress_container = state.get("_dim_reduction_progress")
    if not progress_container:
        return
    
    progress_container.clear()
    
    # 進捗UIを初期化
    with progress_container:
        ui.label(f"🔄 {method} 実行中...").classes("text-lg font-bold q-mb-sm")
        progress_bar = ui.linear_progress(value=0, show_value=True).classes("w-full")
        status_label = ui.label("データ前処理中...").classes("text-grey q-mt-sm")
        with ui.expansion("📝 詳細ログ", icon="terminal").classes("w-full q-mt-sm border border-grey-8"):
            log_area = ui.label("").classes("text-xs text-grey-500 font-mono q-pa-sm whitespace-pre-wrap")
    
    try:
        # 特徴量データを準備（欠損値除去）
        X = df[features].dropna()
        
        if len(X) < 2:
            ui.error("有効なデータが少なすぎます")
            return
        
        # パラメータ準備
        params = {}
        if method == 't-SNE':
            if perplexity is not None:
                params['perplexity'] = perplexity
            if learning_rate is not None:
                params['learning_rate'] = learning_rate
        
        # ステータス更新
        status_label.text = f"{method} 計算開始（{len(X)}サンプル, {len(features)}特徴量）..."
        progress_bar.value = 0.1
        log_area.text = f"開始時刻: {time.strftime('%H:%M:%S')}\n"
        
        # バックグラウンドで実行
        result_container = {'result': None, 'error': None}
        
        def run_in_thread():
            try:
                # 注: set_textなどはメインスレッド外で呼ぶと警告が出る場合があるが、NiceGUIの仕様を確認
                # ここでは安全のため値の更新のみ
                result = compute_dimensionality_reduction(
                    X.values,
                    method=method.lower(),
                    n_components=n_components,
                    **params
                )
                result_container['result'] = result
            except Exception as e:
                result_container['error'] = str(e)
                result_container['traceback'] = traceback.format_exc()
        
        # スレッド起動
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        
        # タイムアウト監視
        timeout_seconds = 300 if method == 't-SNE' else 120
        start_time = time.time()
        
        while thread.is_alive():
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                status_label.set_text(f"❌ {method} が {timeout_seconds}秒でタイムアウトしました。")
                ui.error(f"{method} タイムアウト")
                return
            
            # 進捗更新（擬似的に）
            progress = min(0.9, 0.2 + (elapsed / timeout_seconds) * 0.7)
            progress_bar.set_value(progress)
            status_label.set_text(f"計算中... {elapsed:.1f}秒経過（最大{timeout_seconds}秒）")
            ui.sleep(0.5)
        
        # 結果チェック
        if result_container['error']:
            status_label.set_text("❌ エラー発生")
            log_area.text += f"\nERROR: {result_container['error']}\n{result_container.get('traceback', '')}"
            ui.error(f"{method} 実行エラー")
            return
        
        result = result_container['result']
        if result is None:
            status_label.set_text(f"❌ {method} に失敗しました")
            ui.error("計算結果が空です")
            return
        
        coords, explained_variance = result
        
        # 進捗更新
        status_label.text = "結果を保存中..."
        progress_bar.value = 0.95
        
        # 結果をデータフレームに変換
        coord_cols = [f"{method}_dim{i+1}" for i in range(n_components)]
        result_df = pd.DataFrame(coords, columns=coord_cols, index=X.index)
        
        # 元のデータと結合（色分け用に全特徴量を保持）
        viz_df = pd.concat([X, result_df], axis=1)
        
        # 結果を保存
        if "dim_reduction_results" not in state:
            state["dim_reduction_results"] = {}
        
        result_key = f"{method}_{n_components}d"
        state["dim_reduction_results"][result_key] = {
            'method': method,
            'n_components': n_components,
            'data': viz_df,
            'features': features,
            'explained_variance': explained_variance
        }
        
        # 完了
        progress_bar.value = 1.0
        status_label.text = f"✅ {method} 完了！（{time.time() - start_time:.1f}秒）"
        log_area.text += f"完了時刻: {time.strftime('%H:%M:%S')}\n総計算時間: {time.time() - start_time:.1f}秒"
        
        ui.notify(f"{method} 計算完了", color="positive")
        # 画面更新
        _render_multiple_results_with_coloring(state["dim_reduction_results"], features, [], state)
        
    except Exception as e:
        status_label.text = "❌ 予期せぬエラー"
        log_area.text += f"\nUNEXPECTED ERROR: {str(e)}\n{traceback.format_exc()}"
        ui.error(f"システムエラー: {str(e)}")


def _render_multiple_results_with_coloring(results: Dict[str, Any], numeric_cols: List[str], 
                                           categorical_cols: List[str], state: dict):
    """色分け機能付きで複数の次元削減結果を表示"""
    
    ui.separator().classes("q-my-md")
    ui.label("📈 可視化結果（変数別色分け）").classes("text-xl font-bold q-mb-md text-cyan")
    
    ui.markdown("""
    **左パネルの変数をクリック**すると、その変数の値に応じてサンプルを色分け表示します。
    - **連続変数**: 青（低）→ 赤（高）のグラデーション
    - **カテゴリ変数**: カテゴリごとに異なる色
    """).classes("text-sm text-grey-5 q-mb-md")
    
    # 結果タブ
    with ui.tabs().classes("w-full q-mb-md") as tabs:
        tab_list = []
        for key, result in results.items():
            method = result['method']
            n_comp = result['n_components']
            tab_list.append(ui.tab(key, label=f"{method} ({n_comp}D)"))
    
    with ui.tab_panels(tabs, value=tab_list[-1]).classes("w-full bg-transparent"):
        for i, (key, result) in enumerate(results.items()):
            with ui.tab_panel(tab_list[i]):
                _render_single_result_with_coloring(result, numeric_cols, categorical_cols, state)


def _render_single_result_with_coloring(result: Dict[str, Any], numeric_cols: List[str],
                                         categorical_cols: List[str], state: dict):
    """単一の次元削減結果を色分け機能付きで表示"""
    
    method = result['method']
    viz_df = result['data']
    features = result['features']
    explained_variance = result.get('explained_variance', None)
    n_components = result['n_components']
    
    # 説明分散率の表示（PCAの場合）
    if explained_variance is not None and method == 'PCA':
        with ui.row().classes("w-full q-gutter-md q-mb-sm"):
            for i, var in enumerate(explained_variance[:min(3, n_components)]):
                ui.label(f"PC{i+1} 寄与率: {var:.2%}").classes("text-xs text-grey-5 bg-grey-9 q-pa-xs rounded")
    
    # 2カラムレイアウト（左：変数選択、右：プロット）
    with ui.row().classes("w-full no-wrap"):
        
        # 左パネル：変数選択
        with ui.column().classes("w-64 q-pr-md border-r border-grey-8"):
            ui.label("🎨 色分け変数").classes("text-sm font-bold q-mb-sm text-grey-4")
            
            plot_id = f"{method}_{n_components}d"
            
            with ui.scroll_area().classes("h-96"):
                # 連続変数
                if numeric_cols:
                    with ui.expansion("📊 連続変数", icon="trending_up", value=True).classes("w-full text-xs"):
                        for col in numeric_cols:
                            ui.button(
                                text=col,
                                on_click=lambda c=col: _update_scatter_color(
                                    c, 'continuous', viz_df, method, n_components, state
                                )
                            ).props("flat align=left no-caps size=sm").classes("w-full q-py-xs text-cyan")
                
                # カテゴリ変数
                if categorical_cols:
                    with ui.expansion("📋 カテゴリ変数", icon="category", value=False).classes("w-full text-xs q-mt-sm"):
                        for col in categorical_cols:
                            ui.button(
                                text=col,
                                on_click=lambda c=col: _update_scatter_color(
                                    c, 'categorical', viz_df, method, n_components, state
                                )
                            ).props("flat align=left no-caps size=sm").classes("w-full q-py-xs text-amber")
                
                # デフォルト表示（均一色）
                ui.button(
                    text="🔘 均一色（リセット）",
                    on_click=lambda: _update_scatter_color(
                        None, 'uniform', viz_df, method, n_components, state
                    )
                ).props("flat align=left no-caps size=sm").classes("w-full q-mt-sm text-grey-5")
        
        # 右パネル：散布図
        with ui.column().classes("flex-grow"):
            # プロットコンテナ（動的更新用）
            container_key = f"_scatter_plot_container_{plot_id}"
            state[container_key] = ui.column().classes("w-full")
            
            # 初期表示（均一色）
            with state[container_key]:
                _create_scatter_plot(viz_df, method, n_components, color_var=None, color_type='uniform')
    
    # データテーブル
    with ui.expansion("📋 データ詳細 (Top 10)", icon="table_chart").classes("w-full q-mt-md"):
        ui.table.from_pandas(viz_df.head(10)).classes("w-full").props("dense flat bordered dark")


def _update_scatter_color(variable: Optional[str], color_type: str,
                          viz_df: pd.DataFrame, method: str, n_components: int,
                          state: dict):
    """散布図の色分けを更新"""
    
    plot_id = f"{method}_{n_components}d"
    container_key = f"_scatter_plot_container_{plot_id}"
    container = state.get(container_key)
    
    if container:
        container.clear()
        with container:
            _create_scatter_plot(viz_df, method, n_components, variable, color_type)
    
    # 通知
    if variable:
        ui.notify(f"🎨 {variable} で色分けしました", color="cyan", timeout=1500, position='top-right')
    else:
        ui.notify("🔘 元の色に戻しました", color="grey", timeout=1500, position='top-right')


def _create_scatter_plot(df: pd.DataFrame, method: str, n_components: int,
                         color_var: Optional[str] = None, 
                         color_type: str = 'uniform'):
    """色分け機能付き散布図を作成（正方形プロット）"""
    
    dim_cols = [col for col in df.columns if col.startswith(f"{method}_dim")]
    
    if n_components == 3 and len(dim_cols) >= 3:
        fig = _create_3d_scatter(df, dim_cols, color_var, color_type)
    else:
        fig = _create_2d_scatter(df, dim_cols, color_var, color_type)
    
    ui.plotly(fig).classes("w-full h-[600px]")


def _create_2d_scatter(df: pd.DataFrame, dim_cols: List[str],
                       color_var: Optional[str], color_type: str) -> go.Figure:
    """2D正方形散布図を作成"""
    
    x_col, y_col = dim_cols[0], dim_cols[1]
    fig = go.Figure()
    
    if color_type == 'uniform' or color_var is None:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col],
            mode='markers',
            marker=dict(size=10, color='rgba(0, 188, 212, 0.7)', line=dict(width=1, color='rgba(0, 188, 212, 1)')),
            name='Samples',
            text=[f"Index: {idx}" for idx in df.index],
            hovertemplate="%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
        ))
        title = f"{method_label(x_col)} vs {method_label(y_col)}"
    
    elif color_type == 'continuous':
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col],
            mode='markers',
            marker=dict(
                size=12, color=df[color_var], colorscale='RdBu_r', reversescale=True,
                colorbar=dict(title=color_var, thickness=15, len=0.7, titlefont=dict(color="white"), tickfont=dict(color="white")),
                line=dict(width=0.5, color='white'), showscale=True
            ),
            text=df[color_var],
            hovertemplate=f"Val: %{{text:.4f}}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>",
            name=color_var
        ))
        title = f"Colored by {color_var}"
    
    elif color_type == 'categorical':
        categories = df[color_var].unique()
        colors = px.colors.qualitative.Plotly
        for i, category in enumerate(categories):
            mask = df[color_var] == category
            category_df = df[mask]
            fig.add_trace(go.Scatter(
                x=category_df[x_col], y=category_df[y_col], mode='markers',
                marker=dict(size=10, color=colors[i % len(colors)], line=dict(width=0.5, color='white'), opacity=0.8),
                name=str(category),
                hovertemplate=f"Cat: {category}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>"
            ))
        title = f"Colored by {color_var}"
    
    fig.update_layout(
        title=dict(text=title, font=dict(color="white")),
        xaxis=dict(
            title=x_col, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", 
            tickfont=dict(color="grey"), scaleanchor="y", scaleratio=1
        ),
        yaxis=dict(title=y_col, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", tickfont=dict(color="grey")),
        showlegend=(color_type == 'categorical'),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        width=600, height=600, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def _create_3d_scatter(df: pd.DataFrame, dim_cols: List[str],
                       color_var: Optional[str], color_type: str) -> go.Figure:
    """3D正方形散布図を作成"""
    
    x_col, y_col, z_col = dim_cols[0], dim_cols[1], dim_cols[2]
    fig = go.Figure()
    
    if color_type == 'uniform' or color_var is None:
        fig.add_trace(go.Scatter3d(
            x=df[x_col], y=df[y_col], z=df[z_col], mode='markers',
            marker=dict(size=5, color='rgba(0, 188, 212, 0.7)', opacity=0.8),
            name='Samples'
        ))
        title = "3D Visualization"
    
    elif color_type == 'continuous':
        fig.add_trace(go.Scatter3d(
            x=df[x_col], y=df[y_col], z=df[z_col], mode='markers',
            marker=dict(size=6, color=df[color_var], colorscale='RdBu_r', colorbar=dict(title=color_var), opacity=0.8),
            name=color_var
        ))
        title = f"3D Colored by {color_var}"
    
    elif color_type == 'categorical':
        categories = df[color_var].unique()
        colors = px.colors.qualitative.Plotly
        for i, category in enumerate(categories):
            mask = df[color_var] == category
            category_df = df[mask]
            fig.add_trace(go.Scatter3d(
                x=category_df[x_col], y=category_df[y_col], z=category_df[z_col], mode='markers',
                marker=dict(size=5, color=colors[i % len(colors)], opacity=0.8),
                name=str(category)
            ))
        title = f"3D Colored by {color_var}"
    
    fig.update_layout(
        title=dict(text=title, font=dict(color="white")),
        scene=dict(
            xaxis=dict(title=x_col, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title=y_col, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(title=z_col, backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            aspectmode='cube'
        ),
        width=600, height=600, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def method_label(col_name: str) -> str:
    """PCA_dim1 -> PC1 などのラベル変換"""
    if "_dim" in col_name:
        parts = col_name.split("_dim")
        return f"{parts[0]}{parts[1]}"
    return col_name
