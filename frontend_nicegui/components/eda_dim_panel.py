"""
EDA - 次元削減パネル
"""
import pandas as pd
import numpy as np
from nicegui import ui
import plotly.graph_objects as go
from typing import Dict, Any, Optional
from backend.data.eda_core import compute_dimensionality_reduction


def dim_reduction_panel(state: dict):
    """次元削減パネル（t-SNE, PCA, UMAP）"""
    
    ui.label("📊 次元削減・可視化").classes("text-2xl font-bold q-mb-md")
    
    ui.markdown("""
    高次元データを2次元または3次元に圧縮して可視化します。
    データのクラスタリングや外れ値の検出に役立ちます。
    """).classes("text-body2 text-grey-7 q-mb-md")
    
    # データが読み込まれているか確認
    if "df" not in state or state["df"] is None:
        ui.warning("先にデータを読み込んでください")
        return
    
    df = state["df"]
    
    # 数値列のみを抽出
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        ui.error("次元削減には2つ以上の数値列が必要です")
        return
    
    # 設定UI
    with ui.card().classes("w-full q-mb-md"):
        ui.label("⚙️ 設定").classes("text-lg font-semibold q-mb-sm")
        
        with ui.row().classes("w-full q-gutter-md"):
            # 手法選択
            method = ui.select(
                options=["PCA", "t-SNE", "UMAP"],
                label="次元削減手法",
                value="PCA"
            ).classes("w-48")
            
            # 次元数
            n_components = ui.select(
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
            ).classes("flex-grow")
            
            # t-SNE用パラメータ
            with ui.column().classes("q-gutter-sm"):
                perplexity = ui.number(
                    label="t-SNE Perplexity",
                    value=5.0,
                    min=1.0,
                    max=50.0
                ).classes("w-48").bind_visibility_from(method, 'value', lambda v: v == 't-SNE')
                
                learning_rate = ui.number(
                    label="学習率",
                    value=200.0,
                    min=10.0,
                    max=1000.0
                ).classes("w-48").bind_visibility_from(method, 'value', lambda v: v == 't-SNE')
        
        # 実行ボタン
        ui.button(" 次元削減を実行", on_click=lambda: _run_dim_reduction(
            df,
            selected_features.value,
            method.value,
            n_components.value,
            perplexity.value if method.value == 't-SNE' else None,
            learning_rate.value if method.value == 't-SNE' else None,
            state
        )).props("unelevated color=primary").classes("q-mt-md")
    
    # 結果表示領域（キャッシュから復元）
    if "dim_reduction_results" in state:
        _render_multiple_results(state["dim_reduction_results"])


def _run_dim_reduction(df: pd.DataFrame, features: list, method: str, 
                       n_components: int, perplexity: Optional[float] = None,
                       learning_rate: Optional[float] = None, state: dict = None):
    """次元削減を実行"""
    
    if not features or len(features) < 2:
        ui.error("2つ以上の特徴量を選択してください")
        return
    
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
    
    with ui.spinner(size='3em').classes('q-ma-md'):
        ui.label(f"{method} 計算中...").classes('text-grey')
    
    try:
        # 次元削減実行
        result = compute_dimensionality_reduction(
            X.values,
            method=method.lower(),
            n_components=n_components,
            **params
        )
        
        if result is None:
            ui.error("次元削減に失敗しました")
            return
        
        coords, explained_variance = result
        
        # 結果をデータフレームに変換
        coord_cols = [f"{method}_dim{i+1}" for i in range(n_components)]
        result_df = pd.DataFrame(coords, columns=coord_cols, index=X.index)
        
        # 元のデータと結合
        viz_df = pd.concat([X.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
        
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
        
        ui.notify(f"{method} を成功させました", color="positive")
        
        # 結果を表示
        _render_multiple_results(state["dim_reduction_results"])
        
    except Exception as e:
        ui.error(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()


def _render_multiple_results(results: Dict[str, Any]):
    """複数の次元削減結果を表示"""
    
    ui.separator().classes("q-my-md")
    ui.label("📈 可視化結果").classes("text-xl font-bold q-mb-md")
    
    # 結果タブ
    with ui.tabs().classes("w-full") as tabs:
        tab_list = []
        for key, result in results.items():
            method = result['method']
            n_comp = result['n_components']
            tab_list.append(ui.tab(f"{method} ({n_comp}D)"))
    
    with ui.tab_panels(tabs, value=tab_list[0]).classes("w-full"):
        for i, (key, result) in enumerate(results.items()):
            with ui.tab_panel(tab_list[i]):
                _render_single_result(result)


def _render_single_result(result: Dict[str, Any]):
    """単一の次元削減結果を表示"""
    
    method = result['method']
    viz_df = result['data']
    features = result['features']
    explained_variance = result.get('explained_variance', None)
    
    # 説明分散率の表示（PCAの場合）
    if explained_variance is not None and method == 'PCA':
        with ui.row().classes("w-full q-gutter-md"):
            for i, var in enumerate(explained_variance[:2]):
                ui.label(f"第{i+1}主成分: {var:.2%}").classes("text-sm text-grey-7")
    
    # 3Dかどうか判定
    n_components = len([col for col in viz_df.columns if col.startswith(f"{method}_dim")])
    is_3d = n_components == 3
    
    # Plotlyで可視化
    fig = _create_scatter_plot(viz_df, method, is_3d)
    
    # NiceGUIで表示
    ui.plotly(fig).classes("w-full")
    
    # データテーブル
    with ui.expansion("📋 データ詳細", icon="table").classes("w-full q-mt-md"):
        ui.table.from_pandas(viz_df.head(10)).classes("w-full").props("dense")


def _create_scatter_plot(df: pd.DataFrame, method: str, is_3d: bool = False) -> go.Figure:
    """散布図を作成"""
    
    dim_cols = [col for col in df.columns if col.startswith(f"{method}_dim")]
    
    if is_3d and len(dim_cols) >= 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=df[dim_cols[0]],
            y=df[dim_cols[1]],
            z=df[dim_cols[2]],
            mode='markers',
            marker=dict(
                size=5,
                color='blue',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title=dim_cols[0],
                yaxis_title=dim_cols[1],
                zaxis_title=dim_cols[2]
            ),
            title=f"{method} 3D Visualization",
            margin=dict(l=0, r=0, b=0, t=30)
        )
    else:
        fig = go.Figure(data=[go.Scatter(
            x=df[dim_cols[0]],
            y=df[dim_cols[1]],
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6,
                line=dict(width=1, color='darkblue')
            )
        )])
        
        fig.update_layout(
            xaxis_title=dim_cols[0],
            yaxis_title=dim_cols[1],
            title=f"{method} 2D Visualization",
            showlegend=False,
            hovermode='closest'
        )
    
    return fig
