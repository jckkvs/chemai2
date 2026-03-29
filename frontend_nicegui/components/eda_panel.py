# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/eda_panel.py

探索的データ分析（EDA）+ 次元削減パネル（NiceGUI版）。

機能:
  1. 統計サマリー（mean/std/min/max/欠損率）
  2. 数値列の分布ヒストグラム
  3. 相関行列ヒートマップ
  4. PCA / t-SNE / UMAP 次元削減散布図
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui


def render_eda_panel(state: dict) -> None:
    """EDA + 次元削減パネルをレンダリング。"""
    
    container = ui.column().classes("full-width")
    
    def _build():
        container.clear()
        with container:
            with ui.row().classes("full-width justify-between items-center q-mb-sm"):
                ui.label("").classes("col-grow")
                ui.button("🔄 最新のデータで更新", on_click=_build).props("outline color=cyan size=sm no-caps").tooltip("データタブで読み込んだ内容を反映")
                
            df = state.get("df")
            if df is None:
                with ui.card().classes("glass-card q-pa-xl full-width flex flex-center items-center text-center"):
                    ui.icon("analytics", color="grey-6", size="xl").classes("q-mb-md")
                    ui.label("データが読み込まれていません。").classes("text-h6 text-grey-5")
                    ui.label("「📂 データ設定」タブでデータを読み込んでから、上の更新ボタンを押してください。").classes("text-caption text-grey-6 q-mt-sm")
                return

            target_col = state.get("target_col", "")

            with ui.card().classes("full-width q-pa-md").style(
                "border:1px solid rgba(0,188,212,0.3);border-radius:12px;"
                "background:rgba(0,20,40,0.25);"
            ):
                with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
                    ui.icon("query_stats", color="cyan").classes("text-h5")
                    ui.label("探索的データ分析（EDA）").classes("text-h6")

                with ui.tabs().classes("full-width").props(
                    "dense no-caps active-color=cyan indicator-color=cyan scrollable"
                ) as eda_tabs:
                    ui.tab("eda_stats", label="📋 統計サマリー")
                    ui.tab("eda_dist", label="📊 分布")
                    ui.tab("eda_corr", label="🔥 相関行列")
                    ui.tab("eda_dr", label="🌀 次元削減")
                    ui.tab("eda_pairplot", label="📈 Pairplot")
                    ui.tab("eda_outliers", label="🎯 外れ値")
                    ui.tab("eda_missing", label="🧩 欠損解析")

                with ui.tab_panels(eda_tabs, value="eda_stats").classes("full-width bg-transparent"):
                    # ── 統計サマリー ──
                    with ui.tab_panel("eda_stats"):
                        _render_stats(df, target_col)

                    # ── 分布 ──
                    with ui.tab_panel("eda_dist"):
                        _render_distribution(df, target_col)

                    # ── 相関行列 ──
                    with ui.tab_panel("eda_corr"):
                        _render_correlation(df, target_col)

                    # ── 次元削減 ──
                    with ui.tab_panel("eda_dr"):
                        _render_dim_reduction(df, state)

                    # ── Pairplot ──
                    with ui.tab_panel("eda_pairplot"):
                        _render_pairplot(df, target_col)

                    # ── 外れ値 ──
                    with ui.tab_panel("eda_outliers"):
                        _render_outliers(df, target_col)

                    # ── 欠損解析 ──
                    with ui.tab_panel("eda_missing"):
                        _render_missing(df)
                    
    state["_refresh_eda_main"] = _build
    _build()


def _render_stats(df: pd.DataFrame, target_col: str) -> None:
    """統計サマリーテーブル。"""
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        ui.label("数値列がありません").classes("text-caption text-grey")
        return

    stats = num_df.describe().T
    stats["欠損率(%)"] = (num_df.isna().mean() * 100).round(1)
    stats = stats.reset_index().rename(columns={"index": "列名"})

    # 目的変数をハイライト
    display_cols = ["列名", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "欠損率(%)"]
    existing_cols = [c for c in display_cols if c in stats.columns]

    # 数値を丸める
    for c in existing_cols:
        if c not in ("列名", "count"):
            stats[c] = stats[c].apply(lambda x: round(x, 4) if isinstance(x, float) else x)

    rows_data = stats[existing_cols].to_dict("records")
    columns = [{"name": c, "label": c, "field": c, "sortable": True} for c in existing_cols]

    ui.table(columns=columns, rows=rows_data).classes("full-width").props(
        "dense flat separator=cell"
    ).style("font-size:0.8rem;")


def _render_distribution(df: pd.DataFrame, target_col: str) -> None:
    """数値列の分布ヒストグラム（Plotly）。"""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        ui.label("数値列がありません").classes("text-caption text-grey")
        return

    # 列選択
    default_col = target_col if target_col in num_cols else num_cols[0]
    col_select = ui.select(
        num_cols, value=default_col, label="列を選択",
    ).props("outlined dense").classes("q-mb-sm")

    chart_container = ui.column().classes("full-width")

    def _update_chart():
        chart_container.clear()
        col = col_select.value
        if not col or col not in df.columns:
            return
        try:
            import plotly.express as px
            series = df[col].dropna()
            fig = px.histogram(
                series, nbins=30, title=f"{col} の分布",
                template="plotly_dark",
                color_discrete_sequence=["#00d4ff"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0", size=11),
                margin=dict(l=40, r=20, t=40, b=30),
                height=350,
            )
            with chart_container:
                ui.plotly(fig).classes("full-width")
        except ImportError:
            with chart_container:
                ui.label("Plotlyが未インストールです").classes("text-caption text-grey")

    col_select.on("update:model-value", lambda: _update_chart())
    _update_chart()


def _render_correlation(df: pd.DataFrame, target_col: str) -> None:
    """相関行列ヒートマップ（Plotly）。"""
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        ui.label("数値列が2列未満です").classes("text-caption text-grey")
        return

    # 50列以上なら目的変数との相関のみ表示
    max_cols = 30
    if num_df.shape[1] > max_cols and target_col in num_df.columns:
        corr_with_target = num_df.corr()[target_col].abs().sort_values(ascending=False)
        top_cols = corr_with_target.head(max_cols).index.tolist()
        num_df = num_df[top_cols]
        ui.label(f"上位 {max_cols} 列のみ表示").classes("text-caption text-amber q-mb-xs")

    corr = num_df.corr()

    try:
        import plotly.express as px
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            template="plotly_dark",
            title="相関行列",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0", size=10),
            margin=dict(l=60, r=20, t=40, b=60),
            height=max(400, min(800, num_df.shape[1] * 25)),
        )
        ui.plotly(fig).classes("full-width")
    except ImportError:
        ui.label("Plotlyが未インストールです").classes("text-caption text-grey")


def _render_dim_reduction(df: pd.DataFrame, state: dict) -> None:
    """PCA / t-SNE / UMAP 次元削減散布図。"""
    num_df = df.select_dtypes(include="number").dropna()
    target_col = state.get("target_col", "")

    if num_df.shape[1] < 2 or num_df.shape[0] < 5:
        ui.label("数値列が2列未満またはデータが少なすぎます").classes("text-caption text-grey")
        return

    with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
        method_select = ui.select(
            ["PCA", "t-SNE"],
            value="PCA",
            label="手法",
        ).props("outlined dense").classes("col-3")

    chart_container = ui.column().classes("full-width")

    async def _run_dr():
        chart_container.clear()
        method = method_select.value

        # 特徴量準備
        feature_cols = [c for c in num_df.columns if c != target_col]
        if len(feature_cols) < 2:
            with chart_container:
                ui.label("特徴量が2列未満です").classes("text-caption text-grey")
            return

        X = num_df[feature_cols].values
        y = num_df[target_col].values if target_col in num_df.columns else None

        # スケーリング
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)

        try:
            if method == "PCA":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                X_2d = reducer.fit_transform(X_scaled)
                exp_var = reducer.explained_variance_ratio_
                axis_labels = (
                    f"PC1 ({exp_var[0]:.1%})",
                    f"PC2 ({exp_var[1]:.1%})",
                )
            else:  # t-SNE
                from sklearn.manifold import TSNE
                n_samples = min(X_scaled.shape[0], 5000)
                perp = min(30, n_samples - 1)
                reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
                X_2d = reducer.fit_transform(X_scaled[:n_samples])
                y = y[:n_samples] if y is not None else None
                axis_labels = ("t-SNE 1", "t-SNE 2")

            # Plotly 散布図
            import plotly.express as px
            plot_df = pd.DataFrame({
                axis_labels[0]: X_2d[:, 0],
                axis_labels[1]: X_2d[:, 1],
            })
            if y is not None:
                plot_df[target_col] = y[:len(X_2d)]

            fig = px.scatter(
                plot_df,
                x=axis_labels[0], y=axis_labels[1],
                color=target_col if y is not None else None,
                color_continuous_scale="Viridis",
                template="plotly_dark",
                title=f"{method} 2D散布図",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0", size=11),
                margin=dict(l=40, r=20, t=40, b=30),
                height=450,
            )
            with chart_container:
                ui.plotly(fig).classes("full-width")
                if method == "PCA":
                    cum = np.cumsum(exp_var)
                    ui.label(
                        f"累積寄与率: PC1={cum[0]:.1%}, PC2={cum[1]:.1%}"
                    ).classes("text-caption text-grey q-mt-xs")

        except ImportError as ie:
            with chart_container:
                ui.label(f"必要なライブラリが未インストールです: {ie}").classes("text-caption text-red")
        except Exception as e:
            with chart_container:
                ui.label(f"エラー: {e}").classes("text-caption text-red")

    ui.button(f"🌀 次元削減を実行", on_click=_run_dr).props(
        "color=cyan no-caps size=sm"
    ).classes("q-mb-sm")


def _render_pairplot(df: pd.DataFrame, target_col: str) -> None:
    """Pairplot（変数の散布図マトリックス）。"""
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        ui.label("数値列が2列未満です").classes("text-caption text-grey")
        return

    num_cols = num_df.columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
        default_cols = [target_col] + num_cols[:4]
    else:
        default_cols = num_cols[:5]

    ui.label("Pairplot (変数間の散布図マトリックス)").classes("text-subtitle2 q-mb-xs")
    ui.label("描画する特徴量を選択してください（最大5〜6個程度を推奨）").classes("text-caption text-grey q-mb-sm")

    selected_cols = ui.select(
        num_df.columns.tolist(), value=default_cols, label="特徴量", multiple=True
    ).props("outlined dense use-chips").classes("full-width q-mb-md")

    chart_container = ui.column().classes("full-width")

    def _draw():
        chart_container.clear()
        cols = selected_cols.value
        if not cols or len(cols) < 2:
            with chart_container:
                ui.label("2つ以上の特徴量を選択してください").classes("text-caption text-amber")
            return

        try:
            import plotly.express as px
            plot_df = num_df[cols].dropna(how="any").copy()
            if plot_df.empty:
                with chart_container:
                    ui.label("選択した列の共通有効データがありません").classes("text-caption text-amber")
                return

            if len(plot_df) > 1000:
                plot_df = plot_df.sample(1000, random_state=42)
                with chart_container:
                    ui.label("※ データサイズが大きいため、ランダムに1000件サンプリング表示しています").classes("text-caption text-amber q-mb-sm")

            color_col = target_col if target_col in cols and df[target_col].nunique() < 20 else None

            fig = px.scatter_matrix(
                plot_df,
                dimensions=cols,
                color=color_col if color_col in plot_df.columns else None,
                template="plotly_dark",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0", size=10),
                margin=dict(l=20, r=20, t=40, b=20),
                height=max(450, len(cols)*120),
            )
            fig.update_traces(diagonal_visible=False)
            with chart_container:
                ui.plotly(fig).classes("full-width")
        except Exception as e:
            with chart_container:
                ui.label(f"描画エラー: {e}").classes("text-caption text-red")

    ui.button("描画更新", on_click=_draw).props("outline color=cyan size=sm no-caps").classes("q-mb-md")
    _draw()

def _render_outliers(df: pd.DataFrame, target_col: str) -> None:
    """外れ値検出とボックスプロット表示。"""
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        ui.label("数値列がありません").classes("text-caption text-grey")
        return

    ui.label("外れ値検出 (四分位範囲 IQR 方式に基づくボックスプロット)").classes("text-subtitle2 q-mb-sm")

    num_cols = num_df.columns.tolist()
    default_col = target_col if target_col in num_cols else num_cols[0]
    col_select = ui.select(
        num_cols, value=default_col, label="列を選択",
    ).props("outlined dense").classes("q-mb-sm")

    chart_container = ui.column().classes("full-width")
    def _draw():
        chart_container.clear()
        col = col_select.value
        if not col or col not in num_cols:
            return
        series = num_df[col].dropna()
        if series.empty:
            return

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        with chart_container:
            with ui.row().classes("q-gutter-md q-mb-md items-center"):
                ui.label(f"外れ値数 (IQR): {len(outliers)}").classes("text-body2 text-bold text-amber")
                ui.label(f"全有効データ数: {len(series)}").classes("text-caption text-grey")
                ui.label(f"下限: {lower_bound:.4g} | 上限: {upper_bound:.4g}").classes("text-caption text-cyan")

            try:
                import plotly.express as px
                fig = px.box(
                    num_df, y=col,
                    template="plotly_dark",
                    points="outliers",
                    title=f"{col} のボックスプロット",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0f0", size=11),
                    margin=dict(l=40, r=20, t=40, b=30),
                    height=350,
                )
                ui.plotly(fig).classes("full-width")
            except Exception as e:
                ui.label(f"描画エラー: {e}").classes("text-caption text-red")

    col_select.on("update:model-value", lambda: _draw())
    _draw()


def _render_missing(df: pd.DataFrame) -> None:
    """欠損値パターン解析（棒グラフ・ヒートマップ）。"""
    ui.label("欠損解析").classes("text-subtitle2 q-mb-sm")
    total_missing = df.isna().sum().sum()
    if total_missing == 0:
        ui.label("データセットに欠損値はありません。").classes("text-body1 text-green text-bold")
        return

    with ui.row().classes("q-gutter-md q-mb-md"):
        ui.label(f"総欠損セル数: {total_missing}").classes("text-body2 text-bold text-amber")
        ui.label(f"欠損行を含む行数: {df.isna().any(axis=1).sum()}").classes("text-body2 text-grey")

    try:
        import plotly.express as px
        missing_counts = df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)

        fig_bar = px.bar(
            x=missing_counts.values,
            y=missing_counts.index,
            orientation='h',
            title="列ごとの欠損値数",
            labels={"x": "欠損数", "y": "列名"},
            template="plotly_dark",
            color_discrete_sequence=["#fbbf24"],
            text_auto=True
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0", size=11),
            margin=dict(l=10, r=20, t=40, b=30),
            height=max(300, len(missing_counts) * 25),
        )
        ui.plotly(fig_bar).classes("full-width")

        # 欠損マップ
        if len(df) > 1000:
            sample_df = df.sample(1000, random_state=42).sort_index()
            missing_matrix = sample_df.isna().astype(int).T
            map_title = "欠損値ヒートマップ (ランダムサンプリング1000行) (黄: 欠損)"
        else:
            missing_matrix = df.isna().astype(int).T
            map_title = "欠損値ヒートマップ (黄: 欠損)"

        fig_map = px.imshow(
            missing_matrix,
            color_continuous_scale=["rgba(0,0,0,0)", "#fbbf24"],
            title=map_title,
            template="plotly_dark",
        )
        fig_map.update_layout(
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=40, b=20),
            height=max(300, missing_matrix.shape[0] * 15),
        )
        fig_map.update_xaxes(showticklabels=False)
        ui.plotly(fig_map).classes("full-width q-mt-md")

    except Exception as e:
        ui.label(f"描画エラー: {e}").classes("text-caption text-red")
