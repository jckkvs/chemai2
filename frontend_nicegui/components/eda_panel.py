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
    df = state.get("df")
    if df is None:
        ui.label("データが読み込まれていません。").classes("text-caption text-grey")
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
            "dense no-caps active-color=cyan indicator-color=cyan"
        ) as eda_tabs:
            ui.tab("eda_stats", label="📋 統計サマリー")
            ui.tab("eda_dist", label="📊 分布")
            ui.tab("eda_corr", label="🔥 相関行列")
            ui.tab("eda_dr", label="🌀 次元削減")

        with ui.tab_panels(eda_tabs, value="eda_stats").classes("full-width"):

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
