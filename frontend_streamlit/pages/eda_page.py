"""
frontend_streamlit/pages/eda_page.py
EDA（探索的データ分析）ページ。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def render() -> None:
    st.markdown("## 🔍 データ探索 (EDA)")

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
        if st.button("📂 データ読み込みへ"):
            st.session_state["page"] = "data_load"
            st.rerun()
        return

    target_col = st.session_state.get("target_col")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 分布", "🔗 相関", "📈 散布図", "🎯 目的変数分析"])

    with tab1:
        _plot_distributions(df, numeric_cols)

    with tab2:
        _plot_correlation(df, numeric_cols)

    with tab3:
        _plot_scatter(df, numeric_cols, target_col)

    with tab4:
        if target_col:
            _plot_target(df, target_col)
        else:
            st.info("データ読み込みページで目的変数を選択してください。")


def _plot_distributions(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """数値列のヒストグラム一覧を表示する。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not numeric_cols:
        st.info("数値列がありません。")
        return

    cols_disp = st.multiselect("表示する列", numeric_cols, default=numeric_cols[:min(6, len(numeric_cols))])
    if not cols_disp:
        return

    ncols = min(3, len(cols_disp))
    nrows = (len(cols_disp) + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=cols_disp,
                        vertical_spacing=0.12)

    colors = ["#00d4ff", "#7b2ff7", "#ff6b9d", "#4ade80", "#fbbf24", "#f97316"]
    for i, col in enumerate(cols_disp):
        row, colnum = divmod(i, ncols)
        data = df[col].dropna()
        fig.add_trace(
            go.Histogram(x=data, name=col, marker_color=colors[i % len(colors)],
                         opacity=0.7, nbinsx=30),
            row=row + 1, col=colnum + 1,
        )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
        showlegend=False,
        height=max(300, nrows * 250),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_correlation(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """相関行列のヒートマップを表示する。"""
    import plotly.figure_factory as ff
    import plotly.graph_objects as go

    if len(numeric_cols) < 2:
        st.info("数値列が2列以上必要です。")
        return

    show_cols = numeric_cols[:20]  # 最大20列
    corr = df[show_cols].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
        height=max(400, len(show_cols) * 35),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_scatter(df: pd.DataFrame, numeric_cols: list[str], target_col: str | None) -> None:
    """散布図を表示する。"""
    import plotly.express as px

    if len(numeric_cols) < 2:
        st.info("数値列が2列以上必要です。")
        return

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X軸", numeric_cols, index=0, key="scatter_x")
    with col2:
        default_y = 1 if len(numeric_cols) > 1 else 0
        y_col = st.selectbox("Y軸", numeric_cols, index=default_y, key="scatter_y")

    color_col = target_col if target_col and target_col in df.columns else None

    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        opacity=0.7,
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_target(df: pd.DataFrame, target_col: str) -> None:
    """目的変数の分布・外れ値等を分析して表示する。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    y = df[target_col].dropna()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**{target_col} の基本統計**")
        stats = y.describe()
        for stat_name, val in stats.items():
            st.markdown(f"- **{stat_name}**: `{val:.4g}`")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["ヒストグラム", "ボックスプロット"])
        fig.add_trace(go.Histogram(x=y, marker_color="#00d4ff", opacity=0.7, name="分布"), row=1, col=1)
        fig.add_trace(go.Box(y=y, marker_color="#7b2ff7", name="箱ひげ"), row=1, col=2)
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0"),
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
