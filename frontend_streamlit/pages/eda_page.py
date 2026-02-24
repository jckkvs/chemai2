"""
frontend_streamlit/pages/eda_page.py
EDA（探索的データ分析）ページ。バックエンドのeda.pyを統合した強化版。
"""
from __future__ import annotations

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

    # ─── データ全体サマリーカード ───────────────────────────────────────
    try:
        from backend.data.eda import summarize_dataframe, detect_outliers
        summary = summarize_dataframe(df)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("行数", f"{summary['n_rows']:,}")
        col2.metric("列数", f"{summary['n_cols']}")
        col3.metric("数値列", f"{summary['n_numeric']}")
        col4.metric("欠損率", f"{summary['total_null_rate']:.1%}")
        col5.metric("重複行", f"{summary['n_duplicates']:,}")
    except Exception:
        pass

    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 分布", "🔗 相関", "📈 散布図", "🎯 目的変数", "⚠️ 外れ値"]
    )

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

    with tab5:
        _show_outliers(df, numeric_cols)


# ─── 分布タブ ──────────────────────────────────────────────────────────

def _plot_distributions(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """数値列のヒストグラム一覧を表示する。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not numeric_cols:
        st.info("数値列がありません。")
        return

    cols_disp = st.multiselect(
        "表示する列", numeric_cols,
        default=numeric_cols[:min(6, len(numeric_cols))]
    )
    if not cols_disp:
        return

    show_stats = st.checkbox("📋 統計値を表示", value=True)

    if show_stats:
        try:
            from backend.data.eda import compute_column_stats
            stats = compute_column_stats(df[cols_disp])
            stat_data = {
                "列": [s.name for s in stats],
                "平均": [f"{s.mean:.4g}" if s.mean is not None else "—" for s in stats],
                "標準偏差": [f"{s.std:.4g}" if s.std is not None else "—" for s in stats],
                "歪度": [f"{s.skewness:.3f}" if s.skewness is not None else "—" for s in stats],
                "欠損率": [f"{s.null_rate:.1%}" for s in stats],
            }
            st.dataframe(pd.DataFrame(stat_data), use_container_width=True)
        except Exception:
            pass

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
                         opacity=0.75, nbinsx=30),
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


# ─── 相関タブ ──────────────────────────────────────────────────────────

def _plot_correlation(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """相関行列のヒートマップを表示する。"""
    import plotly.graph_objects as go

    if len(numeric_cols) < 2:
        st.info("数値列が2列以上必要です。")
        return

    method = st.selectbox("相関係数の種類", ["pearson", "spearman", "kendall"], key="corr_method")

    try:
        from backend.data.eda import compute_correlation
        show_cols = numeric_cols[:20]
        corr = compute_correlation(df[show_cols], method=method)
    except Exception:
        show_cols = numeric_cols[:20]
        corr = df[show_cols].corr(method=method)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 9},
        zmin=-1, zmax=1,
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
        height=max(400, len(show_cols) * 35),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 目的変数との相関ランキング
    target_col = st.session_state.get("target_col")
    if target_col and target_col in corr.columns:
        st.markdown("#### 🏆 目的変数との相関ランキング")
        target_corr = corr[target_col].drop(index=target_col, errors="ignore").abs().sort_values(ascending=False)
        ranking_df = target_corr.reset_index()
        ranking_df.columns = ["特徴量", f"|相関| vs {target_col}"]
        st.dataframe(ranking_df.round(4), use_container_width=True)


# ─── 散布図タブ ────────────────────────────────────────────────────────

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
        trendline="ols",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── 目的変数タブ ──────────────────────────────────────────────────────

def _plot_target(df: pd.DataFrame, target_col: str) -> None:
    """目的変数の分析結果を表示する。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    try:
        from backend.data.eda import analyze_target
        analysis = analyze_target(df, target_col)
        task = analysis.get("task", "regression")
    except Exception:
        analysis = {}
        task = "regression"

    y = df[target_col].dropna()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**{target_col} の統計サマリー**")
        if task == "regression":
            for key in ["mean", "std", "min", "p25", "p50", "p75", "max", "skewness"]:
                val = analysis.get(key)
                if val is not None:
                    st.markdown(f"- **{key}**: `{val:.4g}`")
        else:
            class_counts = analysis.get("class_counts", {})
            class_balance = analysis.get("class_balance", {})
            for cls, count in class_counts.items():
                pct = class_balance.get(cls, 0)
                st.markdown(f"- **クラス {cls}**: {count}件 ({pct:.1%})")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if task == "regression":
            fig = make_subplots(rows=1, cols=2, subplot_titles=["ヒストグラム", "ボックスプロット"])
            fig.add_trace(go.Histogram(x=y, marker_color="#00d4ff", opacity=0.7, name="分布"), row=1, col=1)
            fig.add_trace(go.Box(y=y, marker_color="#7b2ff7", name="箱ひげ"), row=1, col=2)
        else:
            counts = y.value_counts()
            fig = go.Figure(go.Bar(
                x=[str(c) for c in counts.index],
                y=counts.values,
                marker_color=["#00d4ff", "#7b2ff7", "#ff6b9d", "#4ade80"][:len(counts)],
            ))

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0"),
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── 外れ値タブ ────────────────────────────────────────────────────────

def _show_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """外れ値検出結果を表示する。"""
    if not numeric_cols:
        st.info("数値列がありません。")
        return

    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox("検出手法", ["iqr", "zscore", "modified_zscore"], key="outlier_method")
    with col2:
        k = st.slider("IQR係数 / Z閾値", 1.0, 5.0, 1.5 if method == "iqr" else 3.0, 0.1)

    try:
        from backend.data.eda import detect_outliers
        results = detect_outliers(df, method=method, k=k, z_threshold=k, cols=numeric_cols[:20])

        rows = []
        for r in results:
            rows.append({
                "列": r.col,
                "外れ値数": r.n_outliers,
                "外れ値率": f"{r.outlier_rate:.1%}",
                "下限": f"{r.lower_bound:.4g}" if r.lower_bound is not None else "—",
                "上限": f"{r.upper_bound:.4g}" if r.upper_bound is not None else "—",
            })

        if rows:
            import pandas as pd
            result_df = pd.DataFrame(rows)
            # 外れ値数でカラーハイライト
            st.dataframe(result_df, use_container_width=True)

            total = sum(r.n_outliers for r in results)
            if total > 0:
                st.warning(f"⚠️ 合計 **{total}件** の外れ値が検出されました。")
            else:
                st.success("✅ 外れ値は検出されませんでした。")
    except Exception as e:
        st.error(f"外れ値検出エラー: {e}")
