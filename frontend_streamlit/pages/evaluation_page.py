"""frontend_streamlit/pages/evaluation_page.py - モデル評価ページ（スタブ）"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st


def render() -> None:
    st.markdown("## 📊 モデル評価")
    result = st.session_state.get("automl_result")
    if result is None:
        st.warning("⚠️ まずAutoMLを実行してください。")
        if st.button("🤖 AutoMLへ"):
            st.session_state["page"] = "automl"
            st.rerun()
        return

    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")

    st.markdown(f"""
<div class="card">
<h4>🏆 最良モデル: {result.best_model_key}</h4>
<ul>
<li>タスク: {result.task}</li>
<li>スコア ({result.scoring}): <code>{result.best_score:.4f}</code></li>
<li>学習時間: {result.elapsed_seconds:.1f}秒</li>
</ul>
</div>""", unsafe_allow_html=True)

    # 予測 vs 実測
    if df is not None and target_col:
        X = df.drop(columns=[target_col])
        y_true = df[target_col].values
        try:
            y_pred = result.best_pipeline.predict(X)
            if result.task == "regression":
                _show_regression_metrics(y_true, y_pred)
            else:
                _show_classification_metrics(y_true, y_pred)
        except Exception as e:
            st.error(f"予測中にエラー: {e}")


def _show_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import plotly.graph_objects as go

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    col1, col2, col3 = st.columns(3)
    for col, val, label in [(col1, f"{rmse:.4f}", "RMSE"), (col2, f"{mae:.4f}", "MAE"), (col3, f"{r2:.4f}", "R²")]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(color="#00d4ff", opacity=0.6), name="予測"))
    rng = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    fig.add_trace(go.Scatter(x=rng, y=rng, mode="lines",
                             line=dict(color="#fbbf24", dash="dash"), name="完全一致"))
    fig.update_layout(
        title="実測 vs 予測", xaxis_title="実測値", yaxis_title="予測値",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{acc:.4f}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{f1:.4f}</div><div class="metric-label">F1 (weighted)</div></div>', unsafe_allow_html=True)
    st.text(classification_report(y_true, y_pred))
