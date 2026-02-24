"""
frontend_streamlit/pages/automl_page.py
AutoML実行ページ。ワンクリックでデータ→学習→結果を実行する。
"""
from __future__ import annotations

import time
import pandas as pd
import numpy as np
import streamlit as st

from backend.models.automl import AutoMLEngine, AutoMLResult
from backend.data.preprocessor import PreprocessConfig


def render() -> None:
    st.markdown("## 🤖 AutoML 実行")

    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")

    if df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
        if st.button("📂 データ読み込みへ"):
            st.session_state["page"] = "data_load"
            st.rerun()
        return

    if not target_col:
        st.warning("⚠️ 目的変数が選択されていません。データ読み込みページで設定してください。")
        return

    # ── 設定パネル ────────────────────────────────────────────
    with st.expander("⚙️ AutoML 詳細設定", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            cv_folds = st.slider("CV分割数", 2, 10, 5)
            max_models = st.slider("試すモデル数(最大)", 1, 15, 8)
        with col2:
            timeout = st.slider("タイムアウト(秒)", 30, 3600, 300)
            numeric_scaler = st.selectbox(
                "数値スケーラー",
                ["auto", "standard", "robust", "minmax", "none"],
            )
        with col3:
            task_override = st.selectbox(
                "タスク", ["auto", "regression", "classification"]
            )
            smiles_col = st.selectbox(
                "SMILES列（化合物の場合）",
                ["なし"] + df.columns.tolist(),
                index=0,
            )
            smiles_col = None if smiles_col == "なし" else smiles_col

    # 設定サマリー
    task_display = st.session_state.get("task", "auto")
    if task_override != "auto":
        task_display = task_override

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, str(df.shape[0]), "学習サンプル数"),
        (col2, str(df.shape[1] - 1), "特徴量数"),
        (col3, str(cv_folds), "CV分割数"),
        (col4, str(max_models), "試行モデル数"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{val}</div>
  <div class="metric-label">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 学習済み結果があれば表示
    existing = st.session_state.get("automl_result")
    if existing is not None:
        st.success(f"✅ 前回の結果: 最良モデル = **{existing.best_model_key}** | スコア = `{existing.best_score:.4f}`")
        if st.button("🔄 再実行", use_container_width=True):
            st.session_state["automl_result"] = None
            st.rerun()
        _show_leaderboard(existing)
        return

    # ── 実行ボタン ────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        run_clicked = st.button("🚀  AutoML を実行する", use_container_width=True)

    if not run_clicked:
        st.markdown("""
<div style="text-align:center; padding:3rem; color:#555;">
<div style="font-size:3rem;">🤖</div>
<div style="margin-top:1rem;">「AutoML を実行する」を押してワンクリック機械学習を開始</div>
</div>""", unsafe_allow_html=True)
        return

    # ── 実行 ─────────────────────────────────────────────────
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_area = st.empty()
    log_lines: list[str] = []

    def _cb(step: int, total: int, msg: str) -> None:
        progress_bar.progress(step / total)
        status_text.markdown(f"**{msg}**")
        log_lines.append(f"[{step}/{total}] {msg}")
        log_area.markdown(
            "<br>".join(f'<span style="color:#8888aa;font-size:0.85rem;">{l}</span>'
                        for l in log_lines[-5:]),
            unsafe_allow_html=True,
        )

    cfg = PreprocessConfig(
        numeric_scaler=numeric_scaler,
        exclude_smiles=True,
        exclude_constant=True,
    )
    engine = AutoMLEngine(
        task=task_override,
        cv_folds=cv_folds,
        max_models=max_models,
        timeout_seconds=timeout,
        progress_callback=_cb,
    )

    try:
        start = time.time()
        result = engine.run(df, target_col=target_col, smiles_col=smiles_col,
                            preprocess_config=cfg)
        elapsed = time.time() - start

        st.session_state["automl_result"] = result
        progress_bar.progress(1.0)
        status_text.empty()
        log_area.empty()

        st.balloons()
        st.success(
            f"✅ AutoML 完了！ ({elapsed:.1f}秒) | "
            f"最良モデル: **{result.best_model_key}** | "
            f"スコア: `{result.best_score:.4f}`"
        )

        if result.warnings:
            for w in result.warnings:
                st.warning(f"⚠️ {w}")

        _show_leaderboard(result)

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ AutoML実行エラー: {e}")
        import traceback
        with st.expander("エラー詳細"):
            st.code(traceback.format_exc())


def _show_leaderboard(result: AutoMLResult) -> None:
    """モデルリーダーボードを表示する。"""
    st.markdown("### 🏆 モデルリーダーボード")

    # スコアを正の方向に変換して表示
    scores = result.model_scores
    details = result.model_details

    df_lb = pd.DataFrame([
        {
            "ランク": i + 1,
            "モデル": k,
            "スコア（平均）": f"{v:.4f}",
            "標準偏差": f"{details[k]['std']:.4f}" if k in details else "-",
            "学習時間(s)": f"{details[k]['fit_time']:.2f}" if k in details else "-",
            "最良": "🏆" if k == result.best_model_key else "",
        }
        for i, (k, v) in enumerate(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)
        )
    ])
    st.dataframe(df_lb, use_container_width=True, hide_index=True)

    # バーチャート
    st.markdown("### 📊 スコア比較")
    import plotly.graph_objects as go

    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    keys = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    colors = ["#7b2ff7" if k == result.best_model_key else "#00d4ff" for k in keys]

    fig = go.Figure(go.Bar(
        x=vals, y=keys,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
        xaxis=dict(gridcolor="#333", title=result.scoring),
        yaxis=dict(gridcolor="#333"),
        height=max(300, len(keys) * 40),
        margin=dict(l=120, r=50, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 モデル評価へ", use_container_width=True):
            st.session_state["page"] = "evaluation"
            st.rerun()
    with col2:
        if st.button("💡 SHAP解釈へ", use_container_width=True):
            st.session_state["page"] = "interpret"
            st.rerun()
