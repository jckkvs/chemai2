"""frontend_streamlit/pages/evaluation_page.py - モデル評価ページ（強化版）"""
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

    # ─── サマリーカード ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    metric_data = [
        (col1, result.best_model_key, "🏆 最良モデル"),
        (col2, result.task, "📋 タスク"),
        (col3, f"{result.best_score:.4f}", f"📈 {result.scoring}"),
        (col4, f"{result.elapsed_seconds:.1f}s", "⏱️ 学習時間"),
    ]
    for col, val, label in metric_data:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="font-size:1.3rem">'
                f'{val}</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ─── タブ ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 予測精度", "📊 全モデル比較", "📉 学習曲線", "🔍 残差/混同行列"]
    )

    # ─── Tab1: 予測精度 ───────────────────────────────────────────
    with tab1:
        if df is not None and target_col:
            X = df.drop(columns=[target_col])
            y_true = df[target_col].values
            try:
                y_pred = result.best_pipeline.predict(X)
                if result.task == "regression":
                    _show_regression_metrics(y_true, y_pred)
                else:
                    _show_classification_metrics(y_true, y_pred, result.best_pipeline, X)
            except Exception as e:
                err_msg = str(e)
                if "columns are missing" in err_msg:
                    st.warning(
                        "⚠️ コラム不一致エラーが発生しました。"
                        "旐しいバージョンのAutoMLで学習したモデルとセッションデータの不一致です。\n"
                        "「🔄 AutoMLを再実行」ボタンから再学習してください。"
                    )
                else:
                    st.error(f"予測中にエラー: {e}")
        else:
            st.info("データまたは目的変数が設定されていません。")

    # ─── Tab2: 全モデル比較 ──────────────────────────────────────
    with tab2:
        scores = result.all_scores if hasattr(result, "all_scores") else {}
        if scores:
            import plotly.express as px
            rows = []
            for key, cv_res in scores.items():
                mean_v = cv_res.get("mean_test_score")
                std_v = cv_res.get("std_test_score", 0)
                fit_t = cv_res.get("fit_time", np.array([0])).mean()
                rows.append({
                    "モデル": key,
                    "CV平均スコア": mean_v or 0,
                    "CV標準偏差": std_v or 0,
                    "学習時間(s)": round(fit_t, 2),
                    "最良": "⭐" if key == result.best_model_key else "",
                })
            cmp_df = pd.DataFrame(rows).sort_values("CV平均スコア", ascending=False)
            st.dataframe(
                cmp_df.style.background_gradient(
                    subset=["CV平均スコア"], cmap="RdYlGn"
                ),
                use_container_width=True,
            )

            # バーチャート
            fig = px.bar(
                cmp_df, x="モデル", y="CV平均スコア",
                error_y="CV標準偏差",
                color="CV平均スコア",
                color_continuous_scale="RdYlGn",
                template="plotly_dark",
                title=f"全モデルCV比較 ({result.scoring})",
                text=cmp_df["CV平均スコア"].map("{:.3f}".format),
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0"),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("全モデルのCVスコアが取得できませんでした。")
            # 最良モデルのみ表示
            st.markdown(f"**最良モデル**: `{result.best_model_key}` — スコア: `{result.best_score:.4f}`")

    # ─── Tab3: 学習曲線 ──────────────────────────────────────────
    with tab3:
        if df is not None and target_col:
            st.markdown("### 📉 学習曲線")
            st.caption("サンプル数を増やしたときの訓練・検証スコアの推移")
            cv_folds = st.slider("CV分割数", 2, 10, 5, key="lc_cv")
            scoring_lc = result.scoring if result.task == "regression" else "accuracy"

            if st.button("▶️ 学習曲線を計算", key="btn_lc"):
                with st.spinner("計算中..."):
                    try:
                        from backend.data.benchmark import compute_learning_curve
                        # Pipeline内にSMILES Transformerが有るので元データ（SMILES列含む）をそのまま渡す
                        X = df.drop(columns=[target_col])
                        y = df[target_col]
                        lc = compute_learning_curve(
                            result.best_pipeline, X, y,
                            scoring=scoring_lc, cv=cv_folds, n_points=8,
                        )
                        st.session_state["_lc_data"] = lc
                    except Exception as e:
                        st.error(f"❌ 学習曲線の計算エラー: {e}")

            lc = st.session_state.get("_lc_data")
            if lc is not None:
                import plotly.graph_objects as go
                ts = lc["train_sizes"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts, y=lc["train_scores_mean"],
                    mode="lines+markers", name="訓練スコア",
                    line=dict(color="#00d4ff"),
                    error_y=dict(array=lc["train_scores_std"],
                                 color="rgba(0,212,255,0.3)", thickness=1),
                ))
                fig.add_trace(go.Scatter(
                    x=ts, y=lc["val_scores_mean"],
                    mode="lines+markers", name="検証スコア",
                    line=dict(color="#ff6b9d"),
                    error_y=dict(array=lc["val_scores_std"],
                                 color="rgba(255,107,157,0.3)", thickness=1),
                ))
                fig.update_layout(
                    title=f"学習曲線 ({scoring_lc})",
                    xaxis_title="サンプル数", yaxis_title=scoring_lc,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0f0"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("データが読み込まれていません。")

    # ─── Tab4: 残差/混同行列 ─────────────────────────────────────
    with tab4:
        if df is not None and target_col:
            X = df.drop(columns=[target_col])
            y_true = df[target_col].values
            try:
                y_pred = result.best_pipeline.predict(X)
                if result.task == "regression":
                    _show_residuals(y_true, y_pred)
                else:
                    _show_confusion_matrix(y_true, y_pred)
            except Exception as e:
                err_msg = str(e)
                if "columns are missing" in err_msg:
                    st.warning(
                        "⚠️ コラム不一致エラー。「🔄 AutoML再実行」で再学習してください。"
                    )
                else:
                    st.error(f"❌ エラー: {e}")
        else:
            st.info("データが読み込まれていません。")


# ============================================================
# 内部ヘルパー
# ============================================================

def _show_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import plotly.graph_objects as go

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)

    cols = st.columns(4)
    for col, val, label in [
        (cols[0], f"{rmse:.4f}", "RMSE"),
        (cols[1], f"{mae:.4f}", "MAE"),
        (cols[2], f"{r2:.4f}", "R²"),
        (cols[3], f"{mape:.2f}%", "MAPE"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color="#00d4ff", opacity=0.6, size=6), name="予測"
    ))
    rng = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    fig.add_trace(go.Scatter(
        x=rng, y=rng, mode="lines",
        line=dict(color="#fbbf24", dash="dash"), name="完全一致"
    ))
    fig.update_layout(
        title="実測 vs 予測", xaxis_title="実測値", yaxis_title="予測値",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"), template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pipeline: Any = None,
    X: Any = None,
) -> None:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    cols = st.columns(4)
    for col, val, label in [
        (cols[0], f"{acc:.4f}", "Accuracy"),
        (cols[1], f"{f1:.4f}", "F1 (weighted)"),
        (cols[2], f"{prec:.4f}", "Precision"),
        (cols[3], f"{rec:.4f}", "Recall"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )
    st.markdown("---")
    st.text(classification_report(y_true, y_pred, zero_division=0))


def _show_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """残差プロット。"""
    import plotly.express as px
    residuals = y_pred - y_true
    res_df = pd.DataFrame({"予測値": y_pred, "残差": residuals})

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(
            res_df, x="予測値", y="残差",
            title="残差プロット (予測値 vs 残差)",
            template="plotly_dark",
            color_discrete_sequence=["#00d4ff"],
            opacity=0.6,
        )
        fig1.add_hline(y=0, line_dash="dash", line_color="#fbbf24")
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0"),
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(
            res_df, x="残差", nbins=30,
            title="残差の分布",
            template="plotly_dark",
            color_discrete_sequence=["#7b2ff7"],
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0"),
        )
        st.plotly_chart(fig2, use_container_width=True)


def _show_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """混同行列の可視化。"""
    from sklearn.metrics import confusion_matrix
    import plotly.figure_factory as ff

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    labels_str = [str(l) for l in labels]

    fig = ff.create_annotated_heatmap(
        z=cm[::-1],
        x=labels_str,
        y=labels_str[::-1],
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="混同行列",
        xaxis_title="予測クラス",
        yaxis_title="実測クラス",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


# type alias for type checking only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
