"""
frontend_streamlit/pages/automl_page.py

AutoML 完全パイプライン実行ページ。
EDA → 前処理 → モデル学習 → 評価 → 次元削減 → SHAP解析まで
ワンクリックで実行する。
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
import streamlit as st

from backend.models.automl import AutoMLEngine, AutoMLResult
from backend.data.preprocessor import PreprocessConfig
from backend.data.type_detector import TypeDetector
from backend.data.eda import summarize_dataframe, compute_column_stats, detect_outliers
from backend.data.dim_reduction import run_pca
from backend.data.benchmark import evaluate_regression, evaluate_classification


# ─────────────────────────────────────────────────────────────
# データクラス: フルパイプライン結果
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """フルパイプライン実行結果を保持するデータクラス。"""
    eda_summary: dict[str, Any] = field(default_factory=dict)
    col_stats: list = field(default_factory=list)
    outlier_results: list = field(default_factory=list)
    automl_result: AutoMLResult | None = None
    model_score: Any | None = None          # ModelScore (回帰/分類)
    pca_df: pd.DataFrame | None = None
    pca_evr: np.ndarray | None = None
    shap_importances: dict[str, float] = field(default_factory=dict)
    task: str = "regression"
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# メイン render 関数
# ─────────────────────────────────────────────────────────────

def render(run_config: dict | None = None) -> None:
    st.markdown("## 🤖 解析結果")

    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")

    if df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
        if st.button("🏠 ホームへ", key="goto_home_a"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    if not target_col:
        st.warning("⚠️ 目的変数が選択されていません。ホームで設定してください。")
        if st.button("🏠 ホームへ", key="goto_home_b"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    # ── ホームから渡された設定で即実行 ──────────────────────
    if run_config is not None:
        _run_full_pipeline(
            df          = df,
            target_col  = run_config["target_col"],
            smiles_col  = run_config.get("smiles_col"),
            numeric_scaler = run_config.get("scaler", "auto"),
            task_override  = run_config.get("task", "auto"),
            cv_folds    = run_config.get("cv_folds", 5),
            models      = run_config.get("models", []),
            timeout     = run_config.get("timeout", 300),
            do_eda      = run_config.get("do_eda", True),
            do_prep     = run_config.get("do_prep", True),
            do_ml       = True,
            do_eval     = run_config.get("do_eval", True),
            do_pca      = run_config.get("do_pca", True),
            do_shap     = run_config.get("do_shap", True),
        )
        return

    # ── データ概要バー ────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    metric_items = [
        (c1, str(df.shape[0]), "サンプル数"),
        (c2, str(df.shape[1] - 1), "特徴量数"),
        (c3, target_col, "目的変数"),
        (c4, str(df.select_dtypes(include="number").shape[1]), "数値列数"),
    ]
    for col, val, lbl in metric_items:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="font-size:1.2rem;">'
                f'{val}</div><div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Pipeline設定パネル（折り畳み） ────────────────────────
    with st.expander("⚙️ パイプライン詳細設定", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**全般設定**")
            timeout    = st.slider("タイムアウト(秒)", 30, 3600, 300, key="pl_to")
            numeric_scaler  = st.selectbox("数値スケーラー",
                ["auto", "standard", "robust", "minmax", "none"], key="pl_scaler")
        with col_b:
            st.markdown("**タスク & SMILES**")
            task_override   = st.selectbox("タスク",
                ["auto", "regression", "classification"], key="pl_task")
            
            smiles_options = ["なし"] + df.columns.tolist()
            smiles_default_idx = 0
            for i, col in enumerate(df.columns):
                if col.lower() == "smiles":
                    smiles_default_idx = i + 1
                    break
                    
            smiles_col_raw  = st.selectbox("SMILES列",
                smiles_options, index=smiles_default_idx, key="pl_smiles")
            smiles_col = None if smiles_col_raw == "なし" else smiles_col_raw

        st.markdown("**実行フェーズ選択**")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            do_pca   = st.checkbox("次元削減(PCA)を表示", value=True, key="pl_pca")
        with c_p2:
            do_shap  = st.checkbox("SHAP解析(特徴量重要度)を表示", value=True, key="pl_shap")

        st.markdown("**高度なパラメータ設定 (JSON)**")
        extra_params_raw = st.text_area(
            "高度な引数をJSON形式で直接指定（例: {\"cv\": {\"shuffle\": true}, \"pca\": {\"whiten\": true}}）",
            value="{}",
            help="sklearnの各コンポーネントに渡す追加引数を指定できます。キー: 'cv', 'pca', 'tsne', 'umap', 'model'",
            key="pl_extra_json"
        )
        import json
        try:
            extra_params = json.loads(extra_params_raw)
        except json.JSONDecodeError:
            st.error("❌ 高度なパラメータのJSON形式が正しくありません。")
            extra_params = {}

    # ── 既存結果の表示 ─────────────────────────────────────
    existing_pl = st.session_state.get("pipeline_result")
    existing_ml = st.session_state.get("automl_result")

    # タブ表示 (結果がある場合)
    if existing_pl is not None:
        _show_pipeline_result(existing_pl)
        st.markdown("---")
        col_re1, col_re2 = st.columns(2)
        with col_re1:
            if st.button("🔄 パイプラインを再実行", use_container_width=True, key="rerun_pl"):
                st.session_state["pipeline_result"] = None
                st.session_state["automl_result"] = None
                st.rerun()
        with col_re2:
            if existing_ml and st.button("🤖 AutoMLのみ再実行", use_container_width=True, key="rerun_ml"):
                st.session_state["automl_result"] = None
                st.rerun()
        return

    elif existing_ml is not None:
        # 旧形式 (AutoMLのみ) 結果表示
        st.success(f"✅ 前回の結果: 最良モデル = **{existing_ml.best_model_key}** | "
                   f"スコア = `{existing_ml.best_score:.4f}`")
        _show_leaderboard(existing_ml)
        if st.button("🔄 再実行", use_container_width=True, key="rerun_old"):
            st.session_state["automl_result"] = None
            st.rerun()
        return

    # ── 実行ボタン ───────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        run_clicked = st.button(
            "🚀 フルパイプライン実行（EDA → AutoML → SHAP）",
            use_container_width=True,
            key="run_full",
            type="primary",
        )

    if not run_clicked:
        st.markdown("""
<div style="text-align:center; padding:3rem; color:#555;">
  <div style="font-size:3rem;">🔬</div>
  <div style="margin-top:1rem; color:#8888aa;">
    「フルパイプライン実行」を押すと、EDA → 前処理 → AutoML → 評価 → 次元削減 → SHAP を自動で実行します
  </div>
  <div style="margin-top:0.5rem; font-size:0.85rem; color:#666;">
    各フェーズは上の⚙️設定で個別にON/OFFできます
  </div>
</div>""", unsafe_allow_html=True)
        return

    # ── フルパイプライン実行 ─────────────────────────────────
    _run_full_pipeline(
        df=df,
        target_col=target_col,
        smiles_col=smiles_col,
        numeric_scaler=numeric_scaler,
        task_override=task_override,
        cv_folds=5,      # 固定または自動
        models=[],       # 空リストを渡すとバックエンド側で全モデルを自動選択
        timeout=timeout,
        do_eda=True,     # デフォルトON
        do_prep=True,
        do_ml=True,
        do_eval=True,
        do_pca=do_pca,
        do_shap=do_shap,
        extra_params=extra_params,
    )


# ─────────────────────────────────────────────────────────────
# フルパイプライン実行
# ─────────────────────────────────────────────────────────────

def _run_full_pipeline(
    df: pd.DataFrame,
    target_col: str,
    *,
    smiles_col: str | None,
    numeric_scaler: str,
    task_override: str,
    cv_folds: int,
    models: list[str],
    timeout: int,
    do_eda: bool,
    do_prep: bool,
    do_ml: bool,
    do_eval: bool,
    do_pca: bool,
    do_shap: bool,
    extra_params: dict[str, Any] | None = None,
) -> None:
    """フルパイプラインを順番に実行してsession_stateに保存する。"""
    extra_params = extra_params or {}
    TOTAL_PHASES = sum([do_eda, do_prep, do_ml, do_eval, do_pca, do_shap])
    result = PipelineResult()
    start_time = time.time()

    progress_bar = st.progress(0.0)
    phase_status = st.empty()
    log_area     = st.empty()
    log_lines: list[str] = []
    phase_done   = 0

    def _log(msg: str) -> None:
        log_lines.append(msg)
        log_area.markdown(
            "<br>".join(
                f'<span style="color:#8888aa;font-size:0.82rem;">{l}</span>'
                for l in log_lines[-6:]
            ),
            unsafe_allow_html=True,
        )

    def _advance(phase_name: str) -> None:
        nonlocal phase_done
        phase_done += 1
        progress_bar.progress(phase_done / max(TOTAL_PHASES, 1))
        phase_status.markdown(f"**▶ {phase_name}**")
        _log(f"✅ {phase_name} 完了")

    try:
        # ── Phase 1: EDA ─────────────────────────────────
        if do_eda:
            phase_status.markdown("**Phase 1/6 — EDA 実行中…**")
            result.eda_summary  = summarize_dataframe(df)
            result.col_stats    = compute_column_stats(df)
            result.outlier_results = detect_outliers(df, method="iqr")
            _advance("Phase 1: EDA")

        # ── Phase 2: 前処理確認 ───────────────────────────
        if do_prep:
            phase_status.markdown("**Phase 2/6 — 前処理パイプライン構築中…**")
            from backend.data.type_detector import TypeDetector as _TypeDetector
            detector = _TypeDetector()
            dr = detector.detect(df)
            st.session_state["detection_result"] = dr
            st.session_state["step_preprocess_done"] = True
            _advance("Phase 2: 前処理")

        # ── Phase 3: AutoML ──────────────────────────────
        if do_ml:
            phase_status.markdown("**Phase 3/6 — AutoML 実行中（しばらくお待ちください）…**")
            automl_log: list[str] = []

            def _cb(step: int, total: int, msg: str) -> None:
                progress_bar.progress(
                    (phase_done / max(TOTAL_PHASES, 1))
                    + (step / total) / max(TOTAL_PHASES, 1)
                )
                _log(f"  [{step}/{total}] {msg}")

            cfg = PreprocessConfig(
                numeric_scaler=numeric_scaler,
                exclude_smiles=True,
                exclude_constant=True,
            )
            engine = AutoMLEngine(
                task=task_override,
                cv_folds=cv_folds,
                model_keys=models if models else None,
                timeout_seconds=timeout,
                progress_callback=_cb,
            )
            # engine.run に extra_params を渡す (backend側での対応が必要な場合は別途修正)
            automl_res = engine.run(
                df, target_col=target_col, smiles_col=smiles_col,
                preprocess_config=cfg,
                cv_extra_params=extra_params.get("cv", {}),
            )
            result.automl_result = automl_res
            result.task = automl_res.task if hasattr(automl_res, "task") else task_override
            st.session_state["automl_result"] = automl_res
            if automl_res.warnings:
                result.warnings.extend(automl_res.warnings)
            _advance("Phase 3: AutoML")

        X_base = df.drop(columns=[target_col])
        if result.automl_result and getattr(result.automl_result, "processed_X", None) is not None:
            X_base = result.automl_result.processed_X

        # ── Phase 4: 評価 ────────────────────────────────
        if do_eval and result.automl_result is not None:
            phase_status.markdown("**Phase 4/6 — モデル評価中…**")
            try:
                ar = result.automl_result
                # best_pipeline は AutoMLEngine.run() で既に fit済み
                bp: Pipeline = ar.best_pipeline
                y = df[target_col].values

                y_pred = bp.predict(X_base)
                task_str = result.task if result.task in ("regression", "classification") else "regression"
                if task_str == "regression":
                    result.model_score = evaluate_regression(y, y_pred,
                                                             model_key=ar.best_model_key,
                                                             cv_mean=ar.best_score)
                else:
                    y_prob = bp.predict_proba(X_base) if hasattr(bp, "predict_proba") else None
                    result.model_score = evaluate_classification(y, y_pred, y_prob=y_prob,
                                                                 model_key=ar.best_model_key,
                                                                 cv_mean=ar.best_score)
            except Exception as e:
                result.warnings.append(f"評価エラー: {e}")
                _log(f"⚠️ 評価スキップ: {e}")
            _advance("Phase 4: 評価")

        # ── Phase 5: PCA 次元削減 ────────────────────────
        if do_pca:
            phase_status.markdown("**Phase 5/6 — PCA 次元削減中…**")
            try:
                # 数値列が2列以上あれば実行可能（SMILES記述子も含める）
                if X_base.select_dtypes(include="number").shape[1] >= 2:
                    from backend.data.dim_reduction import DimReductionConfig as _DimReductionConfig
                    from backend.data.dim_reduction import DimReducer as _DimReducer
                    
                    pca_cfg = _DimReductionConfig(
                        method="pca", n_components=2, 
                        method_params=extra_params.get("pca", {})
                    )
                    reducer = _DimReducer(pca_cfg)
                    emb_df = reducer.fit_transform(X_base)
                    
                    result.pca_df  = pd.DataFrame(emb_df, columns=["PC1", "PC2"], index=df.index)
                    result.pca_evr = reducer.explained_variance_ratio_
            except Exception as e:
                result.warnings.append(f"PCAエラー: {e}")
                _log(f"⚠️ PCAスキップ: {e}")
            _advance("Phase 5: 次元削減")

        # ── Phase 6: SHAP 解析 ──────────────────────────
        if do_shap and result.automl_result is not None:
            phase_status.markdown("**Phase 6/6 — SHAP 特徴量重要度計算中…**")
            try:
                ar = result.automl_result
                bp: Pipeline = ar.best_pipeline
                # Pipelineの最終ステップ(model)から重要度を取得
                final_estimator = bp[-1] if hasattr(bp, '__getitem__') else bp

                # 変換後の特徴量名を取得
                try:
                    preprocessor_step = bp[:-1]  # model以外
                    X_transformed = preprocessor_step.transform(X_base)
                    try:
                        feat_names = bp[:-1].get_feature_names_out().tolist()
                    except Exception:
                        feat_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
                except Exception:
                    feat_names = X_base.select_dtypes(
                        include="number"
                    ).columns.tolist()

                if hasattr(final_estimator, "feature_importances_"):
                    imps = final_estimator.feature_importances_
                    names = feat_names[:len(imps)]
                    result.shap_importances = dict(
                        sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
                    )
                elif hasattr(final_estimator, "coef_"):
                    coefs = np.abs(final_estimator.coef_.ravel())
                    names = feat_names[:len(coefs)]
                    result.shap_importances = dict(
                        sorted(zip(names, coefs), key=lambda x: x[1], reverse=True)
                    )
                else:
                    # SHAP KernelExplainer（低速・フォールバック）
                    import shap
                    X_s = X_base.select_dtypes(include="number").fillna(0)
                    explainer = shap.KernelExplainer(
                        bp.predict, shap.sample(X_s, min(50, len(X_s)))
                    )
                    shap_vals = explainer.shap_values(X_s.head(50))
                    mean_abs = np.abs(shap_vals).mean(axis=0)
                    result.shap_importances = dict(
                        sorted(zip(X_s.columns.tolist(), mean_abs),
                               key=lambda x: x[1], reverse=True)
                    )
            except Exception as e:
                result.warnings.append(f"SHAP計算エラー: {e}")
                _log(f"⚠️ SHAPスキップ: {e}")
            _advance("Phase 6: SHAP")

    except Exception as e:
        progress_bar.empty()
        phase_status.empty()
        st.error(f"❌ パイプラインエラー: {e}")
        with st.expander("エラー詳細"):
            st.code(traceback.format_exc())
        return

    result.elapsed = time.time() - start_time
    st.session_state["pipeline_result"] = result
    st.session_state["step_eda_done"] = do_eda
    st.session_state["step_preprocess_done"] = do_prep

    progress_bar.progress(1.0)
    phase_status.empty()
    log_area.empty()
    st.balloons()
    st.rerun()


# ─────────────────────────────────────────────────────────────
# 結果表示
# ─────────────────────────────────────────────────────────────

def _show_pipeline_result(pr: PipelineResult) -> None:
    """PipelineResult を6フェーズタブで表示する。"""
    ar = pr.automl_result

    # ヘッダー
    best_info = (
        f"最良モデル: **{ar.best_model_key}** | スコア: `{ar.best_score:.4f}`"
        if ar else "AutoML は実行されていません"
    )
    st.success(f"✅ パイプライン完了！ ({pr.elapsed:.1f}秒) | {best_info}")

    for w in pr.warnings:
        st.warning(f"⚠️ {w}")

    # タブ
    tabs = st.tabs(["📊 EDA", "⚙️ 前処理", "🤖 AutoML", "📈 評価", "📐 次元削減", "💡 SHAP"])

    # ── Tab 0: EDA ──────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 📊 データ探索サマリー")
        if pr.eda_summary:
            summ = pr.eda_summary
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("行数", f"{summ.get('n_rows', '-'):,}")
            c2.metric("列数", summ.get('n_cols', '-'))
            c3.metric("欠損セル数", summ.get('n_missing', '-'))
            c4.metric("重複行数", summ.get('n_duplicates', '-'))

        if pr.col_stats:
            st.markdown("#### 列ごとの統計")
            rows = []
            for cs in pr.col_stats:
                rows.append({
                    "列名": cs.name,
                    "型": cs.dtype,
                    "ユニーク数": cs.n_unique,
                    "欠損率": f"{cs.null_rate:.1%}",
                    "平均": f"{cs.mean:.3g}" if cs.mean is not None else "-",
                    "標準偏差": f"{cs.std:.3g}" if cs.std is not None else "-",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if pr.outlier_results:
            n_out_cols = sum(1 for r in pr.outlier_results if r.n_outliers > 0)
            st.info(f"IQR外れ値検出: **{n_out_cols}列** に外れ値あり")
        else:
            st.info("EDA は実行されていません。⚙️ 設定で有効化してください。")

    # ── Tab 1: 前処理 ────────────────────────────────────────
    with tabs[1]:
        st.markdown("### ⚙️ 前処理パイプライン")
        dr = st.session_state.get("detection_result")
        if dr:
            col_types: dict[str, list[str]] = {}
            for col, info in dr.column_info.items():
                t = info.col_type.value if hasattr(info.col_type, "value") else str(info.col_type)
                col_types.setdefault(t, []).append(col)

            rows = [{"列型": t, "列数": len(cols), "列名": ", ".join(cols[:5]) + ("..." if len(cols) > 5 else "")}
                    for t, cols in col_types.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("TypeDetector によって自動で各列の型が判定され、最適なスケーラー/エンコーダが選定されます。")
        else:
            st.info("前処理は実行されていませんでした。⚙️ 設定で有効化してください。")

    # ── Tab 2: AutoML ────────────────────────────────────────
    with tabs[2]:
        if ar:
            _show_leaderboard(ar)
        else:
            st.info("AutoML は実行されていません。⚙️ 設定で有効化してください。")

    # ── Tab 3: 評価 ─────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 📈 モデル評価")
        score = pr.model_score
        if score is not None:
            # to_dict() はNoneフィールドを自動除外する
            score_dict = {
                k: v for k, v in score.to_dict().items()
                if k not in ("model_key", "task")
            }
            label_map = {
                "rmse": "RMSE", "mae": "MAE", "r2": "R²",
                "accuracy": "Accuracy", "f1_weighted": "F1 (weighted)",
                "roc_auc": "ROC-AUC", "cv_mean": "CV Mean",
                "cv_std": "CV Std", "train_time": "学習時間(s)",
            }
            c_cols = st.columns(min(len(score_dict), 4))
            for i, (k, v) in enumerate(score_dict.items()):
                with c_cols[i % 4]:
                    st.metric(label_map.get(k, k.upper()),
                              f"{v:.4f}" if isinstance(v, float) else str(v))
        else:
            st.info("評価は実行されていません。⚙️ 設定で有効化するか、先にAutoMLを実行してください。")

    # ── Tab 4: 次元削減 ─────────────────────────────────────
    with tabs[4]:
        st.markdown("### 📐 PCA 次元削減")
        if pr.pca_df is not None:
            evr = pr.pca_evr
            if evr is not None and len(evr) >= 2:
                st.caption(f"PC1 寄与率: {evr[0]:.1%} | PC2 寄与率: {evr[1]:.1%} | 累積: {evr.sum():.1%}")

            df_plot = pr.pca_df.copy()
            target_series = st.session_state["df"][st.session_state["target_col"]].values
            df_plot["target"] = target_series[:len(df_plot)]

            fig = go.Figure(go.Scatter(
                x=df_plot["PC1"], y=df_plot["PC2"],
                mode="markers",
                marker=dict(
                    color=df_plot["target"],
                    colorscale="Viridis",
                    showscale=True,
                    size=6,
                    opacity=0.7,
                ),
                text=[f"{st.session_state['target_col']}={v:.3g}" for v in df_plot["target"]],
                hovertemplate="%{text}<extra></extra>",
            ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0"),
                xaxis=dict(title="PC1", gridcolor="#333"),
                yaxis=dict(title="PC2", gridcolor="#333"),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("次元削減は実行されていません。⚙️ 設定で有効化してください。")

    # ── Tab 5: SHAP ─────────────────────────────────────────
    with tabs[5]:
        st.markdown("### 💡 特徴量重要度 (SHAP/Feature Importance)")
        if pr.shap_importances:
            top_n = 20
            items = list(pr.shap_importances.items())[:top_n]
            keys = [k for k, _ in reversed(items)]
            vals = [v for _, v in reversed(items)]

            fig = go.Figure(go.Bar(
                x=vals, y=keys,
                orientation="h",
                marker_color=[
                    f"rgba({int(255 * v / max(vals))}, 100, 255, 0.85)" for v in vals
                ],
                text=[f"{v:.4f}" for v in vals],
                textposition="outside",
            ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0"),
                xaxis=dict(title="重要度", gridcolor="#333"),
                yaxis=dict(gridcolor="#333"),
                height=max(300, len(keys) * 28),
                margin=dict(l=150, r=60, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP解析は実行されていません。⚙️ 設定で有効化してください。")

    # ── 次のアクションボタン ─────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 モデル評価（詳細）", use_container_width=True, key="goto_eval"):
            st.session_state["page"] = "evaluation"
            st.rerun()
    with c2:
        if st.button("💡 SHAP 詳細解析", use_container_width=True, key="goto_shap"):
            st.session_state["page"] = "interpret"
            st.rerun()
    with c3:
        if st.button("📐 次元削減（詳細）", use_container_width=True, key="goto_dimred"):
            st.session_state["page"] = "dim_reduction"
            st.rerun()


# ─────────────────────────────────────────────────────────────
# リーダーボード表示（共通）
# ─────────────────────────────────────────────────────────────

def _show_leaderboard(result: AutoMLResult) -> None:
    """モデルリーダーボードとスコアバーチャートを表示する。"""
    st.markdown("### 🏆 モデルリーダーボード")
    scores  = result.model_scores
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

    st.markdown("### 📊 スコア比較")
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    keys = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    colors = ["#7b2ff7" if k == result.best_model_key else "#00d4ff" for k in keys]

    fig = go.Figure(go.Bar(
        x=vals, y=keys, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
        xaxis=dict(gridcolor="#333", title=result.scoring),
        yaxis=dict(gridcolor="#333"),
        height=max(300, len(keys) * 40),
        margin=dict(l=120, r=50, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)
