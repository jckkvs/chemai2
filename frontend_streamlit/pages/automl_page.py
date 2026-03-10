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
    smiles_transformer: Any = None
    smiles_correlations: dict[str, float] = field(default_factory=dict)


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
            selected_descriptors = run_config.get("selected_descriptors", None),
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
    # ── Expert 設定パネル（整理されたタブ形式） ────────────────
    st.markdown("### 🛠️ 専門家向け詳細設定")
    with st.expander("⚙️ パイプライン構成をカスタマイズする", expanded=False):
        from backend.models.factory import list_models, get_model_registry
        from backend.models.cv_manager import list_cv_methods, _CV_REGISTRY
        import inspect

        # タブの定義
        cfg_tab1, cfg_tab2, cfg_tab3 = st.tabs([
            "🎯 基本構成 & モデル選択", 
            "🧠 各モデルの詳細引数", 
            "🔬 高度な設定 (CV・前処理)"
        ])

        # --- Tab 1: 基本構成 & モデル選択 ---
        with cfg_tab1:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**タスク & 実行フラグ**")
                task_override = st.selectbox(
                    "タスク種別",
                    ["auto", "regression", "classification"],
                    key="pl_task",
                    help="autoの場合は目的変数のデータ型から自動判定します"
                )
                
                # SMILES列の選択
                smiles_options = ["なし"] + df.columns.tolist()
                smiles_default_idx = 0
                for i, col in enumerate(df.columns):
                    if col.lower() == "smiles":
                        smiles_default_idx = i + 1
                        break
                smiles_col_raw = st.selectbox("SMILES列 (化合物データの場合)", smiles_options, index=smiles_default_idx, key="pl_smiles")
                smiles_col = None if smiles_col_raw == "なし" else smiles_col_raw
                
                timeout = st.slider("全体タイムアウト(秒)", 30, 3600, 300, key="pl_to")

            with col_t2:
                st.markdown("**評価対象モデルの選択**")
                task_for_models = st.session_state.get("task", "auto")
                if task_for_models == "auto":
                    # 仮判定
                    task_for_models = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
                
                all_model_list = list_models(task_for_models, available_only=True)
                all_model_keys = [m["key"] for m in all_model_list]
                
                default_sel_models = st.session_state.get("adv_models", all_model_keys)
                selected_model_keys = st.multiselect(
                    "評価に含めるモデル",
                    options=all_model_keys,
                    default=[k for k in default_sel_models if k in all_model_keys],
                    format_func=lambda k: next((m["name"] for m in all_model_list if m["key"] == k), k),
                    key="cfg_models",
                    help="AutoMLで比較するベースモデルを選択します"
                )

            st.markdown("---")
            st.markdown("**出力オプション**")
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                do_pca = st.checkbox("次元削減(PCA)を表示", value=True, key="pl_pca")
            with c_p2:
                do_shap = st.checkbox("SHAP解析(重要度)を表示", value=True, key="pl_shap")

        # --- Tab 2: 各モデルの詳細引数 ---
        with cfg_tab2:
            st.markdown("各モデルのハイパーパラメータを個別に調整できます。")
            # task_for_models を再計算（cfg_tab1 のローカル変数に依存しないように）
            _task2 = st.session_state.get("task", "auto")
            if _task2 in ("auto", "auto（自動）"):
                _task2 = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
            registry = get_model_registry(_task2)
            
            # Tab1 で選択されたモデルキーを確実に取得する
            active_models = selected_model_keys if selected_model_keys else st.session_state.get("cfg_models", [])
            model_params_dict = {}

            if not active_models:
                st.info("左のタブでモデルを選択してください。")
            else:
                for mkey in active_models:
                    entry = registry.get(mkey, {})
                    mname = entry.get("name", mkey)
                    m_item = entry.get("class") or entry.get("factory")
                    
                    if m_item:
                        with st.expander(f"⚙️ {mname} ({mkey})"):
                            target_func = m_item.__init__ if inspect.isclass(m_item) else m_item
                            try:
                                msig = inspect.signature(target_func)
                            except Exception:
                                st.warning(f"{mkey} の引数を取得できませんでした。")
                                continue
                                
                            default_ps = entry.get("default_params", {})
                            m_p_vals = {}
                            m_cols = st.columns(3)
                            m_idx = 0
                            
                            for pname, pinfo in msig.parameters.items():
                                if pname in ("self", "kwargs", "args"):
                                    continue
                                
                                dval = default_ps.get(pname, pinfo.default if pinfo.default is not inspect.Parameter.empty else None)
                                anno = pinfo.annotation
                                
                                with m_cols[m_idx % 3]:
                                    k_p = f"mp_{mkey}_{pname}"
                                    if isinstance(dval, bool) or anno is bool:
                                        m_p_vals[pname] = st.checkbox(pname, value=bool(dval) if dval is not None else False, key=k_p)
                                    elif isinstance(dval, int) or anno is int:
                                        iv = int(dval) if dval is not None and not isinstance(dval, str) else 0
                                        m_p_vals[pname] = st.number_input(pname, value=iv, key=k_p)
                                    elif isinstance(dval, float) or anno is float:
                                        fv = float(dval) if dval is not None else 0.0
                                        m_p_vals[pname] = st.number_input(pname, value=fv, format="%.4f", key=k_p)
                                    elif dval is None:
                                        raw = st.text_input(pname, value="", key=k_p, help="空白=None")
                                        m_p_vals[pname] = None if raw.strip() == "" else raw.strip()
                                    else:
                                        m_p_vals[pname] = st.text_input(pname, value=str(dval), key=k_p)
                                m_idx += 1
                            model_params_dict[mkey] = m_p_vals

        # --- Tab 3: 高度な設定 (CV・前処理) ---
        with cfg_tab3:
            st.markdown("**📊 交差検証 (Cross-Validation)**")
            cv_methods_all = list_cv_methods(task="regression")
            cv_key_map = {m["key"]: f"{m['name']}  —  {m['description']}" for m in cv_methods_all}

            cv_key_sel = st.selectbox(
                "CV手法",
                options=list(cv_key_map.keys()),
                format_func=lambda k: cv_key_map[k],
                index=0,
                key="cfg_cv_key",
            )

            cv_entry = _CV_REGISTRY.get(cv_key_sel, {})
            cv_class = cv_entry.get("class")
            cv_params: dict = {}

            if cv_class:
                sig = inspect.signature(cv_class.__init__)
                cv_p_cols = st.columns(3)
                p_idx = 0
                for pname, pinfo in sig.parameters.items():
                    if pname in ("self", "test_fold"): continue
                    dval = cv_entry.get("default_params", {}).get(pname, pinfo.default if pinfo.default is not inspect.Parameter.empty else None)
                    anno = pinfo.annotation
                    with cv_p_cols[p_idx % 3]:
                        k_cv = f"cv_p_{pname}"
                        if isinstance(dval, bool) or anno is bool:
                            cv_params[pname] = st.checkbox(pname, value=bool(dval), key=k_cv)
                        elif isinstance(dval, int) or anno is int:
                            cv_params[pname] = st.number_input(pname, value=int(dval) if dval is not None else 0, key=k_cv)
                        elif isinstance(dval, float) or anno is float:
                            cv_params[pname] = st.number_input(pname, value=float(dval) if dval is not None else 0.0, format="%.3f", key=k_cv)
                        else:
                            cv_params[pname] = st.text_input(pname, value=str(dval) if dval is not None else "", key=k_cv)
                    p_idx += 1

            requires_groups = cv_entry.get("requires_groups", False)
            groups_sel = st.selectbox("グループ列 (オプション)", ["なし"] + df.columns.tolist(), key="cfg_group_col")
            cfg_group_col = None if groups_sel == "なし" else groups_sel

            st.markdown("---")
            st.markdown("**🔧 前処理 (Preprocessing)**")
            ps_col1, ps_col2 = st.columns(2)
            with ps_col1:
                st.markdown("*数値列*")
                numeric_scaler = st.selectbox("スケーラー", ["auto", "standard", "minmax", "robust", "none"], key="cfg_scaler")
                numeric_imputer = st.selectbox("欠損補完", ["mean", "median", "knn", "constant"], key="cfg_num_imputer")
                add_missing_ind = st.checkbox("欠損指標列を追加", value=True, key="cfg_miss_ind")
            with ps_col2:
                st.markdown("*カテゴリ列*")
                cat_low_encoder = st.selectbox("エンコーダ(低)", ["onehot", "ordinal", "target"], key="cfg_cat_low")
                cat_high_encoder = st.selectbox("エンコーダ(高)", ["ordinal", "target", "binary"], key="cfg_cat_high")
                cat_imputer = st.selectbox("欠損補完", ["most_frequent", "constant"], key="cfg_cat_imputer")


    # デフォルト値（設定パネルを開いていない場合）
    if "cfg_cv_key" not in st.session_state:
        cv_key_sel = "kfold"
        cv_params = {}
        cfg_group_col = None
        numeric_scaler = "auto"
        numeric_imputer = "mean"
        add_missing_ind = True
        cat_low_encoder = "onehot"
        cat_high_encoder = "ordinal"
        cat_imputer = "most_frequent"
        selected_model_keys = []
        model_params_dict = {}
        task_override = "auto"
        smiles_col = None
        timeout = 300
        do_pca = True
        do_shap = True
    else:
        # すべてセッションステートから取得
        cv_key_sel = st.session_state.get("cfg_cv_key", "kfold")
        cfg_group_col = None if st.session_state.get("cfg_group_col", "なし") == "なし" else st.session_state.get("cfg_group_col")
        numeric_scaler = st.session_state.get("cfg_scaler", "auto")
        numeric_imputer = st.session_state.get("cfg_num_imputer", "mean")
        add_missing_ind = st.session_state.get("cfg_miss_ind", True)
        cat_low_encoder = st.session_state.get("cfg_cat_low", "onehot")
        cat_high_encoder = st.session_state.get("cfg_cat_high", "ordinal")
        cat_imputer = st.session_state.get("cfg_cat_imputer", "most_frequent")
        selected_model_keys = st.session_state.get("cfg_models", [])
        
        # model_params_dict は selected_model_keys に基づいて構築済み
        task_override = st.session_state.get("pl_task", "auto")
        smiles_col_raw = st.session_state.get("pl_smiles", "なし")
        smiles_col = None if smiles_col_raw == "なし" else smiles_col_raw
        timeout = st.session_state.get("pl_to", 300)
        do_pca = st.session_state.get("pl_pca", True)
        do_shap = st.session_state.get("pl_shap", True)

    # extra_params は後方互換のため空に
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
        numeric_imputer=numeric_imputer,
        add_missing_indicator=add_missing_ind,
        cat_low_encoder=cat_low_encoder,
        cat_high_encoder=cat_high_encoder,
        categorical_imputer=cat_imputer,
        task_override=task_override,
        cv_key=cv_key_sel,
        cv_params=cv_params,
        cv_groups_col=cfg_group_col,
        models=selected_model_keys,
        model_params=model_params_dict,
        timeout=timeout,
        do_eda=True,
        do_prep=True,
        do_ml=True,
        do_eval=True,
        do_pca=do_pca,
        do_shap=do_shap,
        selected_descriptors=st.session_state.get("adv_desc"),
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
    # 前処理設定
    numeric_scaler: str = "auto",
    numeric_imputer: str = "mean",
    add_missing_indicator: bool = True,
    cat_low_encoder: str = "onehot",
    cat_high_encoder: str = "ordinal",
    categorical_imputer: str = "most_frequent",
    # CV設定
    task_override: str = "auto",
    cv_key: str = "auto",
    cv_params: dict | None = None,
    cv_groups_col: str | None = None,
    cv_folds: int = 5,
    # モデル設定
    models: list[str] | None = None,
    model_params: dict[str, dict] | None = None,
    # その他
    timeout: int = 300,
    do_eda: bool = True,
    do_prep: bool = True,
    do_ml: bool = True,
    do_eval: bool = True,
    do_pca: bool = True,
    do_shap: bool = True,
    selected_descriptors: list[str] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> None:
    """Full pipeline. All CV, preprocessor, and model settings are fully configurable."""
    extra_params = extra_params or {}
    TOTAL_PHASES = sum([do_eda, do_prep, do_ml, do_eval, do_pca, do_shap])
    if smiles_col and smiles_col in df.columns:
        TOTAL_PHASES += 1
        
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
        # ── Phase 0: SMILES記述子変換 ────────────────────
        if smiles_col and smiles_col in df.columns:
            phase_status.markdown("**Phase 0 — SMILES記述子変換 実行中…**")
            try:
                from backend.chem.smiles_transformer import SmilesDescriptorTransformer
                smiles_transformer = SmilesDescriptorTransformer(
                    smiles_col=smiles_col,
                    selected_descriptors=selected_descriptors,
                )
                df = smiles_transformer.fit_transform(df)
                result.smiles_transformer = smiles_transformer
                
                # 相関再計算
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    desc_cols = smiles_transformer._descriptor_cols
                    valid_cols = [c for c in desc_cols if c in df.columns]
                    if valid_cols:
                        corr = df[valid_cols].corrwith(df[target_col]).dropna()
                        result.smiles_correlations = corr.to_dict()
                _advance("Phase 0: SMILES変換")
            except Exception as e:
                result.warnings.append(f"SMILES変換エラー: {e}")
                _log(f"⚠️ SMILES変換スキップ: {e}")
                # df = df.drop(columns=[smiles_col], errors="ignore")
            # 二重処理を防ぐためにNoneにしていたが、Pipeline側で変換させるために維持する
            # smiles_col = None 

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
            numeric_imputer=numeric_imputer,
            add_missing_indicator=add_missing_indicator,
            cat_low_encoder=cat_low_encoder,
            cat_high_encoder=cat_high_encoder,
            categorical_imputer=categorical_imputer,
        )
        engine = AutoMLEngine(
            task=task_override,
            cv_folds=cv_folds,
            cv_key=cv_key,
            cv_groups_col=cv_groups_col,
            model_keys=models if models else None,
            model_params=model_params or {},
            timeout_seconds=timeout,
            progress_callback=_cb,
            selected_descriptors=selected_descriptors,
        )
        automl_res = engine.run(
            df, target_col=target_col, smiles_col=smiles_col,
            preprocess_config=cfg,
            cv_extra_params=cv_params or {},
        )
        result.automl_result = automl_res
        result.task = automl_res.task if hasattr(automl_res, "task") else task_override
        st.session_state["automl_result"] = automl_res
        if automl_res.warnings:
            result.warnings.extend(automl_res.warnings)
        _advance("Phase 3: AutoML")

        # X_base: AutoML側で目的変数の欠損除外等が適用された「実際に学習に使われたデータ」
        if result.automl_result and result.automl_result.processed_X is not None:
            X_base = result.automl_result.processed_X
            y_base = result.automl_result.oof_true if result.automl_result.oof_true is not None else df.loc[X_base.index, target_col].values
        else:
            X_base = df.drop(columns=[target_col])
            y_base = df[target_col].values

        # ── Phase 4: 評価 ────────────────────────────────
        if do_eval and result.automl_result is not None:
            phase_status.markdown("**Phase 4/6 — モデル評価中…**")
            try:
                ar = result.automl_result
                bp: Pipeline = ar.best_pipeline
                y = y_base

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
                    
                    result.pca_df  = pd.DataFrame(emb_df, columns=["PC1", "PC2"], index=X_base.index)
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
                    if 'X_transformed' in locals() and X_transformed is not None:
                        # 変換済みの特徴量と最終モデルを使う
                        X_s = pd.DataFrame(X_transformed, columns=feat_names[:X_transformed.shape[1]]) if isinstance(X_transformed, np.ndarray) else X_transformed.copy()
                        if not hasattr(X_s, "columns"):
                            X_shape = X_s.shape[1] if hasattr(X_s, 'shape') else len(X_s[0])
                            X_s = pd.DataFrame(X_s, columns=[f"f{i}" for i in range(X_shape)])
                        
                        predict_fn = final_estimator.predict
                    else:
                        X_s = X_base.select_dtypes(include="number").fillna(0)
                        
                        # Pipelineに渡す場合、SHAPが内部でNumPy化するため元に戻すラッパーを噛ませる
                        def _wrapped_predict(X_arr):
                            df_in = pd.DataFrame(X_arr, columns=X_s.columns)
                            return bp.predict(df_in)
                            
                        predict_fn = _wrapped_predict

                    X_s = X_s.fillna(0)
                    explainer = shap.KernelExplainer(
                        predict_fn, shap.sample(X_s, min(50, len(X_s)))
                    )
                    shap_vals = explainer.shap_values(X_s.head(50))
                    
                    if isinstance(shap_vals, list):
                        # 分類の場合、クラス1の重要度
                        val_arr = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                    else:
                        val_arr = shap_vals
                    mean_abs = np.abs(val_arr).mean(axis=0)
                    
                    cols = X_s.columns.tolist() if hasattr(X_s, "columns") else feat_names[:len(mean_abs)]
                    result.shap_importances = dict(
                        sorted(zip(cols, mean_abs), key=lambda x: x[1], reverse=True)
                    )
            except Exception as e:
                result.warnings.append(f"SHAP計算エラー: {e}")
                _log(f"⚠️ SHAPスキップ: {e}")
            _advance("Phase 6: SHAP")

    except Exception as e:
        progress_bar.empty()
        phase_status.empty()
        st.error(f"❌ パイプライン実行中にエラーが発生しました: {e}")
        st.info("データに予測可能な特徴量が含まれていない、もしくは全ての列が定数などにより前処理で除外された可能性があります。")
        
        with st.expander("🛠️ エラー詳細（スタックトレース）"):
            st.code(traceback.format_exc(), language="python")
            
        # エラー発生時点でのパイプライン構成図（もし構築されていれば）
        ar = result.automl_result
        if ar and hasattr(ar, "best_pipeline") and ar.best_pipeline:
            with st.expander("🔍 エラー時の暫定パイプライン構成図"):
                try:
                    from sklearn.utils import estimator_html_repr
                    import streamlit.components.v1 as components
                    html_repr = estimator_html_repr(ar.best_pipeline)
                    components.html(html_repr, height=400, scrolling=True)
                except Exception:
                    st.warning("パイプライン構成図の生成に失敗しました。")
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
    tabs = st.tabs(["📊 EDA", "⚙️ 前処理", "🤖 AutoML", "📈 評価", "📐 次元削減", "💡 SHAP", "🧪 SMILES相関"])

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
        if pr.pca_df is not None and pr.pca_evr is not None:
            c_p1, c_p2 = st.columns([1, 2])
            with c_p1:
                st.info(f"PC1 寄与率: {pr.pca_evr[0]:.1%}\nPC2 寄与率: {pr.pca_evr[1]:.1%}")
            with c_p2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pr.pca_df["PC1"], y=pr.pca_df["PC2"],
                    mode="markers",
                    marker=dict(size=8, color="#2E86B1", line=dict(width=1, color="white"))
                ))
                fig.update_layout(title="PCA Projection", xaxis_title="PC1", yaxis_title="PC2")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("次元削減は実行されていません。⚙️ 設定で有効化するか、2列以上の特徴量が必要です。")

    # ── Tab 5: SHAP ─────────────────────────────────────────
    with tabs[5]:
        st.markdown("### 💡 SHAP / 特徴量重要度")
        if pr.shap_importances:
            imp_df = pd.DataFrame(list(pr.shap_importances.items()), columns=["Feature", "Importance"])
            imp_df = imp_df.sort_values(by="Importance", ascending=True).tail(20) # Top 20
            
            fig = go.Figure(go.Bar(
                x=imp_df["Importance"], y=imp_df["Feature"],
                orientation='h', marker_color="#EF7C8E"
            ))
            fig.update_layout(title="Top 20 最重要特徴量", xaxis_title="Mean |SHAP Value| or Feature Importance", height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP解析は実行されていません。⚙️ 設定で有効化するか、サポート外のモデルです。")
            
    # ── Tab 6: SMILES 相関 ──────────────────────────────────
    with tabs[6]:
        if pr and hasattr(pr, "smiles_correlations") and pr.smiles_correlations:
            st.markdown("### 🧪 SMILES 記述子と目的変数の相関ランキング")
            st.caption("SMILESから抽出された各記述子と目的変数とのピアソン相関係数（絶対値順にランキング表示）")
            
            # DataFrame化して絶対値でソート
            corr_dict = pr.smiles_correlations
            df_corr = pd.DataFrame([
                {"記述子": k, "相関係数": v, "絶対値": abs(v)}
                for k, v in corr_dict.items()
            ]).sort_values("絶対値", ascending=False).drop(columns=["絶対値"]).reset_index(drop=True)
            
            # ランキングを見やすく表示
            df_corr.index = df_corr.index + 1
            st.dataframe(df_corr.style.background_gradient(subset=["相関係数"], cmap="coolwarm", vmin=-1.0, vmax=1.0), use_container_width=True)
        else:
            st.info("SMILES抽出が行われていないか、相関係数の計算ができませんでした。")


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
