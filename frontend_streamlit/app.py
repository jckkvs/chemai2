"""
frontend_streamlit/app.py

ChemAI ML Studio - Streamlit メインアプリ
Upload → Select → ワンクリック解析。初心者向けの隠蔽設定と専門家向けの詳細設定を兼備。
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from backend.chem.recommender import get_target_recommendation_by_name

# ── ページ設定 ──────────────────────────────────────────────
st.set_page_config(
    page_title="ChemAI ML Studio",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── グローバルCSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: white;
}
section[data-testid="stSidebar"] * { color: white !important; }

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0f0;
}
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.hero-title {
    background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.3rem;
}
.hero-sub {
    text-align: center;
    color: #8888aa;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-blue   { background: rgba(0,150,255,0.2);   color: #4db8ff; border: 1px solid #4db8ff; }
.badge-purple { background: rgba(123,47,247,0.2);  color: #c084fc; border: 1px solid #c084fc; }
.badge-green  { background: rgba(0,200,100,0.2);   color: #4ade80; border: 1px solid #4ade80; }
.badge-orange { background: rgba(255,160,0,0.2);   color: #fbbf24; border: 1px solid #fbbf24; }
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label { font-size: 0.85rem; color: #8888aa; margin-top: 0.3rem; }
.status-dot-green  { display:inline-block; width:8px; height:8px; border-radius:50%; background:#4ade80; margin-right:6px; }
.status-dot-yellow { display:inline-block; width:8px; height:8px; border-radius:50%; background:#fbbf24; margin-right:6px; }
.status-dot-gray   { display:inline-block; width:8px; height:8px; border-radius:50%; background:#555; margin-right:6px; }
.stButton>button {
    background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 2rem; transition: all 0.3s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,212,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# ── セッションステート初期化 ──────────────────────────────────
def _init_session() -> None:
    defaults = {
        "page": "home",
        "active_tab_idx": 0,          # 0=データ設定 1=解析実行 2=結果確認
        "df": None,
        "file_name": None,
        "detection_result": None,
        "automl_result": None,
        "pipeline_result": None,
        "target_col": None,
        "task": "auto",
        "smiles_col": None,
        "step_eda_done": False,
        "step_preprocess_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ── サイドバー（ステータス表示のみにスリム化） ──────────────────────
with st.sidebar:
    st.markdown("## ⚗️ ChemAI ML Studio")
    st.markdown("---")

    has_data   = st.session_state["df"] is not None
    has_target = bool(st.session_state.get("target_col"))
    has_result = st.session_state["automl_result"] is not None

    # ステップインジケーター
    steps = [
        ("📂 データ読込", has_data),
        ("🎯 目的変数設定", has_target),
        ("🚀 解析実行",   has_result),
    ]
    for step_label, done in steps:
        dot = "status-dot-green" if done else "status-dot-gray"
        st.markdown(
            f'<div style="font-size:0.85rem; margin:6px 0;">'
            f'<span class="{dot}"></span>{step_label}</div>',
            unsafe_allow_html=True,
        )

    if has_data:
        _df = st.session_state["df"]
        st.caption(f"{st.session_state['file_name']}")
        st.caption(f"{_df.shape[0]:,}行 × {_df.shape[1]}列")
    if has_result:
        r = st.session_state["automl_result"]
        st.caption(f"最良: {r.best_model_key} ({r.best_score:.4f})")

    st.markdown("---")
    # タブへの直接ジャンプボタン
    if st.button("📂 データ設定", use_container_width=True, key="sb_tab0"):
        st.session_state["active_tab_idx"] = 0
        st.rerun()
    if has_data and st.button("🚀 解析実行", use_container_width=True, key="sb_tab1"):
        st.session_state["active_tab_idx"] = 1
        st.rerun()
    if has_result and st.button("📊 結果確認", use_container_width=True, key="sb_tab2"):
        st.session_state["active_tab_idx"] = 2
        st.rerun()

    # 詳細ツールへの導線（折り畳み）
    st.markdown("---")
    with st.expander("🔬 詳細ツール（専門家）", expanded=False):
        expert_pages = [
            ("📂", "データ詳細",   "data_load",    has_data),
            ("🔍", "EDA 詳細",     "eda",           has_data),
            ("⚙️", "前処理設定",   "preprocess",    has_data),
            ("🔬", "パイプライン", "pipeline",      has_data),
            ("📐", "次元削減",     "dim_reduction", has_data),
            ("🧬", "化合物解析",   "chem",          True),
            ("📚", "推奨変数ヘルプ","help_page",    True),
        ]
        for icon, label, pkey, enabled in expert_pages:
            if enabled:
                if st.button(f"{icon} {label}", key=f"exp_{pkey}",
                             use_container_width=True):
                    st.session_state["page"] = pkey
                    st.rerun()
            else:
                st.markdown(
                    f'<span style="color:#444466; font-size:0.85rem;">{icon} {label}</span>',
                    unsafe_allow_html=True,
                )

# ── ページルーティング ────────────────────────────────────────
page = st.session_state["page"]

# ===============================================================
# メインUI — 詳細ツールページ以外は 3 タブ構造
# ===============================================================
_EXPERT_PAGES = {"data_load", "eda", "preprocess", "pipeline", "evaluation",
                 "dim_reduction", "chem", "interpret", "help_page"}

if page in _EXPERT_PAGES:
    # サイドバーの「詳細ツール」から入ったページはそのまま表示
    st.markdown(
        f'<div style="margin-bottom:0.5rem;">'
        f'<button onclick="window.history.back()" style="background:none;border:none;'
        f'color:#8888aa;cursor:pointer;font-size:0.9rem;">← 戻る</button></div>',
        unsafe_allow_html=True,
    )
    # 「メインに戻る」ボタン
    if st.button("⬅️ メイン画面に戻る", key="back_to_main"):
        st.session_state["page"] = "home"
        st.rerun()

    if page == "data_load":
        from frontend_streamlit.pages.pipeline import data_load_page
        data_load_page.render()
    elif page == "eda":
        from frontend_streamlit.pages.pipeline import eda_page
        eda_page.render()
    elif page == "preprocess":
        from frontend_streamlit.pages.pipeline import preprocess_page
        preprocess_page.render()
    elif page == "evaluation":
        from frontend_streamlit.pages.pipeline import evaluation_page
        evaluation_page.render()
    elif page == "dim_reduction":
        from frontend_streamlit.pages.pipeline import dim_reduction_page
        dim_reduction_page.render()
    elif page == "chem":
        from frontend_streamlit.pages.tools import chem_page
        chem_page.render()
    elif page == "interpret":
        from frontend_streamlit.pages.pipeline import interpret_page
        interpret_page.render()
    elif page == "pipeline":
        from frontend_streamlit.pages.pipeline import pipeline_page
        pipeline_page.render()
    elif page == "help_page":
        from frontend_streamlit.pages import help_page
        help_page.render_help_page()

else:
    # ── メイン画面：3 タブ ──────────────────────────────────
    has_data   = st.session_state["df"] is not None
    has_result = st.session_state["automl_result"] is not None

    # ヘッダー
    st.markdown('<div class="hero-title">⚗️ ChemAI ML Studio</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">ファイルをアップロードして目的変数を選ぶだけ。'
        'あとは自動でEDA・機械学習・評価・SHAP解析まで完結します。</div>',
        unsafe_allow_html=True,
    )

    # ── ステップインジケーター ──────────────────────────────
    step_html = ""
    step_defs = [
        ("📂 データ設定", has_data),
        ("🚀 解析実行",   has_result),
        ("📊 結果確認",   has_result),
    ]
    for s_label, s_done in step_defs:
        color = "#4ade80" if s_done else "#555577"
        step_html += (
            f'<span style="color:{color}; font-weight:600; margin-right:1rem;">'
            f'{"✅" if s_done else "⬜"} {s_label}</span>'
            f'<span style="color:#444466; margin-right:1rem;">→</span>'
        )
    st.markdown(
        f'<div style="text-align:center; font-size:0.9rem; margin-bottom:1rem;">'
        f'{step_html.rstrip("→ ")}</div>',
        unsafe_allow_html=True,
    )

    # ── タブラベル（未完了はグレー表示） ────────────────────
    tab_labels = [
        "📂 ① データ設定",
        "🚀 ② 解析実行",
        "📊 ③ 結果確認",
    ]
    tab1, tab2, tab3 = st.tabs(tab_labels)

    # ──────────────────────────────────────────────────────────
    # TAB 1: データ設定（アップロード + 目的変数 + 詳細設定）
    # ──────────────────────────────────────────────────────────
    with tab1:
        # セッションが解析済みなら再解析ボタンも表示
        if has_result:
            st.info("🔄 設定を変えて再解析したい場合は、目的変数や詳細設定を変更してから「② 解析実行」タブへ進んでください。")

        # ── ファイルアップロードゾーン ────────────────────────────
        from backend.data.loader import load_from_bytes, get_supported_extensions
        from backend.data.type_detector import TypeDetector

        ext_list = get_supported_extensions()
        uploaded = st.file_uploader(
            "📂 分析したいデータファイルをドロップ",
            type=[e.lstrip(".") for e in ext_list],
            help=f"対応形式: {', '.join(ext_list)}",
            label_visibility="visible",
        )
    
        # サンプルデータボタン（ファイルがない場合のみ表示）
        if uploaded is None and st.session_state["df"] is None:
            st.markdown(
                '<div style="text-align:center; color:#555; margin:0.5rem 0;">または</div>',
                unsafe_allow_html=True,
            )
            def _make_sample(name: str, df: pd.DataFrame, set_smiles: bool = True) -> None:
                st.session_state["df"]               = df
                st.session_state["file_name"]        = name
                st.session_state["automl_result"]    = None
                st.session_state["pipeline_result"]  = None
                st.session_state["precalc_done"]     = False
                st.session_state["smiles_col"]       = None
                detector = TypeDetector()
                dr = detector.detect(df)
                st.session_state["detection_result"] = dr
                if set_smiles and dr.smiles_columns:
                    st.session_state["smiles_col"] = dr.smiles_columns[0]
                elif set_smiles:
                    for col in df.columns:
                        if col.lower() == "smiles":
                            st.session_state["smiles_col"] = col
                            break
                st.session_state["target_col"] = df.columns[-1]
    
            with st.expander("🔧 デバッグ用サンプルデータ", expanded=False):
                st.caption("開発・テスト用。通常はファイルをアップロードしてください。")
                use_smiles = st.checkbox("SMILES（化合物構造）列を含める", value=False, key="demo_smiles")
                c_r, c_c = st.columns(2)
                
                # ダミーSMILESリスト
                _DUMMY_SMILES = ["C", "CC", "CCC", "CCO", "CCN", "c1ccccc1", "c1ccccc1O", "CC(=O)O", "CC(C)C", "C1CCCCC1", "c1ccncc1", "c1ncncn1", "C1COCCO1"]
                
                with c_r:
                    if st.button("🧪 回帰サンプル", use_container_width=True, key="demo_reg"):
                        np.random.seed(42); n = 200
                        if use_smiles:
                            base_df = pd.DataFrame({
                                "SMILES": np.random.choice(_DUMMY_SMILES, n),
                                "solubility_logS": np.random.randn(n) * 2 - 2,
                            })
                        else:
                            base_df = pd.DataFrame({
                                "temperature": np.random.uniform(20, 80, n),
                                "pressure":    np.random.exponential(5, n),
                                "catalyst":    np.random.choice(["A型","B型","C型"], n),
                                "time_h":      np.random.uniform(1, 24, n),
                                "is_active":   np.random.randint(0, 2, n),
                                "yield":       np.random.randn(n) * 10 + 75,
                            })
                        _make_sample("sample_regression.csv", base_df, set_smiles=use_smiles)
                        st.session_state["task"] = "regression"
                        st.session_state["adv_models"] = ["lr", "rf"]  # デバッグ用デフォルトモデル固定
                        st.rerun()
                with c_c:
                    if st.button("🏷️ 分類サンプル", use_container_width=True, key="demo_cls"):
                        np.random.seed(42); n = 200
                        if use_smiles:
                            base_df = pd.DataFrame({
                                "SMILES": np.random.choice(_DUMMY_SMILES, n),
                                "is_toxic": np.random.randint(0, 2, n),
                            })
                        else:
                            base_df = pd.DataFrame({
                                "feature_1": np.random.randn(n),
                                "feature_2": np.random.randn(n),
                                "category":  np.random.choice(["低","中","高"], n),
                                "numeric":   np.random.randint(1, 100, n),
                                "label":     np.random.randint(0, 2, n),
                            })
                        _make_sample("sample_classification.csv", base_df, set_smiles=use_smiles)
                        st.session_state["task"] = "classification"
                        st.session_state["adv_models"] = ["logreg", "rf"]  # デバッグ用デフォルトモデル固定
                        st.rerun()
    
            with st.expander("📚 オープンベンチマークデータをロード"):
                st.markdown("ケモインフォマティクスの評価でよく使われる公開データセットです。")
                
                c_e, c_f, c_l = st.columns(3)
                with c_e:
                    st.markdown("**ESOL** (水溶解度)")
                    st.caption("1,128化合物の実測logS。非常に一般的。")
                    if st.button("📥 ロード", key="load_esol", use_container_width=True):
                        with st.spinner("ダウンロード中..."):
                            try:
                                from backend.data.benchmark_datasets import load_benchmark
                                df_bench = load_benchmark("esol")
                                _make_sample("benchmark_esol.csv", df_bench)
                                st.session_state["target_col"] = "measured log solubility in mols per litre"
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                with c_f:
                    st.markdown("**FreeSolv** (水和自由エネ)")
                    st.caption("642化合物の水和自由エネルギー。")
                    if st.button("📥 ロード", key="load_free", use_container_width=True):
                        with st.spinner("ダウンロード中..."):
                            try:
                                from backend.data.benchmark_datasets import load_benchmark
                                df_bench = load_benchmark("freesolv")
                                _make_sample("benchmark_freesolv.csv", df_bench)
                                st.session_state["target_col"] = "expt"
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                with c_l:
                    st.markdown("**Lipophilicity** (脂溶性)")
                    st.caption("AstraZeneca提供のlogDデータ。4,200件。")
                    if st.button("📥 ロード", key="load_lipo", use_container_width=True):
                        with st.spinner("ダウンロード中..."):
                            try:
                                from backend.data.benchmark_datasets import load_benchmark
                                df_bench = load_benchmark("lipophilicity")
                                _make_sample("benchmark_lipophilicity.csv", df_bench)
                                st.session_state["target_col"] = "exp"
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
    
        # ── ファイル読み込み処理 ─────────────────────────────────
        if uploaded is not None:
            try:
                with st.spinner("読み込み中..."):
                    raw = uploaded.read()
                    df_new = load_from_bytes(raw, uploaded.name)
                st.success(f"✅ `{uploaded.name}` 読み込み完了")
                # セッション更新
                st.session_state["df"]             = df_new
                st.session_state["file_name"]      = uploaded.name
                st.session_state["automl_result"]  = None
                st.session_state["pipeline_result"] = None
                st.session_state["smiles_col"]     = None  # 必ずリセット
                detector = TypeDetector()
                dr = detector.detect(df_new)
                st.session_state["detection_result"] = dr
                # SMILES列を自動検出（TypeDetectorが検出した場合）
                if dr.smiles_columns:
                    st.session_state["smiles_col"] = dr.smiles_columns[0]
                # TypeDetectorが検出できなくても列名が'smiles'なら設定
                else:
                    for col in df_new.columns:
                        if col.lower() == "smiles":
                            st.session_state["smiles_col"] = col
                            break
                st.session_state["target_col"] = df_new.columns[-1]  # 初期値
            except Exception as e:
                st.error(f"❌ 読み込みエラー: {e}")
    
        # ── データがある場合: 設定 + 実行エリア ──────────────────
        df = st.session_state.get("df")
        if df is not None:
            st.markdown("---")
    
            # データ概要 (コンパクト)
            c1, c2, c3, c4 = st.columns(4)
            for col, val, lbl in [
                (c1, f"{df.shape[0]:,}", "行数"),
                (c2, str(df.shape[1]), "列数"),
                (c3, f"{df.isna().mean().mean():.1%}", "欠損率"),
                (c4, str(df.select_dtypes(include='number').shape[1]), "数値列数"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="font-size:1.4rem;">{val}</div>'
                        f'<div class="metric-label">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )
    
            st.markdown("")
    
            # ── 目的変数（必須・常に表示） ──────────────────────
            col_target, col_task = st.columns([3, 2])
            with col_target:
                cur_target = st.session_state.get("target_col") or df.columns[-1]
                cur_idx = df.columns.tolist().index(cur_target) if cur_target in df.columns else -1
                target = st.selectbox(
                    "🎯 目的変数（予測したい列）",
                    options=df.columns.tolist(),
                    index=cur_idx,
                    key="home_target",
                )
                # 目的変数が変更されたら記述子の事前計算（相関用）をリセットする
                if st.session_state.get("target_col_prev") != target:
                    st.session_state["precalc_done"] = False
                    st.session_state["target_col_prev"] = target
                st.session_state["target_col"] = target
                
                # 目的変数に基づく推奨説明変数のヒント表示
                rec = get_target_recommendation_by_name(target)
                if rec:
                    st.info(f"💡 **推奨される説明変数**: {rec.summary} \n\n" + 
                            ", ".join([f"`{d.name}` ({d.library})" for d in rec.descriptors]))
            with col_task:
                task_opt = st.selectbox(
                    "📋 タスク種別",
                    ["auto（自動）", "regression（回帰）", "classification（分類）"],
                    key="home_task",
                )
                st.session_state["task"] = task_opt.split("（")[0]
    
            # ── 詳細設定（折り畳み：初心者には見えない） ────────
            with st.expander("⚙️ 詳細設定（任意）", expanded=False):
                c_top1, c_top2 = st.columns(2)
                with c_top1:
                    st.markdown("**ML構成設定**")
                    cv_folds   = st.slider("CV分割数", 2, 10, 5, key="adv_cv")
                    timeout    = st.slider("タイムアウト(秒)", 30, 3600, 300, key="adv_to")
                    
                with c_top2:
                    st.markdown("**前処理・フェーズ設定**")
                    c_sc, c_sm = st.columns(2)
                    with c_sc:
                        scaler = st.selectbox("スケーラー", ["auto","standard","robust","minmax","none"], key="adv_sc")
                    with c_sm:
                        smiles_options = ["なし"] + df.columns.tolist()
                        smiles_default_idx = 0
                        for i, col in enumerate(df.columns):
                            if col.lower() == "smiles":
                                smiles_default_idx = i + 1
                                break
                        smiles_raw = st.selectbox("SMILES列", smiles_options, index=smiles_default_idx, key="adv_sm")
                        new_smiles_col = None if smiles_raw == "なし" else smiles_raw
                        # SMILES列が変わったら事前計算をリセット
                        if st.session_state.get("smiles_col") != new_smiles_col:
                            st.session_state["precalc_done"] = False
                            st.session_state["precalc_smiles_df"] = None
                        st.session_state["smiles_col"] = new_smiles_col
                    
                    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
                    with c_p1: do_eda  = st.checkbox("EDA", value=True, key="adv_eda")
                    with c_p2: do_prep = st.checkbox("前処理", value=True, key="adv_prep")
                    with c_p3: do_eval = st.checkbox("評価", value=True, key="adv_eval")
                    with c_p4: do_pca  = st.checkbox("PCA", value=True, key="adv_pca")
                    do_shap = st.checkbox("SHAP解析", value=True, key="adv_shap")
    
                st.markdown("---")
                st.markdown("**🤖 使用するモデル（クリックで個別選択）**")
                
                from backend.models.factory import list_models, get_default_automl_models, get_model_registry
                import inspect
                _tmp_task = st.session_state.get("task", "auto")
                if _tmp_task == "auto":
                    _tmp_task = "regression" if pd.api.types.is_float_dtype(df[st.session_state.get("target_col")]) else "classification"
                
                available_models = list_models(task=_tmp_task, available_only=True)
                default_models = get_default_automl_models(task=_tmp_task)
    
                def _get_category(mkey, mname):
                    k = mkey.lower() + mname.lower()
                    if any(x in k for x in ["linear", "ridge", "lasso", "elastic", "logistic", "ard", "huber", "theilsen", "ransac", "pls", "sgd"]): return "線形系"
                    if any(x in k for x in ["svr", "svc", "support", "rbf", "kernel", "gaussian"]): return "カーネル系"
                    if any(x in k for x in ["tree", "forest", "boost", "gbm", "gradient"]): return "決定木系"
                    return "その他"
    
                categories = {"線形系": [], "カーネル系": [], "決定木系": [], "その他": []}
                for m in available_models:
                    cat = _get_category(m["key"], m["name"])
                    categories[cat].append(m)
    
                selected_models = st.session_state.get("adv_models", default_models)
                new_selection = []
                
                sel_tabs = st.tabs(list(categories.keys()))
                for cat_name, t_body in zip(categories.keys(), sel_tabs):
                    with t_body:
                        cat_cols = st.columns(4)
                        for idx, m in enumerate(categories[cat_name]):
                            with cat_cols[idx % 4]:
                                is_checked = m["key"] in selected_models
                                if not m["available"]:
                                    st.checkbox(f"{m['name']} (未実装)", value=False, disabled=True, key=f"c_{m['key']}")
                                else:
                                    if st.checkbox(m["name"], value=is_checked, key=f"c_{m['key']}"):
                                        new_selection.append(m["key"])
                
                st.session_state["adv_models"] = new_selection
                selected_models = new_selection
    
                st.markdown("---")
                st.markdown("**⚙️ モデル詳細設定（パラメータチューニング）**")
                _registry = get_model_registry(_tmp_task)
                st.session_state["model_params"] = {}
                
                if selected_models:
                    model_tabs = st.tabs([_registry.get(k, {}).get("name", k) for k in selected_models])
                    for mkey, m_tab in zip(selected_models, model_tabs):
                        _entry = _registry.get(mkey, {})
                        _mclass = _entry.get("class") or _entry.get("factory")
                        if _mclass:
                            with m_tab:
                                _target = _mclass.__init__ if inspect.isclass(_mclass) else _mclass
                                try:
                                    _sig = inspect.signature(_target)
                                    _dp = _entry.get("default_params", {})
                                    _m_p_vals = {}
                                    _m_cols = st.columns(3)
                                    _m_idx = 0
                                    for pname, pinfo in _sig.parameters.items():
                                        if pname in ("self", "kwargs", "args"): continue
                                        dval = _dp.get(pname, pinfo.default if pinfo.default is not inspect.Parameter.empty else None)
                                        anno = pinfo.annotation
                                        k_p = f"app_mp_{mkey}_{pname}"
                                        
                                        with _m_cols[_m_idx % 3]:
                                            if isinstance(dval, bool) or anno is bool:
                                                _m_p_vals[pname] = st.checkbox(pname, value=bool(dval) if dval is not None else False, key=k_p)
                                            elif isinstance(dval, int) or anno is int:
                                                iv = int(dval) if dval is not None and not isinstance(dval, str) else 0
                                                _m_p_vals[pname] = st.number_input(pname, value=iv, key=k_p)
                                            elif isinstance(dval, float) or anno is float:
                                                fv = float(dval) if dval is not None else 0.0
                                                _m_p_vals[pname] = st.number_input(pname, value=fv, format="%.4f", key=k_p)
                                            elif dval is None:
                                                raw = st.text_input(pname, value="", key=k_p, help="空白=None")
                                                _m_p_vals[pname] = None if raw.strip() == "" else raw.strip()
                                            else:
                                                _m_p_vals[pname] = st.text_input(pname, value=str(dval), key=k_p)
                                        _m_idx += 1
                                    st.session_state["model_params"][mkey] = _m_p_vals
                                except Exception:
                                    st.warning(f"引数を取得できません。")
    
                selected_desc = st.session_state.get("adv_desc", [])

            # ── SMILES 記述子設定（SMILES列指定時のみ表示） ─────────────────
            if st.session_state.get("smiles_col"):
                # --- 記述子事前計算 (相関分析用) ---
                if not st.session_state.get("precalc_done", False):
                    from backend.chem.rdkit_adapter import RDKitAdapter
                    from backend.chem.recommender import get_target_recommendation_by_name

                    smiles_series = df[st.session_state["smiles_col"]]
                    valid_mask = smiles_series.notna()
                    smiles_list = smiles_series[valid_mask].tolist()
                    valid_idx = smiles_series[valid_mask].index
                    n = len(smiles_list)

                    st.info(f"⚙️ **{n} 件のSMILES記述子を計算中...** しばらくお待ちください。")

                    target_name = st.session_state.get("target_col", "")
                    rdkit = RDKitAdapter(compute_fp=False)
                    df_result = pd.DataFrame(index=range(n))

                    with st.spinner("1/3 目的変数「{}」の推奨記述子を計算中...".format(target_name or "?")):
                        rec = get_target_recommendation_by_name(target_name)
                        rec_names = [d.name for d in rec.descriptors] if rec else []
                        if rec_names and rdkit.is_available():
                            try:
                                df_tmp = rdkit.compute(smiles_list, selected_descriptors=rec_names).descriptors
                                df_result = pd.concat([df_result, df_tmp], axis=1)
                            except Exception:
                                pass

                    with st.spinner("2/3 数え上げ系記述子（原子数、環数等）を計算中..."):
                        try:
                            mdata = rdkit.get_descriptors_metadata()
                            count_names = [m.name for m in mdata if m.is_count and m.name not in df_result.columns]
                            if count_names and rdkit.is_available():
                                df_tmp = rdkit.compute(smiles_list, selected_descriptors=count_names).descriptors
                                df_result = pd.concat([df_result, df_tmp], axis=1)
                        except Exception:
                            pass

                    CURATED = ["MolWt","LogP","TPSA","HBA","HBD","RotBonds",
                               "RingCount","AromaticRingCount","FractionCSP3",
                               "HeavyAtoms","MolMR","HallKierAlpha"]
                    with st.spinner("3/3 主要物理化学記述子（分子量・LogP・TPSA等）を計算中..."):
                        curated = [c for c in CURATED if c not in df_result.columns]
                        if curated and rdkit.is_available():
                            try:
                                df_tmp = rdkit.compute(smiles_list, selected_descriptors=curated).descriptors
                                df_result = pd.concat([df_result, df_tmp], axis=1)
                            except Exception:
                                pass

                    df_result = df_result.loc[:, ~df_result.columns.duplicated()]
                    df_result.index = valid_idx
                    df_result = df_result.apply(pd.to_numeric, errors="coerce").convert_dtypes()
                    n_descs = len(df_result.columns)
                    st.success(f"✅ 計算完了！ {n_descs} 個の記述子を抽出しました（{n}件）")
                    st.session_state["precalc_smiles_df"] = df_result
                    st.session_state["precalc_done"] = True
                    st.rerun()
    
                with st.expander("🧪 SMILES記述子設定（任意）", expanded=True):
                    st.markdown("SMILES列から**化学記述子**を計算・選択します。選択した記述子が解析の特徴量として使用されます。")

                    # MolAI PCA 次元数設定
                    with st.expander("🤖 MolAI 設定（CNN+GRU オートエンコーダー記述子）", expanded=False):
                        st.caption(
                            "MolAI は SMILES を CNN Encoder で高次元潜在ベクトルに変換し、"
                            "PCA で低次元化した記述子を生成します。\n\n"
                            "📄 *Mahdizadeh & Eriksson, J. Chem. Inf. Model. 2025, "
                            "DOI: 10.1021/acs.jcim.5c00491*"
                        )
                        _molai_n = st.slider(
                            "PCA 出力次元数 (n_components)",
                            min_value=4, max_value=128,
                            value=st.session_state.get("molai_n_components", 32),
                            step=4,
                            key="slider_molai_n",
                            help="MolAI の潜在ベクトルを何次元に圧縮するか。大きいほど情報量が増えるが過学習リスクも高まる。"
                        )
                        if _molai_n != st.session_state.get("molai_n_components", 32):
                            st.session_state["molai_n_components"] = _molai_n
                            st.session_state["precalc_done"] = False  # 再計算トリガー
                            st.rerun()

                    # --- データフレームのプレビューと型判定 ---
                    st.markdown("### 📊 データプレビュー")
                    import pandas as pd
                    if st.session_state.get("precalc_smiles_df") is not None:
                        # precalc_smiles_df を結合して表示
                        df_preview = pd.concat([df.head(100), st.session_state["precalc_smiles_df"].head(100).fillna(0.0)], axis=1)
                        # 型判定用に数値化を徹底
                        df_preview = df_preview.convert_dtypes()
                    else:
                        df_preview = df.head(100)
    
                    st.dataframe(df_preview, use_container_width=True)
                    
                    # --- 自動型判定 ---
                    st.markdown("### 🔍 変数型の自動判定結果 (TypeDetector)")
                    from backend.data.type_detector import TypeDetector
                    # TypeDetector用に推論精度を高めるため、一時的に数値化
                    detector_df = df_preview.apply(pd.to_numeric, errors='ignore').convert_dtypes()
                    detector = TypeDetector()
                    st.session_state["detection_result"] = detector.detect(detector_df)
    
                    from backend.chem import RDKitAdapter, XTBAdapter, CosmoAdapter, UniPkaAdapter, GroupContribAdapter, MordredAdapter, MolAIAdapter
                    from backend.chem.recommender import (
                        get_target_recommendation_by_name,
                        get_target_categories,
                        get_targets_by_category,
                        get_all_descriptor_categories,
                        get_descriptors_by_category,
                        get_all_target_recommendations
                    )
    
                    _molai_n_comp = st.session_state.get("molai_n_components", 32)
                    all_adapters = [RDKitAdapter(compute_fp=True), MordredAdapter(selected_only=True), XTBAdapter(), CosmoAdapter(), UniPkaAdapter(), GroupContribAdapter(), MolAIAdapter(n_components=_molai_n_comp)]
                    
                    # 辞書化してライブラリごとの記述子を持っておく（タブ3用）
                    lib_descriptors = {}
                    all_available_descriptors = []
                    descriptor_metadata_map = {} # 名前 -> DescriptorMetadata のマッピング
                    for adp in all_adapters:
                        if adp.is_available():
                            lib_name = adp.__class__.__name__.replace("Adapter", "")
                            names = adp.get_descriptor_names()
                            lib_descriptors[lib_name] = names
                            all_available_descriptors.extend(names)
                            
                            # メタデータを蓄積
                            # hasattrガードに加え、BaseChemAdapterで定義したデフォルト(空)も考慮
                            try:
                                mdata = adp.get_descriptors_metadata()
                                for meta in mdata:
                                    descriptor_metadata_map[meta.name] = meta
                            except (AttributeError, NotImplementedError):
                                # 未実装のアダプタはスキップ
                                pass
    
                    # --- 現在の選択状態（セッション）を取得 ---
                    # デフォルトは汎用的に有効な12種の物理化学記述子（目的変数によらずほぼ必ず有効）
                    _DEFAULT_DESCRIPTORS = [
                        "MolWt",           # 分子量：分子の大きさの基本指標、ほぼ全物性に相関
                        "LogP",            # 脂溶性：溶解度、膜透過、耐湿性に広く寄与
                        "TPSA",            # 位相的極性表面積：極性、溶解度、透過性の指標
                        "HBA",             # 水素結合受容体数：沸点、溶解度、機械的強度に影響
                        "HBD",             # 水素結合供与体数：同上（HBAと組み合わせることで効果的）
                        "RotBonds",        # 回転可能結合数：分子の柔軟性（Tg、粘度に影響）
                        "RingCount",       # 環の総数：剛直性、耐熱性の指標
                        "AromaticRingCount", # 芳香環数：UV吸収、屈折率、耐熱性に強く影響
                        "FractionCSP3",    # sp3炭素比率：溶解度、Tg・弾性率との逆相関
                        "HeavyAtoms",      # 重原子数：分子の大きさを別角度で表す
                        "MolMR",           # モル屈折：屈折率・分極率の直接指標
                        "HallKierAlpha",   # Hall-Kier α：枝分かれ度・形状の立体指標
                    ]
                    default_desc = st.session_state.get("adv_desc", _DEFAULT_DESCRIPTORS)
                    current_selected = set([d for d in default_desc if d in all_available_descriptors])
    
                    # 状態更新用コールバック（変更があった場合のみrerunする）
                    def update_desc_state(new_selection: set):
                        st.session_state["adv_desc"] = list(new_selection)
                        st.rerun()
    
                    # --- 事前計算済みデータからの相関算出 ---
                    precalc_df = st.session_state.get("precalc_smiles_df")
                    target_col_val = st.session_state.get("target_col", "")
                    target_series = df[target_col_val] if target_col_val in df.columns and pd.api.types.is_numeric_dtype(df[target_col_val]) else None
                    
                    corr_dict = {}
                    if precalc_df is not None and target_series is not None:
                        try:
                            aligned_target = target_series.loc[precalc_df.index]
                            corr_s = precalc_df.corrwith(aligned_target, method="pearson").abs()
                            corr_dict = corr_s.to_dict()
                        except Exception:
                            pass
                            
                    def sort_descriptors(desc_names):
                        """相関降順で記述子名のリストをソートする"""
                        if not corr_dict:
                            return desc_names
                        return sorted(desc_names, key=lambda d: corr_dict.get(d, 0.0) if pd.notna(corr_dict.get(d, 0.0)) else 0.0, reverse=True)
    
                    tab_corr, tab1, tab_count, tab2, tab3 = st.tabs(["相関係数から選ぶ", "目的変数の系統から選ぶ", "数え上げ系の変数から選ぶ", "記述子の意味から選ぶ", "計算ライブラリから選ぶ"])
    
                    with tab_corr:
                        st.markdown("事前に計算された記述子と目的変数との**相関係数（絶対値）**が高い順に、効果的と思われる記述子を選択できます。")
                        if not corr_dict:
                            st.warning("事前計算が完了していないか、目的変数が数値ではないため利用できません。（SMILES展開が完了するまでお待ちください）")
                        else:
                            c1, c2, c3 = st.columns([1, 1, 2])
                            sorted_corr_descs = sort_descriptors(list(corr_dict.keys()))
                            # 相関が算出できたものだけに絞る
                            sorted_corr_descs = [d for d in sorted_corr_descs if pd.notna(corr_dict.get(d))]
                            
                            if c1.button("上位10件を全選択", key="sel_top_10"):
                                update_desc_state(current_selected.union(sorted_corr_descs[:10]))
                            if c2.button("上位30件を全選択", key="sel_top_30"):
                                update_desc_state(current_selected.union(sorted_corr_descs[:30]))
                            
                            def format_corr_opt(d_name: str) -> str:
                                cval = corr_dict.get(d_name)
                                return f"{d_name} (相関: {cval:.3f})" if cval is not None else d_name
    
                            corr_selected = [d for d in sorted_corr_descs if d in current_selected]
                            
                            new_corr_selected = st.multiselect(
                                "相関順リストから記述子を選択",
                                options=sorted_corr_descs,
                                default=corr_selected,
                                format_func=format_corr_opt,
                                key="mlsel_corr"
                            )
                            
                            # 差分更新
                            if set(new_corr_selected) != set(corr_selected):
                                current_selected.difference_update(sorted_corr_descs)
                                current_selected.update(new_corr_selected)
                                update_desc_state(current_selected)
    
                    with tab1:
                        st.markdown("予測したい目的変数の系統（光、強度など）に合わせて、推奨される記述子のセットを一括で追加・削除できます。")
                        rec = get_target_recommendation_by_name(target_col_val)
                        if rec:
                            st.info(f"💡 現在選択中の目的変数「**{target_col_val}**」は「**{rec.category}**」に属します。\n\n{rec.summary}")
                        
                        categories = get_target_categories()
                        for cat in categories:
                            with st.expander(f"📁 {cat}"):
                                targets = get_targets_by_category(cat)
                                for t in targets:
                                    desc_names = [d.name for d in t.descriptors if d.name in all_available_descriptors]
                                    if not desc_names:
                                        continue
                                        
                                    col_t_left, col_t_right = st.columns([3, 1])
                                    with col_t_left:
                                        st.markdown(f"**{t.target_name}**")
                                        st.caption(t.summary)
                                    with col_t_right:
                                        # このターゲットの全記述子が既に選択されているかチェック
                                        is_all_selected = all(d in current_selected for d in desc_names)
                                        if st.button("一括追加" if not is_all_selected else "一括解除", key=f"btn_tgt_{t.target_name}"):
                                            if not is_all_selected:
                                                update_desc_state(current_selected.union(desc_names))
                                            else:
                                                update_desc_state(current_selected.difference(desc_names))
    
                    with tab_count:
                        st.markdown("原子数、環の数、水素結合数などの**「数え上げ（カウント）」**系記述子を選択できます。化学的な根拠に基づき厳密に定義されたリストのみが表示されます。")
                        
                        # メタデータに基づいて「数え上げ系」を抽出（曖昧なキーワード検索を廃止）
                        count_descs = [
                            dname for dname in all_available_descriptors 
                            if descriptor_metadata_map.get(dname) and descriptor_metadata_map[dname].is_count
                        ]
                        
                        if not count_descs:
                            st.info("利用可能な数え上げ系記述子が見つかりませんでした。")
                        else:
                            c1, c2, _ = st.columns([1, 1, 2])
                            if c1.button("数え上げ系を全選択", key="sel_all_count"):
                                update_desc_state(current_selected.union(count_descs))
                            if c2.button("数え上げ系を全解除", key="desel_all_count"):
                                update_desc_state(current_selected.difference(count_descs))
                            
                            sorted_count_descs = sort_descriptors(count_descs)
                            
                            def format_desc_with_meaning(d_name: str) -> str:
                                m = descriptor_metadata_map.get(d_name)
                                meaning_str = f" ({m.meaning})" if m and m.meaning != d_name else ""
                                cval = corr_dict.get(d_name)
                                corr_str = f" [相関: {cval:.3f}]" if cval is not None else ""
                                return f"{d_name}{meaning_str}{corr_str}"
    
                            cnt_selected = [d for d in sorted_count_descs if d in current_selected]
                            new_cnt_selected = st.multiselect(
                                "リストから抽出（相関順）",
                                options=sorted_count_descs,
                                default=cnt_selected,
                                format_func=format_desc_with_meaning,
                                key="mlsel_count"
                            )
                            if set(new_cnt_selected) != set(cnt_selected):
                                current_selected.difference_update(count_descs)
                                current_selected.update(new_cnt_selected)
                                update_desc_state(current_selected)
    
                    with tab2:
                        from backend.chem.recommender import (
                            get_all_descriptor_categories,
                            get_descriptors_by_category,
                        )
                        st.markdown("物理的・化学的意味のカテゴリごとに記述子を選択できます。")
                        desc_cats = get_all_descriptor_categories()
                        for dcat in desc_cats:
                            descs = get_descriptors_by_category(dcat)
                            valid_descs = [d for d in descs if d.name in all_available_descriptors]
                            if not valid_descs:
                                continue
                                
                            with st.expander(f"🧩 {dcat} ({len(valid_descs)}件)"):
                                # 全選択/全解除ボタン
                                c1, c2, _ = st.columns([1, 1, 3])
                                all_desc_names_in_cat = [d.name for d in valid_descs]
                                if c1.button("全選択", key=f"sel_all_{dcat}"):
                                    update_desc_state(current_selected.union(all_desc_names_in_cat))
                                if c2.button("全解除", key=f"desel_all_{dcat}"):
                                    update_desc_state(current_selected.difference(all_desc_names_in_cat))
                                    
                                # 相関が高い順にソートして表示
                                if corr_dict:
                                    valid_descs = sorted(valid_descs, key=lambda d: corr_dict.get(d.name, 0.0) if pd.notna(corr_dict.get(d.name, 0.0)) else 0.0, reverse=True)

                                for d in valid_descs:
                                    is_checked = d.name in current_selected
                                    
                                    corr_val = corr_dict.get(d.name)
                                    corr_str = f" | 相関: {corr_val:.2f}" if corr_val is not None and pd.notna(corr_val) else ""
                                    
                                    changed = st.checkbox(f"**{d.name}** ({d.library}): {d.meaning}{corr_str}", value=is_checked, key=f"chk_mean_{dcat}_{d.name}")
                                    if changed != is_checked:
                                        if changed:
                                            current_selected.add(d.name)
                                        else:
                                            current_selected.remove(d.name)
                                        st.session_state["adv_desc"] = list(current_selected)

                    with tab3:
                        st.markdown("計算エンジン（ライブラリ）ごとにすべての記述子を個別に選択できます。")
                        for lib, d_names in lib_descriptors.items():
                            if not d_names:
                                continue
                            with st.expander(f"⚙️ {lib} ({len(d_names)}件)"):
                                c1, c2, _ = st.columns([1, 1, 3])
                                if c1.button("全選択", key=f"sel_all_lib_{lib}"):
                                    update_desc_state(current_selected.union(d_names))
                                if c2.button("全解除", key=f"desel_all_lib_{lib}"):
                                    update_desc_state(current_selected.difference(d_names))
                                    
                                # マルチセレクトでまとめて編集させる
                                lib_selected = [d for d in d_names if d in current_selected]
                                
                                # 相関順にソートしたオプションリスト
                                sorted_opts = sort_descriptors(d_names)
                                
                                def format_lib_opt(d_name: str) -> str:
                                    base = d_name
                                    cval = corr_dict.get(d_name)
                                    if cval is not None and pd.notna(cval):
                                        base += f" (相関: {cval:.2f})"
                                    return base

                                new_lib_selected = st.multiselect(
                                    "個別記述子を選択",
                                    options=sorted_opts,
                                    default=lib_selected,
                                    format_func=format_lib_opt,
                                    key=f"mlsel_lib_{lib}"
                                )
                                if set(new_lib_selected) != set(lib_selected):
                                    current_selected.difference_update(d_names)
                                    current_selected.update(new_lib_selected)
                                    update_desc_state(current_selected)

                selected_desc = list(current_selected)
                st.caption(f"✅ 現在 {len(selected_desc)} 件の記述子が選択されています。")
                
                if len(selected_desc) > 0:
                    # --- 相関ヒートマップの統合強化版 ---
                    with st.expander("📊 選択変数と目的変数の相関ヒートマップ", expanded=True):
                        try:
                            # 1. SMILES記述子側のデータ取得
                            _p_df = st.session_state.get("precalc_smiles_df")
                            target_col_val = st.session_state.get("target_col", "")
                            
                            # 2. 元のDFから数値列（既存の説明変数）を抽出
                            # 目的変数以外の、数値型を持つカラム
                            original_numeric_cols = [
                                c for c in df.columns 
                                if c != target_col_val and pd.api.types.is_numeric_dtype(df[c])
                            ]
                            
                            # ヒートマップ用データの準備
                            heatmap_df_parts = []
                            
                            # (A) SMILES側の選択済み記述子
                            if _p_df is not None:
                                available_smiles_descs = [d for d in selected_desc if d in _p_df.columns]
                                if available_smiles_descs:
                                    smiles_part = _p_df[available_smiles_descs].copy()
                                    heatmap_df_parts.append(smiles_part)
                            
                            # (B) 元データの数値列
                            if original_numeric_cols:
                                # SMILES側のindexと合わせる（欠損行があれば除外される）
                                idx = _p_df.index if _p_df is not None else df.index
                                original_part = df.loc[idx, original_numeric_cols].copy()
                                heatmap_df_parts.append(original_part)
                                
                            # (C) 目的変数（必須）
                            if target_col_val in df.columns and pd.api.types.is_numeric_dtype(df[target_col_val]):
                                idx = _p_df.index if _p_df is not None else df.index
                                target_part = df.loc[idx, [target_col_val]].copy()
                                heatmap_df_parts.append(target_part)

                            if heatmap_df_parts:
                                heatmap_df = pd.concat(heatmap_df_parts, axis=1)
                                
                                # 相関行列計算 (Pearson)
                                corr_matrix = heatmap_df.corr(method="pearson")
                                
                                import plotly.express as px
                                fig = px.imshow(
                                    corr_matrix,
                                    text_auto=".2f",
                                    aspect="auto",
                                    color_continuous_scale="RdBu_r",
                                    zmin=-1, zmax=1,
                                    title=f"目的変数「{target_col_val}」と各変数の相互相関"
                                )
                                # 動的に高さを調整（変数が多すぎると潰れるのを防ぐ）
                                calc_height = max(500, len(corr_matrix.columns) * 35)
                                fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=calc_height)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("相関を計算できる数値変数がありません。")
                        except Exception as e:
                            st.warning(f"ヒートマップの描画中にエラーが発生しました: {e}")
                    # ------------------------------------

                    with st.expander("選択中の記述子詳細を確認", expanded=(len(selected_desc) <= 10)):
                        recs = get_all_target_recommendations()
                        desc_details = {}
                        for rec in recs:
                            for d in rec.descriptors:
                                if d.name not in desc_details:
                                    desc_details[d.name] = {"meaning": d.meaning, "targets": set(), "library": d.library}
                                desc_details[d.name]["targets"].add(rec.target_name)

                        if len(selected_desc) <= 30:
                            display_descs = selected_desc
                            hidden_count = 0
                        else:
                            display_descs = selected_desc[:30]
                            hidden_count = len(selected_desc) - 30

                        for d_name in display_descs:
                            info = desc_details.get(d_name)
                            if info:
                                tgts = "、".join(list(info["targets"]))
                                st.markdown(f"- **{d_name}** ({info['library']}): {info['meaning']}<br>&nbsp;&nbsp;<span style='color:#888;font-size:0.85em;'>適した目的変数: {tgts}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"- **{d_name}** <span style='color:#888;font-size:0.85em;'>(全般的な記述子)</span>", unsafe_allow_html=True)
                        
                        if hidden_count > 0:
                            st.markdown(f"- ...他 **{hidden_count}** 件（省略）")
                else:
                    selected_desc = st.session_state.get("adv_desc", [])

            # 詳細設定の値をセッションに保存
            st.session_state["_adv"] = dict(
                cv_folds=cv_folds, models=selected_models, timeout=timeout,
                scaler=scaler,
                do_eda=do_eda, do_prep=do_prep, do_eval=do_eval,
                do_pca=do_pca, do_shap=do_shap,
                selected_descriptors=selected_desc,
            )

        # ── 実行前のデータプレビュー＆型判定結果の確認 ────
        existing_result = st.session_state.get("pipeline_result")
        
        if existing_result is None:
            st.markdown("### 📄 解析対象データのプレビューと変数型")
            st.caption("以下のデータ設定で解析を実行します。もしSMILES列が含まれている場合、解析に使用可能な複数の数値記述子へ自動展開された後の姿が表示されます。")
            with st.expander("プレビューと変数型の自動判定結果", expanded=True):
                # ユーザーの全体データ感触を掴むため、まずは上位100行程度まで展開可能にする
                df_preview = df.head(100).copy()
                added_smiles_cols = []
                
                smiles_col_preview = st.session_state.get("smiles_col")
                if smiles_col_preview and smiles_col_preview in df_preview.columns:
                    try:
                        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
                        selected_desc_preview = st.session_state.get("adv_desc", [])
                        transformer = SmilesDescriptorTransformer(
                            smiles_col=smiles_col_preview, 
                            selected_descriptors=selected_desc_preview if selected_desc_preview else None
                        )
                        df_preview = transformer.fit_transform(df_preview)
                        # SMILES変換で新しく生じた列群を特定
                        added_smiles_cols = [c for c in df_preview.columns if c not in df.columns]
                    except Exception as e:
                        st.error(f"SMILES展開のプレビュー表示に失敗しました: {e}")
                
                # 型推論を明示的に行い、TypeDetectorがfloat64等の判定を行えるようにする
                # ここで以前のバグ（不明判定）を防ぐために数値キャストを挟む
                df_preview = df_preview.apply(pd.to_numeric, errors='ignore').convert_dtypes()

                # まずはデータフレーム全体をドカンと表示する
                if added_smiles_cols:
                    st.markdown("### 📊 データプレビュー (SMILES展開後の状態)")
                else:
                    st.markdown("### 📊 データプレビュー")
                
                # aggridは使えない環境もあるため、Streamlitネイティブの高性能なdataframe表示を活用
                st.dataframe(df_preview, use_container_width=True, height=400)
                
                st.markdown("---")
                
                # その下に型判定結果を表示
                st.markdown("### 🔍 変数型の自動判定結果 (TypeDetector)")
                dr_base = st.session_state.get("detection_result")
                target_col_preview = st.session_state.get("target_col")
                if dr_base:
                    # 生のdr_baseはSMILES展開前なので、SMILES列が含まれる場合はプレビュー用DFで再判定する
                    if added_smiles_cols:
                        from backend.data.type_detector import TypeDetector
                        tmp_dr = TypeDetector().detect(df_preview.drop(columns=[target_col_preview], errors="ignore") if target_col_preview else df_preview)
                    else:
                        tmp_dr = dr_base
                        
                    type_data = []
                    for c in df_preview.columns:
                        t = "❓不明"
                        if target_col_preview and c == target_col_preview:
                            t = "🎯 目的変数"
                        elif c in getattr(tmp_dr, "numeric_columns", []):
                            t = "🔢 数値"
                        elif c in getattr(tmp_dr, "categorical_columns", []):
                            t = "🔤 カテゴリ"
                        elif c in getattr(tmp_dr, "binary_columns", []):
                            t = "0️⃣1️⃣ ２値"
                        elif c in getattr(tmp_dr, "datetime_columns", []):
                            t = "📅 日時"
                        elif c in getattr(tmp_dr, "ignored_columns", []):
                            t = "❌ 除外（一意・定数等）"
                        type_data.append({"列名": c, "判定型": t})
                    
                    # 3列に分けて広く見せるなど工夫（またはシンプルなテーブル）
                    st.dataframe(pd.DataFrame(type_data), use_container_width=True, hide_index=True)
                else:
                    st.info("型判定結果がありません")

        # ── 実行ボタン（主役） ───────────────────────────────
        st.markdown("")

        if existing_result is None:
            c_l, c_m, c_r = st.columns([1, 3, 1])
            with c_m:
                if st.button(
                    "🚀 解析開始  （EDA → AutoML → 評価 → SHAP まで自動実行）",
                    use_container_width=True,
                    key="home_run",
                    type="primary",
                ):
                    # 詳細設定がない場合はデフォルト値を使用
                    adv = st.session_state.get("_adv", {})
                    st.session_state["_run_config"] = dict(
                        target_col = st.session_state["target_col"],
                        smiles_col = st.session_state.get("smiles_col"),
                        task       = st.session_state.get("task", "auto"),
                        cv_folds   = adv.get("cv_folds", 5),
                        models     = adv.get("models", []),
                        model_params = st.session_state.get("model_params", {}),
                        timeout    = adv.get("timeout", 300),
                        scaler     = adv.get("scaler", "auto"),
                        do_eda     = adv.get("do_eda", True),
                        do_prep    = adv.get("do_prep", True),
                        do_eval    = adv.get("do_eval", True),
                        do_pca     = adv.get("do_pca", True),
                        do_shap    = adv.get("do_shap", True),
                        selected_descriptors = adv.get("selected_descriptors", None),
                    )
                    # 解析実行タブへ移動（タブ内で実行される）
                    st.session_state["active_tab_idx"] = 1
                    st.rerun()
        else:
            # 結果サマリーをホームに表示
            ar = st.session_state.get("automl_result")
            if ar:
                st.success(
                    f"✅ 解析完了！ 最良モデル: **{ar.best_model_key}** | "
                    f"スコア: `{ar.best_score:.4f}` | "
                    f"所要時間: {existing_result.elapsed:.1f}秒"
                )

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                if st.button("📊 結果を見る", use_container_width=True, key="view_res"):
                    st.session_state["active_tab_idx"] = 2
                    st.rerun()
            with cc2:
                if st.button("🔄 別データで再解析", use_container_width=True, key="reset"):
                    for k in ["df","file_name","automl_result","pipeline_result",
                              "target_col","detection_result","step_eda_done",
                              "step_preprocess_done","_run_config"]:
                        st.session_state[k] = None if k not in (
                            "step_eda_done","step_preprocess_done") else False
                    st.rerun()
            with cc3:
                if st.button("🔧 詳細ツールへ", use_container_width=True, key="to_expert"):
                    st.session_state["page"] = "eda"
                    st.rerun()

    # ====================================================
    # TAB 2: 解析実行
    # ====================================================
    with tab2:
        if not has_data:
            st.warning("⚠️ まず「📂 ① データ設定」タブでデータをアップロードしてください。")
        else:
            from frontend_streamlit.pages import automl_page
            rc = st.session_state.pop("_run_config", None)
            if rc is not None:
                automl_page.render(run_config=rc)
            else:
                automl_page.render()

    # ====================================================
    # TAB 3: 結果確認
    # ====================================================
    with tab3:
        if not has_result:
            st.info("⏳ 「🚀 ② 解析実行」タブで解析を実行すると、結果がここに表示されます。")
        else:
            # 設定変更・再解析ボタン
            c_re1, c_re2 = st.columns([2, 1])
            with c_re2:
                if st.button("� 設定を変えて再解析", key="tab3_rerun", use_container_width=True):
                    st.session_state["automl_result"] = None
                    st.rerun()
            with c_re1:
                ar = st.session_state.get("automl_result")
                if ar:
                    st.success(
                        f"✅ 最良モデル: **{ar.best_model_key}** | "
                        f"スコア: `{ar.best_score:.4f}` | "
                        f"タスク: {ar.task}"
                    )

            st.markdown("---")
            st.caption("💡 SHAP解釈・次元削減・化合物解析は左サイドバー「詳細ツール」からも利用できます。")

            # モデル評価を直接表示
            try:
                from frontend_streamlit.pages.pipeline import evaluation_page
                evaluation_page.render()
            except Exception as e:
                st.error(f"❌ 評価ページの読み込みエラー: {e}")
                import traceback
                st.code(traceback.format_exc())

