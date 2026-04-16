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
        "results": None,  # 互換性用
        "col_role_exclude": [],
        "col_role_info": [],
        "col_role_weight": None,
        "col_role_group": None,
        "col_role_time": None,
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
    has_result = st.session_state.get("automl_result") is not None
    has_pl     = st.session_state.get("pipeline_result") is not None

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
    if has_result and st.button("📊 結果確認（詳細）", use_container_width=True, key="sb_tab2"):
        st.session_state["page"] = "evaluation"
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

    # ── 解析開始ボタン（タブの外・常時表示） ──────────────────
    _df_btn = st.session_state.get("df")
    if _df_btn is not None:
        existing_result = st.session_state.get("pipeline_result")
        if existing_result is None:
            c_l, c_m, c_r = st.columns([1, 3, 1])
            with c_m:
                _run_clicked = st.button(
                    "🚀 解析開始  （EDA → AutoML → 評価 → SHAP まで自動実行）",
                    use_container_width=True,
                    key="home_run",
                    type="primary",
                )
        else:
            _run_clicked = False
            ar = st.session_state.get("automl_result")
            if ar:
                st.success(
                    f"✅ 解析完了！ 最良モデル: **{ar.best_model_key}** | "
                    f"スコア: `{ar.best_score:.4f}` | "
                    f"所要時間: {existing_result.elapsed:.1f}秒"
                )
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("📊 結果を確認する", use_container_width=True, key="view_res"):
                    st.session_state["page"] = "evaluation"
                    st.rerun()
            with cc2:
                if st.button("🔄 データを変えてやり直す", use_container_width=True, key="reset"):
                    for k in ["df","file_name","automl_result","pipeline_result",
                              "target_col","detection_result","step_eda_done",
                              "step_preprocess_done","_run_config"]:
                        st.session_state[k] = None if k not in ("step_eda_done","step_preprocess_done") else False
                    st.rerun()

        # ボタンが押された場合にrun_configを構築
        if _run_clicked:
            adv = st.session_state.get("_adv", {})
            _target_lk = st.session_state.get("target_col")
            _smiles_lk = st.session_state.get("smiles_col")
            _excl_lk = st.session_state.get("col_role_exclude", [])
            _feat_lk = [c for c in _df_btn.columns if c not in [_target_lk, _smiles_lk] + _excl_lk]
            _X_lk = _df_btn[_feat_lk].select_dtypes(include="number")
            _leakage_group_labels = None
            _leakage_recommended_cv = None
            _n_cv_folds = adv.get("cv_folds", 5)
            # 少量データ（50件未満）ではリーケージ検出の偽陽性率が高いためスキップ
            if _X_lk.shape[1] >= 2 and len(_df_btn) >= 50 and len(_df_btn) <= 5000:
                try:
                    from backend.data.leakage_detector import detect_leakage
                    _lk_report = detect_leakage(_X_lk, _df_btn[_target_lk], method="auto")
                    if _lk_report.risk_level in ("medium", "high") and _lk_report.group_labels is not None:
                        # GroupKFold適用には、グループ数 >= n_splits が必須
                        _n_groups = _lk_report.n_groups
                        if _n_groups >= _n_cv_folds:
                            _leakage_group_labels = _lk_report.group_labels
                            _leakage_recommended_cv = _lk_report.recommended_cv
                        else:
                            import logging as _logging
                            _logging.getLogger(__name__).info(
                                f"リーケージ検出: グループ数({_n_groups}) < CV folds({_n_cv_folds})のため"
                                f"GroupKFold適用をスキップ。通常のKFoldを使用します。"
                            )
                    st.session_state["leakage_report"] = _lk_report
                except Exception:
                    pass
            st.session_state["_run_config"] = dict(
                target_col=st.session_state["target_col"],
                smiles_col=st.session_state.get("smiles_col"),
                task=st.session_state.get("task", "auto"),
                cv_folds=adv.get("cv_folds", 5),
                models=adv.get("models", []),
                model_params=st.session_state.get("model_params", {}),
                timeout=adv.get("timeout", 300),
                scaler=adv.get("scaler", "auto"),
                do_eda=adv.get("do_eda", True),
                do_prep=adv.get("do_prep", True),
                do_eval=adv.get("do_eval", True),
                do_pca=adv.get("do_pca", True),
                do_shap=adv.get("do_shap", True),
                selected_descriptors=adv.get("selected_descriptors", None),
                monotonic_constraints_dict=st.session_state.get("_monotonic_constraints_dict", {}),
                leakage_group_labels=_leakage_group_labels,
                leakage_recommended_cv=_leakage_recommended_cv,
                cv_groups_col=st.session_state.get("col_role_group"),
                exclude_cols=list(set(st.session_state.get("col_role_exclude", []) + st.session_state.get("col_role_info", []))),
                col_role_time=st.session_state.get("col_role_time"),
                sample_weight_col=st.session_state.get("col_role_weight"),
            )
            st.rerun()
    else:
        _run_clicked = False

    # ── 解析実行中（_run_configがあれば即座に実行） ───────────
    _rc_pending = st.session_state.pop("_run_config", None)
    if _rc_pending is not None:
        from frontend_streamlit.pages import automl_page
        automl_page.render(run_config=_rc_pending)
        st.stop()

    # ── タブラベル ────────────────────────────────────────────
    tab_labels = [
        "📂 ① データ設定",
        "📊 ② 結果確認",
    ]
    tab1, tab2 = st.tabs(tab_labels)

    # ──────────────────────────────────────────────────────────
    # TAB 1: データ設定（アップロード + 目的変数 + 詳細設定）
    # ──────────────────────────────────────────────────────────
    with tab1:

        from backend.data.loader import load_from_bytes, get_supported_extensions
        from backend.data.type_detector import TypeDetector

        # ── データ設定 5サブタブ ──────────────────────────────────────
        ds_tab1, ds_tab2, ds_tab3, ds_tab4, ds_tab5 = st.tabs([
            "📂 データ読込",
            "🏷️ 列の役割設定",
            "⚗️ SMILES特徴量設計",
            "📊 特徴量EDA",
            "⚙️ パイプライン設計",
        ])

        # ══════════════════════════════════════════════════════════════
        # サブタブ1: データ読込
        # ══════════════════════════════════════════════════════════════
        with ds_tab1:
            ext_list = get_supported_extensions()
            uploaded = st.file_uploader(
                "📂 分析したいデータファイルをドロップ",
                type=[e.lstrip(".") for e in ext_list],
                help=f"対応形式: {', '.join(ext_list)}",
                label_visibility="visible",
            )

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
                    
                    # ── 基本サンプルの生成 ──────────────────
                    st.markdown("##### 📦 基本サンプル (動的生成)")
                    use_smiles = st.checkbox("SMILES（化合物構造）列を含める", value=True, key="demo_smiles")
                    c_r, c_c = st.columns(2)
                    _DUMMY_SMILES = ["C", "CC", "CCC", "CCO", "CCN", "c1ccccc1", "c1ccccc1O", "CC(=O)O", "CC(C)C", "C1CCCCC1", "c1ccncc1", "c1ncncn1", "C1COCCO1"]
                    with c_r:
                        if st.button("🧪 回帰サンプル (25件)", use_container_width=True, key="demo_reg"):
                            np.random.seed(42); n = 25
                            if use_smiles:
                                base_df = pd.DataFrame({"SMILES": np.random.choice(_DUMMY_SMILES, n), "solubility_logS": np.random.randn(n) * 2 - 2})
                            else:
                                base_df = pd.DataFrame({
                                    "temperature": np.random.uniform(20, 80, n),
                                    "pressure":    np.random.exponential(5, n),
                                    "catalyst":    np.random.choice(["A型","B型","C型"], n),
                                    "time_h":      np.random.uniform(1, 24, n),
                                    "yield":       np.random.randn(n) * 10 + 75,
                                })
                            _make_sample("sample_regression.csv", base_df, set_smiles=use_smiles)
                            st.session_state["task"] = "regression"
                            st.rerun()
                    with c_c:
                        if st.button("🏷️ 分類サンプル (25件)", use_container_width=True, key="demo_cls"):
                            np.random.seed(42); n = 25
                            if use_smiles:
                                base_df = pd.DataFrame({"SMILES": np.random.choice(_DUMMY_SMILES, n), "is_toxic": np.random.randint(0, 2, n)})
                            else:
                                base_df = pd.DataFrame({
                                    "feature_1": np.random.randn(n),
                                    "category":  np.random.choice(["低","中","高"], n),
                                    "label":     np.random.randint(0, 2, n),
                                })
                            _make_sample("sample_classification.csv", base_df, set_smiles=use_smiles)
                            st.session_state["task"] = "classification"
                            st.rerun()

                    # ── 特殊デバッグ用（ファイルベース） ────
                    st.markdown("---")
                    st.markdown("##### 🧪 理論・デバッグ用セット (固定データ)")
                    
                    DEBUG_SAMPLES = {
                        "mixture_smiles_only": {
                            "name": "🧪 混合物回帰 (SMILESのみ)",
                            "file": "mixture_smiles_only.csv",
                            "desc": "化合物SMILESとその割合のみの混合物データ（50行）。",
                            "target": "Target_BoilingPoint_C",
                            "task": "regression"
                        },
                        "mixture_smiles_numeric": {
                            "name": "🧪 混合物回帰 (SMILES + 数値)",
                            "file": "mixture_smiles_numeric.csv",
                            "desc": "SMILES, WT%, および外部条件（温度・pH等）を含む混合物（50行）。",
                            "target": "Target_Yield_pct",
                            "task": "regression"
                        },
                        "mixture_regression_debug": {
                            "name": "🧪 混合物回帰 (汎用)",
                            "file": "mixture_regression_debug.csv",
                            "desc": "加重平均変換と各記述子エンジンの結合テスト用（50行）。",
                            "target": "Boiling_Point_C",
                            "task": "regression"
                        },
                        "monotonicity_test": {
                            "name": "📈 単調性制約テスト",
                            "file": "monotonicity_test.csv",
                            "desc": "分子量・LogP等と溶解度の関係。制約適用の検証用。",
                            "target": "Solubility_mg_L",
                            "task": "regression"
                        },
                    }

                    for skey, sinfo in DEBUG_SAMPLES.items():
                        c_left, c_right = st.columns([0.4, 0.6])
                        with c_left:
                            if st.button(sinfo["name"], key=f"btn_{skey}", use_container_width=True):
                                try:
                                    csv_path = Path("data/samples/debug") / sinfo["file"]
                                    if not csv_path.exists():
                                        # scripts/ を上位ディレクトリから探索
                                        csv_path = Path(__file__).parent.parent / "data" / "samples" / "debug" / sinfo["file"]
                                    
                                    df_dbg = pd.read_csv(csv_path)
                                    _make_sample(sinfo["file"], df_dbg)
                                    st.session_state["target_col"] = sinfo["target"]
                                    st.session_state["task"] = sinfo["task"]
                                    st.success(f"Loaded: {sinfo['name']}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error loading {sinfo['file']}: {e}")
                        with c_right:
                            st.caption(sinfo["desc"])

                with st.expander("📚 オープンベンチマークデータをロード"):
                    st.markdown("ケモインフォマティクスの評価でよく使われる公開データセットです。")
                    c_e, c_f, c_l = st.columns(3)
                    with c_e:
                        st.markdown("**ESOL** (水溶解度)")
                        st.caption("1,128化合物の実測logS。")
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

            # ── ファイル読み込み処理 ──────────────────────────────────
            if uploaded is not None:
                try:
                    with st.spinner("読み込み中..."):
                        raw = uploaded.read()
                        df_new = load_from_bytes(raw, uploaded.name)
                    st.success(f"✅ `{uploaded.name}` 読み込み完了")
                    st.session_state["df"]             = df_new
                    st.session_state["file_name"]      = uploaded.name
                    st.session_state["automl_result"]  = None
                    st.session_state["pipeline_result"] = None
                    st.session_state["smiles_col"]     = None
                    detector = TypeDetector()
                    dr = detector.detect(df_new)
                    st.session_state["detection_result"] = dr
                    if dr.smiles_columns:
                        st.session_state["smiles_col"] = dr.smiles_columns[0]
                    else:
                        for col in df_new.columns:
                            if col.lower() == "smiles":
                                st.session_state["smiles_col"] = col
                                break
                    st.session_state["target_col"] = df_new.columns[-1]
                except Exception as e:
                    st.error(f"❌ 読み込みエラー: {e}")

            # 読み込み完了後のメッセージ
            df = st.session_state.get("df")
            if df is not None:
                st.markdown("---")
                fn = st.session_state.get("file_name", "")
                c1, c2, c3, c4 = st.columns(4)
                for col_w, val, lbl in [
                    (c1, f"{df.shape[0]:,}", "行数"),
                    (c2, str(df.shape[1]), "列数"),
                    (c3, f"{df.isna().mean().mean():.1%}", "欠損率"),
                    (c4, str(df.select_dtypes(include='number').shape[1]), "数値列数"),
                ]:
                    with col_w:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-value" style="font-size:1.4rem;">{val}</div>'
                            f'<div class="metric-label">{lbl}</div></div>',
                            unsafe_allow_html=True,
                        )
                st.success(f"✅ `{fn}` が読み込まれています。次は「🏷️ 列の役割設定」タブへ進んでください。")

        # ══════════════════════════════════════════════════════════════
        # サブタブ2: 列の役割設定
        # ══════════════════════════════════════════════════════════════
        with ds_tab2:
            df = st.session_state.get("df")
            if df is None:
                st.warning("⚠️ まず「📂 データ読込」タブでデータを読み込んでください。")
            else:
                all_cols = df.columns.tolist()
                none_opt = ["（なし）"]

                st.markdown("### 🏷️ 各列の役割を設定してください")
                st.caption("🎯 目的変数と説明変数は必須です。それ以外は任意です。")

                col_a, col_b = st.columns(2)

                with col_a:
                    # 目的変数（必須）
                    cur_target = st.session_state.get("target_col") or all_cols[-1]
                    cur_idx = all_cols.index(cur_target) if cur_target in all_cols else len(all_cols) - 1
                    target = st.selectbox(
                        "🎯 目的変数（必須・予測したい列）",
                        options=all_cols,
                        index=cur_idx,
                        key="home_target",
                        help="機械学習で予測する列を選んでください",
                    )
                    if st.session_state.get("target_col_prev") != target:
                        st.session_state["precalc_done"] = False
                        st.session_state["target_col_prev"] = target
                    st.session_state["target_col"] = target

                    # 目的変数の推奨ヒント
                    from backend.chem.recommender import get_target_recommendation_by_name
                    rec = get_target_recommendation_by_name(target)
                    if rec:
                        st.info(f"💡 **推奨される記述子**: {rec.summary}")

                    # タスク種別（自動判定 + 変更は折りたたみ）
                    _auto_task = "regression" if pd.api.types.is_float_dtype(df[target]) else "classification"
                    _task_label = "📈 回帰（数値予測）" if _auto_task == "regression" else "🏷️ 分類（カテゴリ予測）"
                    st.markdown(f"**📋 タスク種別**: {_task_label}（自動判定）")
                    _task_override = st.pills(
                        "変更する場合",
                        ["auto（自動）", "regression（回帰）", "classification（分類）"],
                        default="auto（自動）",
                        key="home_task",
                    )
                    st.session_state["task"] = _task_override.split("（")[0] if _task_override else "auto"

                    # SMILES列（自動検出 + 確認）
                    _det_smiles = st.session_state.get("smiles_col")
                    if _det_smiles and _det_smiles in all_cols:
                        st.markdown(f"**SMILES列**: `{_det_smiles}`")
                        _change_smiles = st.checkbox("SMILES列を変更する", key="chg_sm", value=False)
                        if _change_smiles:
                            smiles_options = none_opt + all_cols
                            smiles_raw = st.selectbox(
                                "SMILES列の変更",
                                smiles_options,
                                index=smiles_options.index(_det_smiles) if _det_smiles in smiles_options else 0,
                                key="adv_sm",
                            )
                            new_smiles_col = None if smiles_raw == "（なし）" else smiles_raw
                        else:
                            new_smiles_col = _det_smiles
                    else:
                        smiles_options = none_opt + all_cols
                        cur_smiles = st.session_state.get("smiles_col")
                        smiles_idx = (smiles_options.index(cur_smiles) if cur_smiles in smiles_options else
                                      next((i+1 for i, c in enumerate(all_cols) if c.lower()=="smiles"), 0))
                        smiles_raw = st.selectbox(
                            "🧬 SMILES列（任意・化合物構造列）",
                            smiles_options,
                            index=smiles_idx,
                            key="adv_sm",
                            help="SMILES形式の化合物構造が含まれる列。指定すると化学記述子を自動計算します。",
                        )
                        new_smiles_col = None if smiles_raw == "（なし）" else smiles_raw
                    if st.session_state.get("smiles_col") != new_smiles_col:
                        st.session_state["precalc_done"] = False
                        st.session_state["precalc_smiles_df"] = None
                    st.session_state["smiles_col"] = new_smiles_col

                with col_b:
                    excl_opts = [c for c in all_cols if c != target and c != new_smiles_col]

                    # 除外する列（任意）
                    cur_excl = [c for c in st.session_state.get("col_role_exclude", []) if c in excl_opts]
                    col_role_exclude = st.multiselect(
                        "🚫 解析から除外する列（任意）",
                        options=excl_opts,
                        default=cur_excl,
                        key="col_role_exclude",
                        help="目的変数・SMILES以外で解析に使わない列（ID列・メモ列等）",
                    )

                    # グループ列（GroupKFold用）
                    group_opts = none_opt + [c for c in excl_opts if c not in col_role_exclude]
                    cur_group = st.session_state.get("col_role_group", "（なし）")
                    if cur_group not in group_opts:
                        cur_group = "（なし）"
                    col_role_group = st.selectbox(
                        "👥 グループ列（任意・GroupKFold用）",
                        options=group_opts,
                        index=group_opts.index(cur_group),
                        key="col_role_group_sel",
                        help="GroupKFold等でリーク防止に使うグループID列（例:バッチID・実験ロット）",
                    )
                    st.session_state["col_role_group"] = None if col_role_group == "（なし）" else col_role_group

                    # 時系列列（任意）
                    time_opts = none_opt + [c for c in excl_opts if c not in col_role_exclude]
                    cur_time = st.session_state.get("col_role_time", "（なし）")
                    if cur_time not in time_opts:
                        cur_time = "（なし）"
                    col_role_time = st.selectbox(
                        "📅 時系列列（任意・時間的順序）",
                        options=time_opts,
                        index=time_opts.index(cur_time),
                        key="col_role_time_sel",
                        help="時間的順序を持つ列（例: 日付・ステップ番号）。TimeSeriesSplit等に使用します。",
                    )
                    st.session_state["col_role_time"] = None if col_role_time == "（なし）" else col_role_time

                    # sample_weight列（任意）
                    weight_opts = none_opt + [c for c in all_cols if c not in [target, new_smiles_col] + col_role_exclude]
                    cur_weight = st.session_state.get("col_role_weight", "（なし）")
                    if cur_weight not in weight_opts:
                        cur_weight = "（なし）"
                    col_role_weight = st.selectbox(
                        "⚖️ Sample weight列（任意・サンプル重み）",
                        options=weight_opts,
                        index=weight_opts.index(cur_weight),
                        key="col_role_weight_sel",
                        help="各サンプルの重みを示す列。信頼度の高いサンプルを重視する場合に指定します。",
                    )
                    st.session_state["col_role_weight"] = None if col_role_weight == "（なし）" else col_role_weight

                    # Info列（解析には使わないがレポートに残す）
                    info_opts = [c for c in all_cols if c not in [target, new_smiles_col] + col_role_exclude]
                    cur_info = [c for c in st.session_state.get("col_role_info", []) if c in info_opts]
                    col_role_info = st.multiselect(
                        "ℹ️ 情報管理列（任意・Info列）",
                        options=info_opts,
                        default=cur_info,
                        key="col_role_info",
                        help="解析には使わないが結果レポートに残す列（例: 化合物名・実験者名）",
                    )

                # 列役割サマリー
                st.markdown("---")
                st.markdown("#### 📋 現在の列役割サマリー")
                role_rows = []
                for c in all_cols:
                    if c == target:
                        role = "🎯 目的変数"
                    elif c == new_smiles_col:
                        role = "🧬 SMILES"
                    elif c in col_role_exclude:
                        role = "🚫 除外"
                    elif c == st.session_state.get("col_role_group"):
                        role = "👥 グループ"
                    elif c == st.session_state.get("col_role_time"):
                        role = "📅 時系列"
                    elif c == st.session_state.get("col_role_weight"):
                        role = "⚖️ weight"
                    elif c in col_role_info:
                        role = "ℹ️ Info"
                    else:
                        role = "✅ 説明変数"
                    role_rows.append({"列名": c, "役割": role})
                st.dataframe(pd.DataFrame(role_rows), use_container_width=True, hide_index=True, height=min(400, 40 + len(role_rows) * 35))

        # ══════════════════════════════════════════════════════════════
        # サブタブ4: 特徴量EDA
        # ══════════════════════════════════════════════════════════════
        with ds_tab3:
            df = st.session_state.get("df")
            if df is None:
                st.warning("⚠️ まず「📂 データ読込」タブでデータを読み込んでください。")
            elif not st.session_state.get("smiles_col"):
                st.info("🧬 SMILES列が設定されていません。「🏷️ 列の役割設定」タブでSMILES列を指定してください。")
            else:
                smiles_col_sf = st.session_state["smiles_col"]
                target_col_sf = st.session_state.get("target_col", "")

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # 全記述子の自動計算（ボタンなし・全エンジン自動）
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                _PRECALC_VERSION = 5  # 列数チェック追加
                _stored_ver = st.session_state.get("_precalc_version", 0)
                if _stored_ver != _PRECALC_VERSION:
                    st.session_state["precalc_done"] = False
                    st.session_state["precalc_smiles_df"] = None
                    st.session_state["_precalc_version"] = _PRECALC_VERSION
                    st.session_state["_desc_sets"] = None  # 古いセットもリセット

                # 列数が少なすぎる場合も強制再計算（RDKit 222列なのに32列は異常）
                _existing_df = st.session_state.get("precalc_smiles_df")
                if _existing_df is not None and len(_existing_df.columns) < 50:
                    st.session_state["precalc_done"] = False
                    st.session_state["precalc_smiles_df"] = None

                if not st.session_state.get("precalc_done", False):
                    from backend.chem.recommender import get_target_recommendation_by_name as _get_rec

                    smiles_series = df[smiles_col_sf]
                    valid_mask = smiles_series.notna()
                    smiles_list = smiles_series[valid_mask].tolist()
                    valid_idx = smiles_series[valid_mask].index
                    n = len(smiles_list)

                    st.info(f"⚙️ **{n} 件のSMILESに対し、利用可能な全エンジンで記述子を自動計算中...**")

                    df_result = pd.DataFrame(index=range(n))
                    _calc_summary = {}  # エンジン別の計算結果サマリ

                    # ── 全エンジン定義（計算順: 高速→中程度→重い） ──
                    _ALL_ENGINES: list[tuple[str, str, str, dict]] = [
                        # (表示名, モジュールパス, クラス名, コンストラクタ引数)
                        ("RDKit",          "backend.chem.rdkit_adapter",          "RDKitAdapter",          {"compute_fp": False}),
                        ("Mordred",        "backend.chem.mordred_adapter",        "MordredAdapter",        {"selected_only": True}),
                        ("GroupContrib",   "backend.chem.group_contrib_adapter",  "GroupContribAdapter",   {}),
                        ("DescriptaStorus","backend.chem.descriptastorus_adapter","DescriptaStorusAdapter",{}),
                        ("MolAI",          "backend.chem.molai_adapter",          "MolAIAdapter",          {"n_components": st.session_state.get("molai_n_components", 6)}),
                        ("scikit-FP",      "backend.chem.skfp_adapter",           "SkfpAdapter",           {"fp_types": ["ECFP", "MACCS"]}),
                        ("UMA",            "backend.chem.uma_adapter",            "UMAAdapter",            {}),
                        ("Mol2Vec",        "backend.chem.mol2vec_adapter",        "Mol2VecAdapter",        {}),
                        ("PaDEL",          "backend.chem.padel_adapter",          "PaDELAdapter",          {}),
                        ("Molfeat",        "backend.chem.molfeat_adapter",        "MolfeatAdapter",        {}),
                        ("XTB",            "backend.chem.xtb_adapter",            "XTBAdapter",            {}),
                        ("UniPKa",         "backend.chem.unipka_adapter",         "UniPkaAdapter",         {}),
                        ("COSMO-RS",       "backend.chem.cosmo_adapter",          "CosmoAdapter",          {}),
                        ("Chemprop",       "backend.chem.chemprop_adapter",       "ChempropAdapter",       {}),
                    ]

                    _progress_bar = st.progress(0, text="記述子計算を開始中...")
                    _total_engines = len(_ALL_ENGINES)

                    for _ei, (_ename, _emod, _ecls, _ekwargs) in enumerate(_ALL_ENGINES):
                        _progress_bar.progress(
                            (_ei + 1) / _total_engines,
                            text=f"{_ename} を計算中... ({_ei + 1}/{_total_engines})"
                        )
                        try:
                            _mod = __import__(_emod, fromlist=[_ecls])
                            _adapter = getattr(_mod, _ecls)(**_ekwargs)
                            if not _adapter.is_available():
                                continue
                            _eres = _adapter.compute(smiles_list)
                            if hasattr(_eres, 'descriptors') and _eres.descriptors is not None:
                                _edf = _eres.descriptors
                                _new_cols = [c for c in _edf.columns if c not in df_result.columns]
                                if _new_cols:
                                    _edf_new = _edf[_new_cols].copy()
                                    _edf_new.index = range(n)
                                    df_result = pd.concat([df_result, _edf_new], axis=1)
                                    _calc_summary[_ename] = len(_new_cols)

                            # MolAI PCA寄与率の保存
                            if _ename == "MolAI" and hasattr(_adapter, '_pca') and _adapter._pca is not None:
                                _evr = _adapter._pca.explained_variance_ratio_
                                st.session_state["molai_explained_variance"] = {
                                    "ratio": _evr.tolist(), "cumulative": _evr.cumsum().tolist(),
                                    "n_components": _ekwargs.get("n_components", 6),
                                }
                        except Exception as e:
                            import logging as _log
                            _log.getLogger(__name__).warning(f"{_ename} スキップ: {e}")

                    _progress_bar.empty()

                    # クリーンアップ
                    df_result = df_result.loc[:, ~df_result.columns.duplicated()]
                    df_result.index = valid_idx
                    df_result = df_result.apply(pd.to_numeric, errors="coerce").convert_dtypes()
                    _n_before_drop = len(df_result.columns)
                    df_result = df_result.dropna(axis=1, how="all")
                    _n_dropped = _n_before_drop - len(df_result.columns)

                    # MolAI列のサマリ追加
                    _molai_cols = [c for c in df_result.columns if c.startswith("molai_") or c.startswith("MolAI_")]
                    if _molai_cols and "MolAI" not in _calc_summary:
                        _calc_summary["MolAI"] = len(_molai_cols)

                    # 計算結果サマリ表示
                    _summary_parts = [f"{eng}: {cnt}個" for eng, cnt in _calc_summary.items()]
                    st.success(
                        f"✅ **{len(df_result.columns)}個**の記述子を計算完了"
                        f"（{', '.join(_summary_parts)}）"
                        + (f" — {_n_dropped}列をNaNにより除外" if _n_dropped > 0 else "")
                    )

                    # 推奨記述子を初期選択に設定
                    rec = _get_rec(target_col_sf)
                    if rec:
                        _init = [d.name for d in rec.descriptors if d.name in df_result.columns]
                    else:
                        _init = [d for d in [
                            "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
                            "NumRotatableBonds", "RingCount", "NumAromaticRings",
                            "FractionCSP3", "HeavyAtomCount", "MolMR", "HallKierAlpha",
                            "BertzCT", "MaxPartialCharge", "MinPartialCharge",
                            "LabuteASA", "qed", "BalabanJ",
                        ] if d in df_result.columns]
                    st.session_state["adv_desc"] = _init
                    st.session_state["precalc_smiles_df"] = df_result
                    st.session_state["precalc_done"] = True
                    st.rerun()

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # 計算完了 → ユーザーはテーブルでチェックするだけ
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                _precalc_df = st.session_state.get("precalc_smiles_df")
                if _precalc_df is not None:
                    from backend.chem.recommender import (
                        get_target_recommendation_by_name as _get_rec2,
                        get_all_target_recommendations as _get_all_recs,
                    )

                    _cur_sel = set(st.session_state.get("adv_desc") or [])
                    n_total = len(_precalc_df.columns)
                    n_sel = len(_cur_sel)

                    st.markdown(
                        f"### ⚗️ 記述子の選択　"
                        f"<span style='color:#4ade80; font-size:0.85em'>全{n_total}個計算済</span>　"
                        f"<span style='color:#60a5fa; font-size:0.85em'>{n_sel}個選択中</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption("全エンジンで記述子は自動計算済みです。テーブルの✅で選択してください。")

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # 事前計算：相関・メタデータ・エンジンマップ（全パネル共通）
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    _corr = {}
                    if target_col_sf in df.columns and pd.api.types.is_numeric_dtype(df[target_col_sf]):
                        try:
                            _corr = _precalc_df.corrwith(df[target_col_sf].loc[_precalc_df.index], method="pearson").abs().to_dict()
                        except Exception:
                            pass

                    _rec_names = set(d.name for d in rec.descriptors) if rec else set()

                    # 全アダプター定義（メタデータ取得・エンジンマップ共通）
                    _ENGINE_MAP_DEFS = [
                        ("RDKit",          "backend.chem.rdkit_adapter",           "RDKitAdapter",          {"compute_fp": False}),
                        ("Mordred",        "backend.chem.mordred_adapter",         "MordredAdapter",        {"selected_only": True}),
                        ("GroupContrib",   "backend.chem.group_contrib_adapter",   "GroupContribAdapter",   {}),
                        ("DescriptaStorus","backend.chem.descriptastorus_adapter", "DescriptaStorusAdapter",{}),
                        ("scikit-FP",      "backend.chem.skfp_adapter",            "SkfpAdapter",           {"fp_types": ["ECFP", "MACCS"]}),
                        ("UMA",            "backend.chem.uma_adapter",             "UMAAdapter",            {}),
                        ("Mol2Vec",        "backend.chem.mol2vec_adapter",         "Mol2VecAdapter",        {}),
                        ("PaDEL",          "backend.chem.padel_adapter",           "PaDELAdapter",          {}),
                        ("Molfeat",        "backend.chem.molfeat_adapter",         "MolfeatAdapter",        {}),
                        ("XTB",            "backend.chem.xtb_adapter",             "XTBAdapter",            {}),
                        ("COSMO-RS",       "backend.chem.cosmo_adapter",           "CosmoAdapter",          {}),
                        ("UniPKa",         "backend.chem.unipka_adapter",          "UniPkaAdapter",         {}),
                        ("Chemprop",       "backend.chem.chemprop_adapter",        "ChempropAdapter",       {}),
                    ]

                    _meta = {}
                    # 全アダプターからメタデータ（化学的意味）を取得
                    for _mm_name, _mm_mod, _mm_cls, _mm_kw in _ENGINE_MAP_DEFS:
                        try:
                            _mm_mod_obj = __import__(_mm_mod, fromlist=[_mm_cls])
                            _mm_adp = getattr(_mm_mod_obj, _mm_cls)(**_mm_kw)
                            if _mm_adp.is_available() and hasattr(_mm_adp, 'get_descriptors_metadata'):
                                for _mm in _mm_adp.get_descriptors_metadata():
                                    if _mm.name not in _meta:
                                        _meta[_mm.name] = {
                                            "meaning": getattr(_mm, 'meaning', '') or '',
                                            "cat": getattr(_mm, 'category', '') or '',
                                        }
                        except Exception:
                            pass
                    # Mordred記述子のフォールバック辞書（importキャッシュに依存しない確実な意味付与）
                    _MORDRED_MEANINGS = {
                        "MW": "分子量 (Da)。沸点・粘度・蒸気圧に直結する基本物性",
                        "nHeavyAtom": "水素以外の原子数。分子骨格の大きさの指標",
                        "nAtom": "水素を含む全原子数",
                        "nBonds": "全化学結合の数",
                        "nBondsO": "酸素原子が関与する結合の数。エーテル・エステル等の極性結合",
                        "nBondsS": "硫黄原子が関与する結合の数。チオール・スルホン等",
                        "nRing": "環構造の総数。剛直性・安定性に寄与",
                        "nHRing": "ヘテロ原子(N,O,S等)を含む環の数。生理活性・反応性に重要",
                        "nARing": "芳香環の数。π電子共役による光吸収・熱安定性",
                        "nBRing": "ベンゼン環(炭素のみの六員芳香環)の数",
                        "nFARing": "縮環構造中の芳香環の数。ナフタレン・アントラセン等",
                        "nFHRing": "縮環構造中のヘテロ環の数",
                        "nFRing": "縮環(融合環)の数。二環以上が辺を共有する構造",
                        "nSpiro": "スピロ環の数。2つの環が1原子を共有する特殊構造",
                        "nBridgehead": "橋頭位原子の数。二環式以上の架橋構造",
                        "nC": "炭素原子数。有機化合物の骨格元素",
                        "nN": "窒素原子数。アミン・アミド・ヘテロ環の存在",
                        "nO": "酸素原子数。ヒドロキシル・カルボニル・エーテルの存在",
                        "nS": "硫黄原子数。チオール・スルフィド・スルホンの存在",
                        "nF": "フッ素原子数。高い電気陰性度による極性・代謝安定性",
                        "nCl": "塩素原子数。疎水性の増大・生理活性への影響",
                        "nBr": "臭素原子数。重い置換基・反応性ハロゲン",
                        "nI": "ヨウ素原子数。最も重いハロゲン・造影剤に利用",
                        "nHet": "ヘテロ原子(C,H以外)の数。極性・反応性の指標",
                        "nHetero": "ヘテロ原子数(別定義)。N,O,S等を含む",
                        "nHBAcc": "水素結合受容体の数。O,N等の孤立電子対を持つ原子",
                        "nHBDon": "水素結合供与体の数。-OH,-NH等",
                        "nHBAcc_Lipin": "Lipinski定義の水素結合受容体数(N+O)",
                        "nHBDon_Lipin": "Lipinski定義の水素結合供与体数(NH+OH)",
                        "nRotB": "回転可能結合数。分子の柔軟性を反映",
                        "RotRatio": "回転可能結合比率。全結合中の回転可能結合の割合",
                        "TPSA": "位相的極性表面積 (Å²)。N,O由来の極性表面。水溶性・膜透過性の指標",
                        "LogP": "LogP (Wildman-Crippen法)。油/水分配係数の対数。疎水性の基本指標",
                        "SLogP": "SLogP (Wildman-Crippen法)。原子ベースLogP推定値",
                        "LabuteASA": "Labute近似溶媒接触表面積。溶媒との相互作用面積",
                        "BertzCT": "Bertz複雑度。分子グラフの構造的複雑さ。分岐・環が多いほど高い",
                        "TopoPSA": "位相的極性表面積(2D版)。3D座標不要のPSA推定",
                        "WPath": "Wiener Path Number。分子グラフの全頂点対間距離の合計。分子サイズの指標",
                        "WPol": "Wiener Polarity Number。距離3の頂点対の数。分岐度を反映",
                        "Lop": "Lopping Index。分子グラフの対称性を反映する指標",
                        "PEOE_VSA1": "PEOE部分電荷ビン1の表面積。最も負に帯電した原子の表面積",
                        "PEOE_VSA2": "PEOE部分電荷ビン2の表面積。負電荷領域",
                        "PEOE_VSA3": "PEOE部分電荷ビン3の表面積。やや負の領域",
                        "PEOE_VSA4": "PEOE部分電荷ビン4の表面積。中性付近の領域",
                        "PEOE_VSA5": "PEOE部分電荷ビン5の表面積。やや正の領域",
                        "PEOE_VSA6": "PEOE部分電荷ビン6の表面積。正電荷領域",
                        "SMR_VSA1": "屈折率ビン1の表面積。最も分極しにくい原子の表面積",
                        "SMR_VSA2": "屈折率ビン2の表面積。低分極率領域",
                        "SMR_VSA3": "屈折率ビン3の表面積。高分極率領域",
                        "SlogP_VSA1": "LogPビン1の表面積。最も親水的な原子の表面積",
                        "SlogP_VSA2": "LogPビン2の表面積。親的な領域",
                        "SlogP_VSA3": "LogPビン3の表面積。疎的な領域",
                        "Kier1": "Kier κ1形状指数。分子の直線性を反映。直鎖に近いほど大きい",
                        "Kier2": "Kier κ2形状指数。分子の分岐度を反映。分岐が多いほど大きい",
                        "Kier3": "Kier κ3形状指数。分子の空間的広がりを反映",
                        "KierFlex": "Kier柔軟性指数 (φ)。1次と2次κの比から柔軟性を推定",
                        "IC0": "0次情報含量。原子種の多様性。Shannon情報エントロピーベース",
                        "IC1": "1次情報含量。隣接原子の結合パターンの多様性",
                        "IC2": "2次情報含量。2結合先までの環境の多様性",
                        "TIC0": "0次全情報含量。IC0を原子数で重み付けした値",
                        "SIC0": "0次構造情報含量。IC0を正規化した値",
                        "SIC1": "1次構造情報含量。IC1を正規化した値",
                        "CIC0": "0次相補情報含量。最大エントロピーとIC0の差",
                        "CIC1": "1次相補情報含量。最大エントロピーとIC1の差",
                        "EState_VSA1": "EState表面積ビン1。電子リッチな原子の表面積",
                        "EState_VSA2": "EState表面積ビン2。やや電子リッチな原子の表面積",
                        "EState_VSA3": "EState表面積ビン3。電子プアな原子の表面積",
                        "MaxEStateIndex": "最大EState指数。最も電子受容しやすい原子の値",
                        "MinEStateIndex": "最小EState指数。最も電子供与しやすい原子の値",
                        "MaxAbsEStateIndex": "最大|EState|指数。電荷偏りが最大の原子",
                        "BCUTc-1h": "BCUT電荷-高値。電荷分布の最大固有値。分子の電荷の偏りパターン",
                        "BCUTc-1l": "BCUT電荷-低値。電荷分布の最小固有値。電荷の均一性",
                        "BCUTdv-1h": "BCUT原子価-高値。原子価分布の最大固有値。結合パターンの偏り",
                        "BCUTdv-1l": "BCUT原子価-低値。原子価分布の最小固有値",
                        "AXp-0dv": "0次原子価連結性指数。原子の孤立した性質(原子価ベース)",
                        "AXp-1dv": "1次原子価連結性指数。隣接原子との結合パターン(原子価ベース)",
                        "AXp-2dv": "2次原子価連結性指数。2結合先までの経路(原子価ベース)",
                        "Ipc": "Bonchev-Trinajstić情報含量。分子グラフの情報量",
                        "BalabanJ": "Balaban J指数。分子グラフの均一性。高い→対称的構造",
                        "FragCpx": "フラグメント複雑度。分子を構成するフラグメントの複雑さ",
                    }
                    # フォールバック辞書でMordred記述子を確実に意味付与
                    for _mord_name, _mord_meaning in _MORDRED_MEANINGS.items():
                        if _mord_name not in _meta or not _meta[_mord_name].get("meaning") or _meta[_mord_name]["meaning"] == _mord_name:
                            _meta[_mord_name] = {"meaning": _mord_meaning, "cat": "Mordred"}

                    # MolAI/Mol2Vec等のパターンベース意味付与（メタデータが登録されていない列向け）
                    for c in _precalc_df.columns:
                        if c not in _meta:
                            if c.startswith("MolAI_") or c.startswith("molai_"):
                                _idx = c.split("_")[-1]
                                _meta[c] = {"meaning": f"MolAI CNN潜在表現のPCA第{_idx}主成分", "cat": "埋め込み"}
                            elif c.startswith("Mol2Vec_"):
                                _idx = c.split("_")[-1]
                                _meta[c] = {"meaning": f"Mol2Vec潜在空間の第{_idx}次元", "cat": "埋め込み"}
                            elif c.startswith("ECFP_"):
                                _meta[c] = {"meaning": f"ECFP (Extended-Connectivity) フィンガープリント ビット{c.split('_')[-1]}", "cat": "FP"}
                            elif c.startswith("MACCS_"):
                                _meta[c] = {"meaning": f"MACCS構造キー ビット{c.split('_')[-1]}", "cat": "FP"}
                            elif c.startswith("TopologicalTorsion_"):
                                _meta[c] = {"meaning": f"トポロジカルトーション FP ビット{c.split('_')[-1]}", "cat": "FP"}
                            elif c.startswith("Avalon_"):
                                _meta[c] = {"meaning": f"Avalon FP ビット{c.split('_')[-1]}", "cat": "FP"}
                            elif c.startswith("FCFP_"):
                                _meta[c] = {"meaning": f"FCFP (Feature-Connectivity) FP ビット{c.split('_')[-1]}", "cat": "FP"}
                            elif c.startswith("DS_"):
                                _meta[c] = {"meaning": f"DescriptaStorus: {c[3:]}", "cat": "DescriptaStorus"}
                            elif c.startswith("Molfeat_"):
                                _meta[c] = {"meaning": f"Molfeat 分子特徴量 次元{c.split('_')[-1]}", "cat": "埋め込み"}
                            elif c.startswith("gasteiger_"):
                                _gast_map = {
                                    "gasteiger_q_max": "Gasteiger最大部分電荷",
                                    "gasteiger_q_min": "Gasteiger最小部分電荷",
                                    "gasteiger_q_range": "Gasteiger電荷レンジ（最大-最小）",
                                    "gasteiger_q_std": "Gasteiger電荷の標準偏差",
                                    "gasteiger_q_abs_mean": "Gasteiger|電荷|の平均",
                                }
                                _meta[c] = {"meaning": _gast_map.get(c, f"Gasteiger電荷統計: {c}"), "cat": "電子状態"}
                    if rec:
                        for d in rec.descriptors:
                            _meta.setdefault(d.name, {})
                            if not _meta[d.name].get("meaning"):
                                _meta[d.name]["meaning"] = d.meaning
                            if not _meta[d.name].get("cat"):
                                _meta[d.name]["cat"] = d.category

                    _engine_map = {}
                    for _em_name, _em_mod, _em_cls, _em_kw in _ENGINE_MAP_DEFS:
                        try:
                            _mod_obj = __import__(_em_mod, fromlist=[_em_cls])
                            _adp_obj = getattr(_mod_obj, _em_cls)(**_em_kw)
                            if _adp_obj.is_available():
                                for _dn in _adp_obj.get_descriptor_names():
                                    _engine_map.setdefault(_dn, _em_name)
                        except Exception:
                            pass
                    # MolAI列の推定（PCA列名パターンから）
                    for c in _precalc_df.columns:
                        if c.startswith("MolAI_") or c.startswith("molai_"):
                            _engine_map.setdefault(c, "MolAI")
                        elif c.startswith("ECFP_") or c.startswith("MACCS_") or c.startswith("TopologicalTorsion_") or c.startswith("Avalon_") or c.startswith("FCFP_"):
                            _engine_map.setdefault(c, "scikit-FP")
                        elif c.startswith("Mol2Vec_"):
                            _engine_map.setdefault(c, "Mol2Vec")
                        elif c.startswith("DS_"):
                            _engine_map.setdefault(c, "DescriptaStorus")

                    # 数え上げ系記述子の判定
                    _count_descs = set()
                    for c in _precalc_df.columns:
                        _clower = c.lower()
                        if (c.startswith("Num") or c.startswith("fr_")
                            or "Count" in c or "Ring" in c
                            or c.startswith("n_") or _clower.endswith("_count")):
                            _count_descs.add(c)

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # 記述子セットの登録・比較（選択の前 — 保存は直接表示）
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    if "_desc_sets" not in st.session_state:
                        _universal = [d for d in [
                            "MolWt","MolLogP","TPSA","NumHAcceptors","NumHDonors",
                            "NumRotatableBonds","RingCount","NumAromaticRings",
                            "FractionCSP3","HeavyAtomCount","MolMR","qed",
                            "BertzCT","MaxPartialCharge","MinPartialCharge",
                            "LabuteASA",
                        ] if d in (_precalc_df.columns if _precalc_df is not None else [])]
                        _molai_cols = [c for c in (_precalc_df.columns if _precalc_df is not None else []) if c.startswith("MolAI_") or c.startswith("molai_")]
                        _defaults = {}
                        if _universal:
                            _defaults["汎用基本セット"] = _universal
                        if _molai_cols:
                            _defaults["MolAI PCA"] = _molai_cols
                        st.session_state["_desc_sets"] = _defaults
                    _desc_sets = st.session_state["_desc_sets"]

                    # 保存ボタンは直接表示（expander不要）
                    _sc1, _sc2 = st.columns([3, 1])
                    with _sc1:
                        _set_name = st.text_input(
                            "💾 セット保存", value="", key="desc_set_name",
                            placeholder="名前を入力して「保存」（空なら自動番号）",
                            label_visibility="collapsed",
                        )
                    with _sc2:
                        if st.button(f"💾 現在の{n_sel}個を保存", key="btn_save_set", type="secondary", use_container_width=True):
                            _final_name = _set_name.strip()
                            if not _final_name:
                                _n = len(_desc_sets) + 1
                                while f"セット{_n}" in _desc_sets:
                                    _n += 1
                                _final_name = f"セット{_n}"
                            _desc_sets[_final_name] = list(st.session_state.get("adv_desc") or [])
                            st.session_state["_desc_sets"] = _desc_sets
                            st.success(f"✅ 「{_final_name}」保存（{len(st.session_state.get('adv_desc') or [])}個）")
                            st.rerun()

                    # 登録済みセット一覧（ある場合のみ表示）
                    if _desc_sets:
                        with st.expander(f"📋 登録済みセット（{len(_desc_sets)}件）", expanded=False):
                            for _sname, _sdescs in _desc_sets.items():
                                _sc_a, _sc_b, _sc_c = st.columns([4, 1, 1])
                                with _sc_a:
                                    st.markdown(f"**{_sname}** — {len(_sdescs)}個")
                                with _sc_b:
                                    if st.button("読込", key=f"load_set_{_sname}", use_container_width=True):
                                        st.session_state["adv_desc"] = _sdescs
                                        st.rerun()
                                with _sc_c:
                                    if st.button("🗑️", key=f"del_set_{_sname}", use_container_width=True):
                                        del _desc_sets[_sname]
                                        st.session_state["_desc_sets"] = _desc_sets
                                        st.rerun()

                            _use_multi = st.checkbox(
                                "🔄 全セット一括比較",
                                value=st.session_state.get("_use_multi_desc_sets", False),
                                key="chk_multi_desc",
                                help=f"ON: 各セットでモデルを構築し比較。実行時間約{len(_desc_sets)}倍。",
                            )
                            st.session_state["_use_multi_desc_sets"] = _use_multi

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # 記述子の選択（st.tabs: 推奨/エンジン/相関/数え上げ/全テーブル）
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    st.markdown("---")

                    # 全記述子テーブルに使うデータを先に準備
                    _rows_all = []
                    for c in _precalc_df.columns:
                        _r = _corr.get(c, float('nan'))
                        _m = _meta.get(c, {})
                        _rows_all.append({
                            "✅": c in _cur_sel,
                            "★": "★" if c in _rec_names else "",
                            "記述子名": c,
                            "ソース": _engine_map.get(c, ""),
                            "|r|": round(_r, 3) if pd.notna(_r) else None,
                            "ユニーク数": int(_precalc_df[c].nunique(dropna=True)),
                            "意味": _m.get("meaning", ""),
                        })
                    _rows_all.sort(key=lambda x: x["|r|"] if x["|r|"] is not None else -1, reverse=True)

                    # 数え上げ系をprecalc_dfから検出
                    _cnt_in_precalc = sorted(_count_descs & set(_precalc_df.columns))
                    _n_cnt_sel = sum(1 for c in _cnt_in_precalc if c in _cur_sel)

                    # エンジン別集計
                    _avail_engines = sorted(set(_engine_map.values()))
                    _eng_counts = {}
                    for _e in _avail_engines:
                        _e_cols = [c for c, e in _engine_map.items() if e == _e and c in _precalc_df.columns]
                        _eng_counts[_e] = (len(_e_cols), sum(1 for c in _e_cols if c in _cur_sel))

                    # 相関上位
                    _valid_corr = {k: v for k, v in _corr.items() if pd.notna(v) and v > 0}
                    _top_corr = sorted(_valid_corr.items(), key=lambda x: -x[1])

                    # ── クイック操作バー（常時表示） ──
                    st.markdown("---")
                    _qb1, _qb2, _qb3, _qb4 = st.columns(4)
                    with _qb1:
                        if st.button("✅ 全選択", key="quick_all", use_container_width=True):
                            st.session_state["adv_desc"] = list(_precalc_df.columns)
                            st.rerun()
                    with _qb2:
                        if st.button("❌ 全解除", key="quick_none", use_container_width=True):
                            st.session_state["adv_desc"] = []
                            st.rerun()
                    with _qb3:
                        if st.button("💡 推奨のみ", key="quick_rec", use_container_width=True):
                            _rec_in_data = [n for n in _rec_names if n in _precalc_df.columns]
                            st.session_state["adv_desc"] = _rec_in_data
                            st.rerun()
                    with _qb4:
                        if st.button("📈 相関Top20", key="quick_top20", use_container_width=True):
                            _top20 = [k for k, _ in _top_corr[:20]]
                            st.session_state["adv_desc"] = _top20
                            st.rerun()

                    # カテゴリ別分類
                    _CATEGORY_MAP = {
                        "🧪 物理化学": ["MolWt", "MolLogP", "TPSA", "MolMR", "HeavyAtomCount",
                                       "FractionCSP3", "LabuteASA", "BertzCT", "qed",
                                       "MaxPartialCharge", "MinPartialCharge", "HallKierAlpha",
                                       "ExactMolWt", "NumValenceElectrons"],
                        "🔗 トポロジー": ["NumHAcceptors", "NumHDonors", "NumRotatableBonds",
                                        "RingCount", "NumAromaticRings", "NumAliphaticRings",
                                        "NumSaturatedRings", "NumHeteroatoms", "NumHeterocycles"],
                        "🔬 フィンガープリント": [],  # ECFP_, MACCS_ 等のプレフィックスで動的割当
                        "🧠 埋め込み・学習型": [],     # MolAI_, Mol2Vec_ 等
                        "⚛️ 量子化学・物性": [],       # XTB/COSMO系
                    }
                    # 動的カテゴリ割当
                    _col_category = {}
                    for c in _precalc_df.columns:
                        _assigned = False
                        for _cat, _names in _CATEGORY_MAP.items():
                            if c in _names:
                                _col_category[c] = _cat
                                _assigned = True
                                break
                        if not _assigned:
                            if c.startswith("ECFP_") or c.startswith("MACCS_") or c.startswith("TopologicalTorsion_") or c.startswith("Avalon_") or c.startswith("FCFP_") or c.startswith("RDKit_") or c.startswith("AtomPair_") or c.startswith("MAP_") or c.startswith("ERG_") or c.startswith("Pattern_") or c.startswith("Layered_") or c.startswith("LINGO_"):
                                _col_category[c] = "🔬 フィンガープリント"
                            elif c.startswith("MolAI_") or c.startswith("molai_") or c.startswith("Mol2Vec_") or c.startswith("Molfeat_"):
                                _col_category[c] = "🧠 埋め込み・学習型"
                            elif _engine_map.get(c) in ("XTB", "COSMO-RS", "UniPKa"):
                                _col_category[c] = "⚛️ 量子化学・物性"
                            elif c in _count_descs:
                                _col_category[c] = "🔗 トポロジー"
                            else:
                                _col_category[c] = "🧪 物理化学"

                    _cat_counts = {}
                    for _cat in _CATEGORY_MAP:
                        _cat_cols = [c for c in _precalc_df.columns if _col_category.get(c) == _cat]
                        _cat_sel = sum(1 for c in _cat_cols if c in _cur_sel)
                        _cat_counts[_cat] = (len(_cat_cols), _cat_sel)

                    # タブ
                    from backend.chem.recommender import get_all_target_recommendations as _get_all_recs_p
                    all_recs = _get_all_recs_p()
                    _matched_rec = rec

                    _tab_preset, _tab_engine, _tab_cat, _tab_corr, _tab_count, _tab_all = st.tabs([
                        f"💡 推奨プリセット",
                        f"🏷️ エンジン別 ({len(_avail_engines)})",
                        f"🧬 カテゴリ別 ({len([v for v in _cat_counts.values() if v[0] > 0])})",
                        f"📈 相関係数順 ({len(_top_corr)})",
                        f"🔢 数え上げ系 ({len(_cnt_in_precalc)})",
                        f"📋 全{n_total}記述子",
                    ])

                    # ── タブ1: 推奨プリセット ──
                    with _tab_preset:
                        # マッチした推奨プリセット（目的変数に対応）
                        if _matched_rec:
                            _avail_rec = [d for d in _matched_rec.descriptors if d.name in _precalc_df.columns]
                            _unavail_rec = [d for d in _matched_rec.descriptors if d.name not in _precalc_df.columns]
                            _n_a = sum(1 for d in _avail_rec if d.name in _cur_sel)
                            st.markdown(f"**🎯 {_matched_rec.target_name}**（目的変数に対応）")
                            st.caption(_matched_rec.summary)
                            if _avail_rec:
                                st.dataframe(
                                    pd.DataFrame([{
                                        "選択": "✅" if d.name in _cur_sel else "—",
                                        "記述子名": d.name,
                                        "物理的意味": d.meaning,
                                        "ソース": d.library,
                                        "分類": d.category,
                                    } for d in _avail_rec]),
                                    use_container_width=True, hide_index=True,
                                    height=min(300, 35 + len(_avail_rec) * 35),
                                )
                                if _unavail_rec:
                                    _unames = ", ".join(f"{d.name}({d.library})" for d in _unavail_rec)
                                    st.caption(f"⚠️ 未計算のため未表示: {_unames}")
                                _rb1, _rb2 = st.columns(2)
                                with _rb1:
                                    if _n_a < len(_avail_rec):
                                        _missing = [d.name for d in _avail_rec if d.name not in _cur_sel]
                                        _btn_label = f"✅ {len(_missing)}個を追加: {', '.join(_missing)}"
                                        if st.button(_btn_label, key="radd_matched", type="primary", use_container_width=True):
                                            _cur_sel.update(d.name for d in _avail_rec)
                                            st.session_state["adv_desc"] = list(_cur_sel)
                                            st.rerun()
                                    else:
                                        st.success(f"{_matched_rec.target_name}: 全{len(_avail_rec)}件採用済み", icon="✅")
                                with _rb2:
                                    if _n_a > 0:
                                        _sel_names = [d.name for d in _avail_rec if d.name in _cur_sel]
                                        _del_label = f"❌ {len(_sel_names)}個を解除: {', '.join(_sel_names)}"
                                        if st.button(_del_label, key="rdel_matched", use_container_width=True):
                                            _cur_sel -= {d.name for d in _avail_rec}
                                            st.session_state["adv_desc"] = list(_cur_sel)
                                            st.rerun()
                            else:
                                st.info("計算済み記述子に該当なし")

                        # 他のプリセット（カテゴリ別 — expander可）
                        _other_recs = [r for r in all_recs if not _matched_rec or r.target_name != _matched_rec.target_name]
                        if _other_recs:
                            st.markdown("---")
                            _cat_groups: dict[str, list] = {}
                            for _or in _other_recs:
                                _cat = getattr(_or, 'category', 'その他')
                                _cat_groups.setdefault(_cat, []).append(_or)
                            for _cat_name, _cat_recs in _cat_groups.items():
                                _cat_total = 0; _cat_sel = 0
                                for _or in _cat_recs:
                                    _or_a = [d for d in _or.descriptors if d.name in _precalc_df.columns]
                                    _cat_total += len(_or_a)
                                    _cat_sel += sum(1 for d in _or_a if d.name in _cur_sel)
                                with st.expander(f"**{_cat_name}** ({_cat_sel}/{_cat_total})", expanded=False):
                                    for _or in _cat_recs:
                                        _or_avail = [d for d in _or.descriptors if d.name in _precalc_df.columns]
                                        _or_n = sum(1 for d in _or_avail if d.name in _cur_sel)
                                        if _or_avail:
                                            _desc_names_str = ", ".join(d.name for d in _or_avail)
                                            st.markdown(f"**{_or.target_name}** ({_or_n}/{len(_or_avail)})")
                                            st.caption(f"記述子: {_desc_names_str}")
                                            _oc1, _oc2 = st.columns(2)
                                            with _oc1:
                                                if _or_n < len(_or_avail):
                                                    _or_missing = [d.name for d in _or_avail if d.name not in _cur_sel]
                                                    _or_btn = f"✅ {len(_or_missing)}個追加: {', '.join(_or_missing[:4])}"
                                                    if len(_or_missing) > 4:
                                                        _or_btn += "..."
                                                    if st.button(_or_btn, key=f"radd_{_or.target_name}", use_container_width=True):
                                                        _cur_sel.update(d.name for d in _or_avail)
                                                        st.session_state["adv_desc"] = list(_cur_sel)
                                                        st.rerun()
                                            with _oc2:
                                                if _or_n > 0:
                                                    _or_sel = [d.name for d in _or_avail if d.name in _cur_sel]
                                                    if st.button(f"❌ {len(_or_sel)}個解除", key=f"rdel_{_or.target_name}", use_container_width=True):
                                                        _cur_sel -= {d.name for d in _or_avail}
                                                        st.session_state["adv_desc"] = list(_cur_sel)
                                                        st.rerun()


                    # ── タブ2: エンジン別（個別記述子選択） ──
                    with _tab_engine:
                        st.caption("各エンジンの記述子を個別に選択できます。化学的意味を参考に必要な記述子を選んでください。")
                        for _eng in _avail_engines:
                            _e_total, _e_sel = _eng_counts[_eng]
                            _e_cols = [c for c, e in _engine_map.items() if e == _eng and c in _precalc_df.columns]
                            if not _e_cols:
                                continue

                            with st.expander(f"**{_eng}** — {_e_sel}/{_e_total}個選択中", expanded=(_e_sel > 0 and _e_total <= 30)):
                                # 記述子ごとのデータを構築（|r|降順ソート）
                                _eng_rows = []
                                for _ec in _e_cols:
                                    _ec_corr = _corr.get(_ec, float('nan'))
                                    _ec_corr_val = round(_ec_corr, 3) if pd.notna(_ec_corr) else None
                                    _ec_meaning = _meta.get(_ec, {}).get("meaning", "")
                                    _eng_rows.append({
                                        "✅": _ec in _cur_sel,
                                        "記述子名": _ec,
                                        "|r|": _ec_corr_val,
                                        "物理的意味": _ec_meaning,
                                    })
                                _eng_rows.sort(key=lambda x: x["|r|"] if x["|r|"] is not None else -1, reverse=True)
                                _eng_df = pd.DataFrame(_eng_rows)

                                _eng_edited = st.data_editor(
                                    _eng_df,
                                    column_config={
                                        "✅": st.column_config.CheckboxColumn("選択", default=False),
                                        "記述子名": st.column_config.TextColumn("記述子名", width="medium"),
                                        "|r|": st.column_config.NumberColumn("|r|", format="%.3f", width="small"),
                                        "物理的意味": st.column_config.TextColumn("化学的意味", width="large"),
                                    },
                                    disabled=["記述子名", "|r|", "物理的意味"],
                                    use_container_width=True,
                                    hide_index=True,
                                    key=f"eng_editor_{_eng}",
                                    height=min(500, 40 + len(_eng_rows) * 35),
                                )

                                # チェックボックス変更を反映
                                if _eng_edited is not None:
                                    _eng_new_sel = set(_eng_edited[_eng_edited["✅"] == True]["記述子名"].tolist())
                                    _eng_old_sel = set(c for c in _e_cols if c in _cur_sel)
                                    if _eng_new_sel != _eng_old_sel:
                                        # このエンジンの選択を更新
                                        _cur_sel -= _eng_old_sel
                                        _cur_sel |= _eng_new_sel
                                        st.session_state["adv_desc"] = list(_cur_sel)
                                        st.rerun()

                    # ── タブ3: 相関係数順 ──
                    with _tab_corr:
                        if _top_corr:
                            _corr_n = st.slider("上位N件", 5, min(100, len(_top_corr)), 20, key="corr_top_n")
                            _top_n_names = [k for k, _ in _top_corr[:_corr_n]]
                            _n_already = sum(1 for n in _top_n_names if n in _cur_sel)
                            st.dataframe(
                                pd.DataFrame([{
                                    "✅": "✅" if n in _cur_sel else "—",
                                    "記述子": n,
                                    "|r|": round(v, 3),
                                    "ソース": _engine_map.get(n, ""),
                                    "化学的意味": _meta.get(n, {}).get("meaning", ""),
                                } for n, v in _top_corr[:_corr_n]]),
                                use_container_width=True, hide_index=True,
                                height=min(400, 35 + _corr_n * 35),
                            )
                            _cb1, _cb2 = st.columns(2)
                            with _cb1:
                                if _n_already < _corr_n:
                                    _n_new_corr = _corr_n - _n_already
                                    if st.button(f"✅ |r|上位{_corr_n}件を追加（新規{_n_new_corr}件）", key="corr_add", type="primary", use_container_width=True):
                                        _cur_sel.update(_top_n_names)
                                        st.session_state["adv_desc"] = list(_cur_sel)
                                        st.rerun()
                                else:
                                    st.success(f"上位{_corr_n}件は全て採用済み", icon="✅")
                            with _cb2:
                                if _n_already > 0:
                                    if st.button("解除", key="corr_del", use_container_width=True):
                                        _cur_sel -= set(_top_n_names)
                                        st.session_state["adv_desc"] = list(_cur_sel)
                                        st.rerun()
                        else:
                            st.info("相関係数を算出できません（目的変数が数値でない等）")

                    # ── タブ3: カテゴリ別 ──
                    with _tab_cat:
                        st.caption("記述子を物理的意味で分類。カテゴリ単位で一括操作可能です。")
                        for _cat_name in _CATEGORY_MAP:
                            _cat_cols = [c for c in _precalc_df.columns if _col_category.get(c) == _cat_name]
                            if not _cat_cols:
                                continue
                            _cat_total = len(_cat_cols)
                            _cat_sel = sum(1 for c in _cat_cols if c in _cur_sel)

                            with st.expander(f"**{_cat_name}** ({_cat_sel}/{_cat_total}個選択中)", expanded=(_cat_sel > 0)):
                                # コンパクトなテーブル表示
                                _cat_rows = [{
                                    "✅": "✅" if c in _cur_sel else "—",
                                    "記述子": c,
                                    "ソース": _engine_map.get(c, ""),
                                    "|r|": round(_corr.get(c, float('nan')), 3) if pd.notna(_corr.get(c, float('nan'))) else None,
                                    "意味": _meta.get(c, {}).get("meaning", ""),
                                } for c in _cat_cols]
                                _cat_rows.sort(key=lambda x: x["|r|"] if x["|r|"] is not None else -1, reverse=True)
                                st.dataframe(
                                    pd.DataFrame(_cat_rows),
                                    use_container_width=True, hide_index=True,
                                    height=min(400, 35 + len(_cat_rows) * 35),
                                )
                                _cc1, _cc2 = st.columns(2)
                                with _cc1:
                                    if _cat_sel < _cat_total:
                                        _n_new_cat = _cat_total - _cat_sel
                                        if st.button(f"✅ {_n_new_cat}個追加", key=f"cat_add_{_cat_name}", type="primary", use_container_width=True):
                                            _cur_sel.update(_cat_cols)
                                            st.session_state["adv_desc"] = list(_cur_sel)
                                            st.rerun()
                                    else:
                                        st.success("全件採用済み", icon="✅")
                                with _cc2:
                                    if _cat_sel > 0:
                                        if st.button(f"❌ {_cat_sel}個解除", key=f"cat_del_{_cat_name}", use_container_width=True):
                                            _cur_sel -= set(_cat_cols)
                                            st.session_state["adv_desc"] = list(_cur_sel)
                                            st.rerun()


                    # ── タブ5: 数え上げ系 ──
                    with _tab_count:
                        if _cnt_in_precalc:
                            st.dataframe(
                                pd.DataFrame([{
                                    "✅": "✅" if c in _cur_sel else "—",
                                    "記述子": c,
                                    "ソース": _engine_map.get(c, ""),
                                    "|r|": round(_corr.get(c, float('nan')), 3) if pd.notna(_corr.get(c, float('nan'))) else None,
                                    "意味": _meta.get(c, {}).get("meaning", ""),
                                } for c in _cnt_in_precalc]),
                                use_container_width=True, hide_index=True,
                                height=min(500, 35 + len(_cnt_in_precalc) * 35),
                            )
                            _cb1, _cb2 = st.columns(2)
                            with _cb1:
                                if _n_cnt_sel < len(_cnt_in_precalc):
                                    if st.button(f"✅ 全{len(_cnt_in_precalc)}件追加", key="cnt_add", type="primary", use_container_width=True):
                                        _cur_sel.update(_cnt_in_precalc)
                                        st.session_state["adv_desc"] = list(_cur_sel)
                                        st.rerun()
                                else:
                                    st.success("全件採用済み", icon="✅")
                            with _cb2:
                                if _n_cnt_sel > 0:
                                    if st.button("解除", key="cnt_del", use_container_width=True):
                                        _cur_sel -= set(_cnt_in_precalc)
                                        st.session_state["adv_desc"] = list(_cur_sel)
                                        st.rerun()
                        else:
                            st.info("数え上げ系記述子なし")


                    # ── タブ6: 全記述子テーブル ──
                    with _tab_all:
                        # フィルタ行
                        _ff1, _ff2, _ff3 = st.columns([2, 1, 3])
                        with _ff1:
                            _flt = st.pills("表示", ["全て", "選択中", "未選択", "推奨のみ"], default="全て", key="tbl_flt")
                        with _ff2:
                            _flt_corr = st.slider("|r|≥", 0.0, 1.0, 0.0, 0.01, key="tbl_corr")
                        with _ff3:
                            _srch = st.text_input("検索", key="tbl_srch", placeholder="記述子名で絞り込み")

                        _tdf = pd.DataFrame(_rows_all)
                        if _flt == "選択中":
                            _tdf = _tdf[_tdf["✅"] == True]
                        elif _flt == "未選択":
                            _tdf = _tdf[_tdf["✅"] == False]
                        elif _flt == "推奨のみ":
                            _tdf = _tdf[_tdf["★"] == "★"]
                        if _flt_corr > 0:
                            _tdf = _tdf[_tdf["|r|"].fillna(0) >= _flt_corr]
                        if _srch:
                            _tdf = _tdf[_tdf["記述子名"].str.contains(_srch, case=False, na=False)]

                        st.caption(f"表示中: {len(_tdf)}件 / 全{n_total}件（{n_sel}個選択中）")

                        _edited = st.data_editor(
                            _tdf,
                            column_config={
                                "✅": st.column_config.CheckboxColumn("選択", default=False),
                                "★": st.column_config.TextColumn("推奨", width="small"),
                                "記述子名": st.column_config.TextColumn("記述子名", width="medium"),
                                "ソース": st.column_config.TextColumn("エンジン", width="small"),
                                "|r|": st.column_config.NumberColumn("|r|", format="%.3f", width="small"),
                                "ユニーク数": st.column_config.NumberColumn("ユニーク", width="small"),
                                "意味": st.column_config.TextColumn("物理的意味", width="large"),
                            },
                            disabled=["★", "記述子名", "ソース", "|r|", "ユニーク数", "意味"],
                            use_container_width=True,
                            hide_index=True,
                            key="desc_editor_main",
                            height=min(700, 40 + len(_tdf) * 35),
                        )

                        if _edited is not None:
                            _new_sel = set(_edited[_edited["✅"] == True]["記述子名"].tolist())
                            _invisible = _cur_sel - set(_tdf["記述子名"].tolist())
                            _final = _new_sel | _invisible
                            if _final != _cur_sel:
                                st.session_state["adv_desc"] = list(_final)

                    st.session_state.setdefault("adv_desc", list(_cur_sel))

                    # ── 高度な設定（折りたたみ：詳しい人向け） ──
                    with st.expander("🔧 高度なエンジン設定（通常は変更不要）", expanded=False):
                        st.caption(
                            "全エンジンでの記述子計算は自動的に実行済みです。"
                            "MolAI PCA次元の変更やエンジン個別の切替が必要な場合のみお使いください。"
                        )

                        _ADV_ENGINE_LIST = [
                            ("🧪 RDKit",            "backend.chem.rdkit_adapter",          "RDKitAdapter",          {"compute_fp": False}, "~200種",       "🟢 高速"),
                            ("📐 Mordred",           "backend.chem.mordred_adapter",        "MordredAdapter",        {"selected_only": True},"~1,800種",    "🟡 中程度"),
                            ("🔩 GroupContrib",      "backend.chem.group_contrib_adapter",  "GroupContribAdapter",   {},                     "~15種",        "🟢 高速"),
                            ("📊 DescriptaStorus",   "backend.chem.descriptastorus_adapter","DescriptaStorusAdapter",{},                     "~200種",       "🟢 高速"),
                            ("🤖 MolAI (CNN+PCA)",   "backend.chem.molai_adapter",          "MolAIAdapter",          {"n_components": 6},    "n_comp",       "🟡 中程度"),
                            ("🔑 scikit-FP",         "backend.chem.skfp_adapter",           "SkfpAdapter",           {"fp_types": ["ECFP","MACCS"]},"~4,000+種","🟢 高速"),
                            ("🔀 UMA",               "backend.chem.uma_adapter",            "UMAAdapter",            {},                     "可変",          "🟡 中程度"),
                            ("📝 Mol2Vec",           "backend.chem.mol2vec_adapter",        "Mol2VecAdapter",        {},                     "300次元",      "🟢 高速"),
                            ("📁 PaDEL",             "backend.chem.padel_adapter",          "PaDELAdapter",          {},                     "~1,800種",     "🟡 中程度"),
                            ("🧬 Molfeat",           "backend.chem.molfeat_adapter",        "MolfeatAdapter",        {},                     "可変",          "🟡 中程度"),
                            ("⚛️ XTB",               "backend.chem.xtb_adapter",            "XTBAdapter",            {},                     "~20種",        "🔴 重い"),
                            ("💧 COSMO-RS",          "backend.chem.cosmo_adapter",          "CosmoAdapter",          {},                     "~10種",        "🔴 重い"),
                            ("⚗️ UniPKa",            "backend.chem.unipka_adapter",         "UniPkaAdapter",         {},                     "~5種",         "🟡 中程度"),
                            ("🧪 Chemprop",          "backend.chem.chemprop_adapter",       "ChempropAdapter",       {},                     "可変",          "🔴 重い"),
                        ]

                        for _ae_name, _ae_mod, _ae_cls, _ae_kw, _ae_dims, _ae_cost in _ADV_ENGINE_LIST:
                            try:
                                _ae_mod_obj = __import__(_ae_mod, fromlist=[_ae_cls])
                                _ae_adp = getattr(_ae_mod_obj, _ae_cls)(**_ae_kw)
                                _ae_avail = _ae_adp.is_available()
                            except Exception:
                                _ae_avail = False
                            _ae_status = "✅ 計算済み" if _ae_avail else "🚫 未インストール"
                            _ae_color = "#4ade80" if _ae_avail else "#f87171"
                            st.markdown(
                                f"- {_ae_name} <span style='color:{_ae_color}'>{_ae_status}</span> | {_ae_dims} | {_ae_cost}",
                                unsafe_allow_html=True,
                            )

                        # MolAI PCA次元設定
                        st.markdown("---")
                        st.markdown("**MolAI CNN + PCA 設定**")
                        try:
                            _molai_mod = __import__("backend.chem.molai_adapter", fromlist=["MolAIAdapter"])
                            _MolAIAdp_chk = getattr(_molai_mod, "MolAIAdapter")
                            _molai_avail = _MolAIAdp_chk(n_components=6).is_available()
                        except Exception:
                            _molai_avail = False
                        if _molai_avail:
                            _molai_n = st.slider(
                                "PCA次元数",
                                min_value=1, max_value=256,
                                value=st.session_state.get("molai_n_components", 6),
                                step=1,
                                key="slider_molai_n_adv",
                                help="MolAI CNNの潜在表現をPCAで圧縮する次元数。デフォルト6。",
                            )
                            if _molai_n != st.session_state.get("molai_n_components", 6):
                                st.session_state["molai_n_components"] = _molai_n
                                st.session_state["precalc_done"] = False
                                st.session_state["precalc_smiles_df"] = None
                                st.rerun()

                            # MolAI寄与率グラフ
                            _mev = st.session_state.get("molai_explained_variance")
                            if _mev and _mev.get("ratio"):
                                import plotly.graph_objects as _go_ev
                                _evr = _mev["ratio"]; _evc = _mev["cumulative"]; _nc = _mev["n_components"]
                                _pcs = [f"PC{i+1}" for i in range(len(_evr))]
                                _fig_ev = _go_ev.Figure()
                                _fig_ev.add_bar(x=_pcs, y=[v*100 for v in _evr], name="寄与率 (%)", marker_color="#4c9be8")
                                _fig_ev.add_scatter(x=_pcs, y=[v*100 for v in _evc], name="累積寄与率 (%)",
                                                    mode="lines+markers", yaxis="y2",
                                                    line=dict(color="#f4a261", width=2), marker=dict(size=5))
                                _fig_ev.update_layout(
                                    yaxis=dict(title="寄与率 (%)", range=[0, max(v*100 for v in _evr)*1.15]),
                                    yaxis2=dict(title="累積 (%)", overlaying="y", side="right", range=[0,105], showgrid=False),
                                    legend=dict(orientation="h", y=1.15), height=250,
                                    margin=dict(l=10, r=10, t=30, b=30),
                                )
                                st.plotly_chart(_fig_ev, use_container_width=True)
                        else:
                            st.info("MolAIは現在利用できません。")

                        # 再計算（通常不要）
                        with st.expander("⚙️ 記述子の再計算（通常は不要）", expanded=False):
                            st.caption("分子構造データを変更した場合のみ再計算が必要です。")
                            if st.button("🔄 全記述子を再計算する", key="btn_recalc_adv"):
                                st.session_state["precalc_done"] = False
                                st.session_state["precalc_smiles_df"] = None
                                st.rerun()



        # ══════════════════════════════════════════════════════════════
        # サブタブ4: 特徴量EDA
        # ══════════════════════════════════════════════════════════════
        with ds_tab4:
            df = st.session_state.get("df")
            if df is None:
                st.warning("⚠️ まず「📂 データ読込」タブでデータを読み込んでください。")
            else:
                # データプレビュー
                smiles_col_eda = st.session_state.get("smiles_col")
                target_col_eda = st.session_state.get("target_col")

                df_preview = df.head(100).copy()
                added_smiles_cols = []

                if smiles_col_eda and smiles_col_eda in df_preview.columns:
                    selected_desc_preview = st.session_state.get("adv_desc") or []
                    _precalc = st.session_state.get("precalc_smiles_df")

                    if _precalc is not None and selected_desc_preview:
                        # 事前計算済みデータから選択された記述子を直接利用
                        _avail_cols = [c for c in selected_desc_preview if c in _precalc.columns]
                        if _avail_cols:
                            _desc_slice = _precalc[_avail_cols].loc[df_preview.index.intersection(_precalc.index)]
                            df_preview = pd.concat([df_preview, _desc_slice], axis=1)
                            added_smiles_cols = _avail_cols
                    else:
                        # 事前計算データがない場合はSmilesDescriptorTransformerで計算
                        try:
                            from backend.chem.smiles_transformer import SmilesDescriptorTransformer
                            transformer = SmilesDescriptorTransformer(
                                smiles_col=smiles_col_eda,
                                selected_descriptors=selected_desc_preview if selected_desc_preview else None
                            )
                            df_preview = transformer.fit_transform(df_preview)
                            added_smiles_cols = [c for c in df_preview.columns if c not in df.columns]
                        except Exception as e:
                            st.warning(f"SMILES展開プレビュー失敗: {e}")

                df_preview = df_preview.apply(pd.to_numeric, errors='ignore').convert_dtypes()

                st.markdown("### 📊 データプレビュー")
                from frontend_streamlit.pages.pipeline import eda_page
                eda_page.render_preview(df_preview, target_col_eda, added_smiles_cols)

        # ══════════════════════════════════════════════════════════════
        # サブタブ5: パイプライン設計
        # ══════════════════════════════════════════════════════════════
        with ds_tab5:
            df = st.session_state.get("df")
            if df is None:
                st.warning("⚠️ まず「📂 データ読込」タブでデータを読み込んでください。")
            else:
                target_col_adv = st.session_state.get("target_col")
                _auto_task_adv = "regression" if pd.api.types.is_float_dtype(df[target_col_adv]) else "classification"

                st.markdown("### ⚙️ パイプラインの詳細設定")
                st.caption("AutoMLの探索範囲や、モデルへの制約を設定します。")

                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    st.markdown("**🛡️ バリデーション設定**")
                    cv_folds = st.number_input("CVフォールド数", 2, 10, 5, key="adv_cv")

                    st.markdown("**⏰ 探索時間設定**")
                    timeout = st.number_input("最大探索時間（秒）", 10, 3600, 300, 30, key="adv_timeout")

                    st.markdown("**📏 スケーラー設定**")
                    scaler = st.selectbox(
                        "数値スケーリング",
                        ["auto", "standard", "robust", "minmax", "none"],
                        index=0, key="adv_scaler"
                    )

                with col_p2:
                    st.markdown("**🤖 使用モデルの選択**")
                    if _auto_task_adv == "regression":
                        model_options = {
                            "lr": "Linear Regression (線形)",
                            "ridge": "Ridge / Lasso",
                            "rf": "Random Forest",
                            "lgbm": "LightGBM (高速GPU対応)",
                            "xgb": "XGBoost",
                            "gp": "Gaussian Process (不確かさ)",
                            "svm": "SVR",
                        }
                    else:
                        model_options = {
                            "logreg": "Logistic Regression",
                            "rf": "Random Forest",
                            "lgbm": "LightGBM",
                            "xgb": "XGBoost",
                            "svm": "SVC",
                            "knn": "K-Neighbors",
                        }

                    # デフォルト全選択
                    selected_models = st.multiselect(
                        "探索対象モデル",
                        options=list(model_options.keys()),
                        format_func=lambda x: model_options[x],
                        default=list(model_options.keys()),
                        key="adv_models_sel",
                    )

                    st.markdown("**🔧 その他オプション**")
                    do_pca = st.checkbox("特徴量次元削減 (PCA) を試行", value=True, key="adv_do_pca")
                    do_shap = st.checkbox("SHAP重要度解析を実行", value=True, key="adv_do_shap")

                # 単調性制約設定
                st.markdown("---")
                st.markdown("**📈 単調性制約 (Monotonic Constraints)**")
                st.caption("「分子量が大きいほど沸点が高くなる」といったドメイン知識を指定できます。")
                
                feat_candidates = [c for c in df.columns if c != target_col_adv and pd.api.types.is_numeric_dtype(df[c])]
                # 展開後の記述子も候補に入れる
                if st.session_state.get("precalc_done"):
                    feat_candidates += [c for c in st.session_state["precalc_smiles_df"].columns if c in (st.session_state.get("adv_desc") or [])]

                if feat_candidates:
                    st.session_state.setdefault("_monotonic_constraints_dict", {})
                    m_dict = st.session_state["_monotonic_constraints_dict"]

                    # リセット・全削除
                    if st.button("🗑️ 制約をすべて解除", key="clear_mono"):
                        st.session_state["_monotonic_constraints_dict"] = {}
                        st.rerun()

                    selected_mono_feat = st.selectbox("制約を追加する列", ["（選択してください）"] + feat_candidates, key="mono_col")
                    if selected_mono_feat != "（選択してください）":
                        m_type = st.radio("制約の方向", ["増加 (Increasing) 📈", "減少 (Decreasing) 📉"], key="mono_type", horizontal=True)
                        if st.button("➕ 制約を追加", key="add_mono"):
                            m_dict[selected_mono_feat] = 1 if "増加" in m_type else -1
                            st.session_state["_monotonic_constraints_dict"] = m_dict
                            st.rerun()

                    if m_dict:
                        st.markdown("**現在の設定済み制約:**")
                        for f, v in m_dict.items():
                            st.markdown(f"- `{f}`: {'📈 増加' if v==1 else '📉 減少'}")
                else:
                    st.info("単調性制約を設定できる数値列がありません。")

                # 詳細設定の保存（st.session_state["_adv"] に集約）
                st.session_state["_adv"] = {
                    "cv_folds": cv_folds,
                    "timeout": timeout,
                    "scaler": scaler,
                    "models": selected_models,
                    "do_pca": do_pca,
                    "do_shap": do_shap,
                    "selected_descriptors": st.session_state.get("adv_desc"),
                }

    # ──────────────────────────────────────────────────────────
    # TAB 2: 結果確認（解析済みのときのみ）
    # ──────────────────────────────────────────────────────────
    with tab2:
        if not has_result:
            st.info("🚀 ①のタブで「解析開始」を押すと、ここに結果が表示されます。")
            st.markdown(
                '<div style="text-align:center; padding:3rem; color:#444;">'
                '<div style="font-size:4rem; margin-bottom:1rem;">📊</div>'
                '<div>解析結果はまだありません</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            render_results_tab()

def render_results_tab():
    """解析結果タブの堅牢な描画関数（デバッグ・エラー監視機能付き）"""
    import traceback
    
    # ── 1. 結果データの取得 ────────────────────────────────────
    pr = st.session_state.get("pipeline_result")
    ar = st.session_state.get("automl_result")
    
    # ── 2. デバッグ用：セッション状態の診断 ──────────────────────
    with st.expander("🛠️ デバッグ・診断情報 (表示が空の場合に確認してください)", expanded=False):
        st.write("### Session State Diagnostics")
        st.write(f"- `pipeline_result` 存在: {pr is not None}")
        st.write(f"- `automl_result` 存在: {ar is not None}")
        st.write(f"- `df` 行数: {len(st.session_state['df']) if st.session_state.get('df') is not None else 'None'}")
        st.write(f"- `target_col`: {st.session_state.get('target_col')}")
        
        # 内部データのキー一覧
        st.write("- `session_state` keys:", list(st.session_state.keys()))
        
        if st.button("🔧 状態を強制再描画 (st.rerun)"):
            st.rerun()

    # ── 3. 描画の実行 ──────────────────────────────────────────
    try:
        if pr is not None:
            # フルパイプライン結果 (EDA/PCA/SHAP/AutoML) を表示
            from frontend_streamlit.pages.automl_page import _show_pipeline_result
            _show_pipeline_result(pr)
        elif ar is not None:
            # 旧形式またはAutoMLのみの結果を表示
            from frontend_streamlit.pages.pipeline import evaluation_page
            evaluation_page.render()
        else:
            st.warning("⚠️ 解析結果をロードできませんでした。")
            st.info("一度「① データ設定」に戻り、もう一度解析を実行してみてください。")
            
    except Exception as e:
        st.error(f"❌ 結果の描画中にエラーが発生しました: {e}")
        st.code(traceback.format_exc())
        st.info("データの整合性が壊れている可能性があります。右上の「やり直す」ボタンで初期化を試みてください。")

# ── フッター ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#555; font-size:0.8rem;">'
    '© 2024 ChemAI ML Studio | Accelerating Material Discovery with Agentic AI'
    '</div>',
    unsafe_allow_html=True,
)
