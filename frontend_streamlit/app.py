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
            st.info("🔄 設定を変えて再解析したい場合は、各タブで設定を変更してから「② 解析実行」タブへ進んでください。")

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
                    use_smiles = st.checkbox("SMILES（化合物構造）列を含める", value=False, key="demo_smiles")
                    c_r, c_c = st.columns(2)
                    _DUMMY_SMILES = ["C", "CC", "CCC", "CCO", "CCN", "c1ccccc1", "c1ccccc1O", "CC(=O)O", "CC(C)C", "C1CCCCC1", "c1ccncc1", "c1ncncn1", "C1COCCO1"]
                    with c_r:
                        if st.button("🧪 回帰サンプル", use_container_width=True, key="demo_reg"):
                            np.random.seed(42); n = 200
                            if use_smiles:
                                base_df = pd.DataFrame({"SMILES": np.random.choice(_DUMMY_SMILES, n), "solubility_logS": np.random.randn(n) * 2 - 2})
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
                            st.session_state["adv_models"] = ["lr", "rf"]
                            st.rerun()
                    with c_c:
                        if st.button("🏷️ 分類サンプル", use_container_width=True, key="demo_cls"):
                            np.random.seed(42); n = 200
                            if use_smiles:
                                base_df = pd.DataFrame({"SMILES": np.random.choice(_DUMMY_SMILES, n), "is_toxic": np.random.randint(0, 2, n)})
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
                            st.session_state["adv_models"] = ["logreg", "rf"]
                            st.rerun()

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
                        st.markdown(f"**🧬 SMILES列**: `{_det_smiles}`（自動検出済み）")
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
                if not st.session_state.get("precalc_done", False):
                    from backend.chem.rdkit_adapter import RDKitAdapter
                    from backend.chem.recommender import get_target_recommendation_by_name as _get_rec

                    smiles_series = df[smiles_col_sf]
                    valid_mask = smiles_series.notna()
                    smiles_list = smiles_series[valid_mask].tolist()
                    valid_idx = smiles_series[valid_mask].index
                    n = len(smiles_list)

                    st.info(f"⚙️ **{n} 件のSMILESに対し、利用可能な全エンジンで記述子を自動計算中...**")

                    rdkit = RDKitAdapter(compute_fp=False)
                    df_result = pd.DataFrame(index=range(n))

                    with st.spinner("全RDKit記述子を計算中..."):
                        if rdkit.is_available():
                            try:
                                all_rdkit_names = rdkit.get_descriptor_names()
                                df_tmp = rdkit.compute(smiles_list, selected_descriptors=all_rdkit_names).descriptors
                                df_result = pd.concat([df_result, df_tmp], axis=1)
                            except Exception:
                                pass

                    with st.spinner("Mordred記述子を計算中..."):
                        try:
                            from backend.chem import MordredAdapter as _MordPre
                            _mord = _MordPre(selected_only=True)
                            if _mord.is_available():
                                _mord_res = _mord.compute(smiles_list)
                                if hasattr(_mord_res, 'descriptors') and _mord_res.descriptors is not None:
                                    _new_cols = [c for c in _mord_res.descriptors.columns if c not in df_result.columns]
                                    if _new_cols:
                                        df_result = pd.concat([df_result, _mord_res.descriptors[_new_cols]], axis=1)
                        except Exception:
                            pass

                    with st.spinner("原子団寄与法記述子を計算中..."):
                        try:
                            from backend.chem import GroupContribAdapter as _GCAPre
                            _gca = _GCAPre()
                            if _gca.is_available():
                                _gca_res = _gca.compute(smiles_list)
                                if hasattr(_gca_res, 'descriptors') and _gca_res.descriptors is not None:
                                    _new_cols = [c for c in _gca_res.descriptors.columns if c not in df_result.columns]
                                    if _new_cols:
                                        df_result = pd.concat([df_result, _gca_res.descriptors[_new_cols]], axis=1)
                        except Exception:
                            pass

                    # MolAI CNN (デフォルト PCA次元=6)
                    try:
                        from backend.chem.molai_adapter import MolAIAdapter as _MolAIAdapterPre
                        _molai_n_pre = st.session_state.get("molai_n_components", 6)
                        _molai_adp = _MolAIAdapterPre(n_components=_molai_n_pre)
                        if _molai_adp.is_available():
                            with st.spinner("MolAI CNN特徴量を計算中..."):
                                _molai_result = _molai_adp.compute(smiles_list)
                                if hasattr(_molai_result, 'descriptors') and _molai_result.descriptors is not None:
                                    _new_cols = [c for c in _molai_result.descriptors.columns if c not in df_result.columns]
                                    if _new_cols:
                                        df_result = pd.concat([df_result, _molai_result.descriptors[_new_cols]], axis=1)
                                if hasattr(_molai_adp, '_pca') and _molai_adp._pca is not None:
                                    _evr = _molai_adp._pca.explained_variance_ratio_
                                    st.session_state["molai_explained_variance"] = {
                                        "ratio": _evr.tolist(), "cumulative": _evr.cumsum().tolist(),
                                        "n_components": _molai_n_pre,
                                    }
                    except Exception:
                        pass

                    # XTB (利用可能なら)
                    try:
                        from backend.chem import XTBAdapter as _XTBPre
                        _xtb = _XTBPre()
                        if _xtb.is_available():
                            with st.spinner("XTB量子化学記述子を計算中..."):
                                _xtb_res = _xtb.compute(smiles_list)
                                if hasattr(_xtb_res, 'descriptors') and _xtb_res.descriptors is not None:
                                    _new_cols = [c for c in _xtb_res.descriptors.columns if c not in df_result.columns]
                                    if _new_cols:
                                        df_result = pd.concat([df_result, _xtb_res.descriptors[_new_cols]], axis=1)
                    except Exception:
                        pass

                    # UniPKa (利用可能なら)
                    try:
                        from backend.chem import UniPkaAdapter as _UniPre
                        _upka = _UniPre()
                        if _upka.is_available():
                            with st.spinner("UniPKa pKa記述子を計算中..."):
                                _upka_res = _upka.compute(smiles_list)
                                if hasattr(_upka_res, 'descriptors') and _upka_res.descriptors is not None:
                                    _new_cols = [c for c in _upka_res.descriptors.columns if c not in df_result.columns]
                                    if _new_cols:
                                        df_result = pd.concat([df_result, _upka_res.descriptors[_new_cols]], axis=1)
                    except Exception:
                        pass

                    # COSMO-RS (利用可能なら)
                    try:
                        from backend.chem import CosmoAdapter as _CosmoPre
                        _cosmo = _CosmoPre()
                        if _cosmo.is_available():
                            with st.spinner("COSMO-RS溶媒和記述子を計算中..."):
                                _cosmo_res = _cosmo.compute(smiles_list)
                                if hasattr(_cosmo_res, 'descriptors') and _cosmo_res.descriptors is not None:
                                    _new_cols = [c for c in _cosmo_res.descriptors.columns if c not in df_result.columns]
                                    if _new_cols:
                                        df_result = pd.concat([df_result, _cosmo_res.descriptors[_new_cols]], axis=1)
                    except Exception:
                        pass

                    # クリーンアップ
                    df_result = df_result.loc[:, ~df_result.columns.duplicated()]
                    df_result.index = valid_idx
                    df_result = df_result.apply(pd.to_numeric, errors="coerce").convert_dtypes()
                    df_result = df_result.dropna(axis=1, how="all")

                    # 推奨記述子を初期選択に設定
                    rec = _get_rec(target_col_sf)
                    if rec:
                        _init = [d.name for d in rec.descriptors if d.name in df_result.columns]
                    else:
                        _init = [d for d in ["MolWt","LogP","TPSA","HBA","HBD","RotBonds",
                                              "RingCount","AromaticRingCount","FractionCSP3",
                                              "HeavyAtoms","MolMR","HallKierAlpha"] if d in df_result.columns]
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

                    _cur_sel = set(st.session_state.get("adv_desc", []))
                    n_total = len(_precalc_df.columns)
                    n_sel = len(_cur_sel)

                    st.markdown(
                        f"### ⚗️ 記述子の選択　"
                        f"<span style='color:#4ade80; font-size:0.85em'>全{n_total}個計算済</span>　"
                        f"<span style='color:#60a5fa; font-size:0.85em'>{n_sel}個選択中</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption("全エンジンで記述子は自動計算済みです。テーブルの✅で選択してください。")

                    # 推奨パネル
                    rec = _get_rec2(target_col_sf)
                    if rec:
                        with st.expander(f"💡 「{rec.target_name}」推奨記述子", expanded=True):
                            st.caption(rec.summary)
                            _rec_avail = [d for d in rec.descriptors if d.name in _precalc_df.columns]
                            if _rec_avail:
                                st.dataframe(pd.DataFrame([{
                                    "記述子": d.name, "ライブラリ": d.library,
                                    "意味": d.meaning, "根拠": d.source,
                                } for d in _rec_avail]), use_container_width=True, hide_index=True)
                                _not_yet = [d.name for d in _rec_avail if d.name not in _cur_sel]
                                if _not_yet and st.button(f"🎯 推奨{len(_rec_avail)}件を全て選択", key="btn_rec_sel", type="primary"):
                                    _cur_sel.update(d.name for d in _rec_avail)
                                    st.session_state["adv_desc"] = list(_cur_sel)
                                    st.rerun()
                                elif not _not_yet:
                                    st.success("✅ 推奨記述子は全て選択済みです")
                    else:
                        with st.expander("💡 推奨プリセットを選択", expanded=False):
                            st.caption(f"「{target_col_sf}」に自動マッチするプリセットがありません。下から選んでください。")
                            all_recs = _get_all_recs()
                            _cats = {}
                            for r in all_recs:
                                _cats.setdefault(r.category, []).append(r)
                            for cat_name, cat_recs in _cats.items():
                                st.markdown(f"**{cat_name}**")
                                _bcols = st.columns(min(4, len(cat_recs)))
                                for i, r in enumerate(cat_recs):
                                    with _bcols[i % len(_bcols)]:
                                        if st.button(r.target_name, key=f"preset_{r.target_name}"):
                                            _avail = [d.name for d in r.descriptors if d.name in _precalc_df.columns]
                                            _cur_sel.update(_avail)
                                            st.session_state["adv_desc"] = list(_cur_sel)
                                            st.rerun()

                    # 相関係数計算
                    _corr = {}
                    if target_col_sf in df.columns and pd.api.types.is_numeric_dtype(df[target_col_sf]):
                        try:
                            _corr = _precalc_df.corrwith(df[target_col_sf].loc[_precalc_df.index], method="pearson").abs().to_dict()
                        except Exception:
                            pass

                    _rec_names = set(d.name for d in rec.descriptors) if rec else set()

                    # メタデータ
                    _meta = {}
                    try:
                        from backend.chem import RDKitAdapter as _RDA_m
                        for m in _RDA_m(compute_fp=False).get_descriptors_metadata():
                            _meta[m.name] = {"meaning": getattr(m, 'description', ''), "cat": getattr(m, 'category', '')}
                    except Exception:
                        pass
                    if rec:
                        for d in rec.descriptors:
                            _meta.setdefault(d.name, {})
                            if not _meta[d.name].get("meaning"):
                                _meta[d.name]["meaning"] = d.meaning
                            if not _meta[d.name].get("cat"):
                                _meta[d.name]["cat"] = d.category

                    # テーブルデータ
                    _rows = []
                    # エンジンごとの記述子名マッピング
                    _engine_map = {}
                    try:
                        from backend.chem import RDKitAdapter as _EM_R, MordredAdapter as _EM_M
                        from backend.chem import GroupContribAdapter as _EM_G, MolAIAdapter as _EM_A
                        from backend.chem import XTBAdapter as _EM_X, CosmoAdapter as _EM_C, UniPkaAdapter as _EM_U
                        for _em_cls, _em_nm in [(_EM_R(compute_fp=False),"RDKit"), (_EM_M(selected_only=True),"Mordred"),
                                                 (_EM_G(),"GroupContrib"), (_EM_X(),"XTB"),
                                                 (_EM_C(),"COSMO-RS"), (_EM_U(),"UniPKa")]:
                            if _em_cls.is_available():
                                for _en in _em_cls.get_descriptor_names():
                                    _engine_map.setdefault(_en, _em_nm)
                        for c in _precalc_df.columns:
                            if c.startswith("MolAI_") or c.startswith("molai_"):
                                _engine_map.setdefault(c, "MolAI")
                    except Exception:
                        pass

                    for c in _precalc_df.columns:
                        _r = _corr.get(c, float('nan'))
                        _m = _meta.get(c, {})
                        _rows.append({
                            "✅": c in _cur_sel,
                            "★": "★" if c in _rec_names else "",
                            "記述子名": c,
                            "ソース": _engine_map.get(c, ""),
                            "|r|": round(_r, 3) if pd.notna(_r) else None,
                            "ユニーク数": int(_precalc_df[c].nunique(dropna=True)),
                            "意味": _m.get("meaning", ""),
                            "分類": _m.get("cat", ""),
                        })
                    _rows.sort(key=lambda x: x["|r|"] if x["|r|"] is not None else -1, reverse=True)

                    # エンジン別タブ（comboboxよりタブの方が選択しやすい）
                    _avail_engines = sorted(set(_engine_map.values()))
                    _engine_tab_labels = ["🔍 全て"] + [f"🏷️ {e}" for e in _avail_engines]
                    _engine_tabs = st.tabs(_engine_tab_labels)

                    for _etab_idx, _etab in enumerate(_engine_tabs):
                        with _etab:
                            _flt_engine = "全て" if _etab_idx == 0 else _avail_engines[_etab_idx - 1]

                            # フィルタ（クリック数最小化）
                            _flt = st.pills("表示", ["全て", "選択中", "未選択", "推奨のみ"], default="全て", key=f"df_flt_{_etab_idx}")
                            _fc_a, _fc_b = st.columns([1, 2])
                            with _fc_a:
                                _flt_corr = st.slider("|r|≥", 0.0, 1.0, 0.0, 0.01, key=f"df_flt_corr_{_etab_idx}")
                            with _fc_b:
                                _srch = st.text_input("検索", key=f"df_srch_{_etab_idx}", placeholder="記述子名で絞り込み")

                            _tdf = pd.DataFrame(_rows)
                            if _flt == "選択中":
                                _tdf = _tdf[_tdf["✅"] == True]
                            elif _flt == "未選択":
                                _tdf = _tdf[_tdf["✅"] == False]
                            elif _flt == "推奨のみ":
                                _tdf = _tdf[_tdf["★"] == "★"]
                            if _flt_engine != "全て":
                                _tdf = _tdf[_tdf["ソース"] == _flt_engine]
                            if _flt_corr > 0:
                                _tdf = _tdf[_tdf["|r|"].fillna(0) >= _flt_corr]
                            if _srch:
                                _tdf = _tdf[_tdf["記述子名"].str.contains(_srch, case=False, na=False)]

                            # data_editor（各タブ内に配置）
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
                                    "分類": st.column_config.TextColumn("分類", width="medium"),
                                },
                                disabled=["★", "記述子名", "ソース", "|r|", "ユニーク数", "意味", "分類"],
                                use_container_width=True,
                                hide_index=True,
                                key=f"desc_editor_{_etab_idx}",
                                height=min(700, 40 + len(_tdf) * 35),
                            )

                            # 選択状態をsession_stateに反映
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

                        from backend.chem import RDKitAdapter as _RDKitAdp, XTBAdapter as _XTBAdp
                        from backend.chem import CosmoAdapter as _CosmoAdp, UniPkaAdapter as _UniPkaAdp
                        from backend.chem import GroupContribAdapter as _GCAAdp, MordredAdapter as _MordAdp
                        from backend.chem import MolAIAdapter as _MolAIAdp

                        _engine_list = [
                            ("🧪 RDKit",        _RDKitAdp(),  "~200種",       "🟢 高速"),
                            ("📐 Mordred",       _MordAdp(selected_only=True), "~1,800種", "🟡 中程度"),
                            ("🤖 MolAI (CNN+PCA)", _MolAIAdp(n_components=st.session_state.get("molai_n_components", 6)), "n_comp", "🟡 中程度"),
                            ("⚛️ XTB",           _XTBAdp(),    "~20種",        "🔴 重い"),
                            ("💧 COSMO-RS",      _CosmoAdp(),  "~10種",        "🔴 重い"),
                            ("⚗️ UniPKa",        _UniPkaAdp(), "~5種",         "🟡 中程度"),
                            ("🔩 GroupContrib",  _GCAAdp(),    "~15種",        "🟢 高速"),
                        ]

                        for _ename, _eadp, _edims, _ecost in _engine_list:
                            _eavail = _eadp.is_available()
                            _estatus = "✅ 計算済み" if _eavail else "🚫 未インストール"
                            _ecolor = "#4ade80" if _eavail else "#f87171"
                            st.markdown(
                                f"- {_ename} <span style='color:{_ecolor}'>{_estatus}</span> | {_edims} | {_ecost}",
                                unsafe_allow_html=True,
                            )

                        # MolAI PCA次元設定
                        st.markdown("---")
                        st.markdown("**MolAI CNN + PCA 設定**")
                        _molai_avail = _MolAIAdp(n_components=6).is_available()
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

                        # 再計算
                        st.markdown("---")
                        if st.button("🔄 全記述子を再計算する", key="btn_recalc_adv"):
                            st.session_state["precalc_done"] = False
                            st.session_state["precalc_smiles_df"] = None
                            st.rerun()

                    # ── 記述子セット登録（複数セットを比較試行） ──
                    st.markdown("---")
                    st.markdown("### 📋 記述子セットの登録・比較")
                    st.caption(
                        "現在の選択を名前付きセットとして保存できます。"
                        "複数セットを登録しておくと、パイプライン実行時に全セットを一括比較できます。"
                    )

                    # セットの初期化
                    if "_desc_sets" not in st.session_state:
                        st.session_state["_desc_sets"] = {}

                    _desc_sets = st.session_state["_desc_sets"]

                    # 現在の選択を保存
                    _sc1, _sc2 = st.columns([3, 1])
                    with _sc1:
                        _set_name = st.text_input(
                            "セット名", value="", key="desc_set_name",
                            placeholder="例: 推奨+MolAI、XTBのみ、全記述子 等",
                        )
                    with _sc2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("💾 現在の選択を保存", key="btn_save_set", type="primary"):
                            if _set_name.strip():
                                _desc_sets[_set_name.strip()] = list(st.session_state.get("adv_desc", []))
                                st.session_state["_desc_sets"] = _desc_sets
                                st.success(f"✅ セット「{_set_name.strip()}」を保存しました（{len(st.session_state.get('adv_desc', []))}個）")
                                st.rerun()
                            else:
                                st.warning("セット名を入力してください。")

                    # クイック登録ショートカット
                    with st.expander("⚡ よく使うセットを一括登録", expanded=True):
                        st.caption("エンジンやカテゴリに基づくセットをワンクリックで登録できます。")
                        _qc1, _qc2, _qc3 = st.columns(3)

                        # RDKit記述子のみ
                        with _qc1:
                            if st.button("RDKit記述子のみ", key="quick_rdkit"):
                                try:
                                    from backend.chem import RDKitAdapter as _QR
                                    _qr = _QR(compute_fp=False)
                                    _qr_names = [n for n in _qr.get_descriptor_names() if n in _precalc_df.columns] if _qr.is_available() else []
                                    _desc_sets["RDKit記述子のみ"] = _qr_names
                                    st.session_state["_desc_sets"] = _desc_sets
                                    st.rerun()
                                except Exception:
                                    pass

                        # MolAIのみ
                        with _qc2:
                            if st.button("MolAIのみ", key="quick_molai"):
                                _molai_cols = [c for c in _precalc_df.columns if c.startswith("MolAI_") or c.startswith("molai_")]
                                _desc_sets["MolAIのみ"] = _molai_cols
                                st.session_state["_desc_sets"] = _desc_sets
                                st.rerun()

                        # 推奨記述子のみ
                        with _qc3:
                            if rec:
                                if st.button(f"推奨（{rec.target_name}）", key="quick_rec"):
                                    _rec_n = [d.name for d in rec.descriptors if d.name in _precalc_df.columns]
                                    _desc_sets[f"推奨（{rec.target_name}）"] = _rec_n
                                    st.session_state["_desc_sets"] = _desc_sets
                                    st.rerun()

                        _qc4, _qc5, _qc6 = st.columns(3)
                        with _qc4:
                            if st.button("推奨 + MolAI", key="quick_rec_molai"):
                                _combined = list(set(
                                    [d.name for d in rec.descriptors if d.name in _precalc_df.columns] if rec else []
                                ) | set(c for c in _precalc_df.columns if c.startswith("MolAI_") or c.startswith("molai_")))
                                _desc_sets["推奨 + MolAI"] = _combined
                                st.session_state["_desc_sets"] = _desc_sets
                                st.rerun()

                        with _qc5:
                            if st.button("全記述子", key="quick_all"):
                                _desc_sets["全記述子"] = list(_precalc_df.columns)
                                st.session_state["_desc_sets"] = _desc_sets
                                st.rerun()

                        with _qc6:
                            if st.button("|r| 上位30件", key="quick_top30"):
                                _sorted_by_r = sorted(
                                    [(c, abs(_corr.get(c, 0)) if pd.notna(_corr.get(c, 0)) else 0) for c in _precalc_df.columns],
                                    key=lambda x: x[1], reverse=True
                                )
                                _desc_sets["|r| 上位30件"] = [c for c, _ in _sorted_by_r[:30]]
                                st.session_state["_desc_sets"] = _desc_sets
                                st.rerun()

                    # 登録済みセット一覧
                    if _desc_sets:
                        st.markdown("#### 登録済みセット")
                        for _sname, _sdescs in _desc_sets.items():
                            _sc_a, _sc_b, _sc_c = st.columns([4, 1, 1])
                            with _sc_a:
                                st.markdown(
                                    f"**{_sname}** — {len(_sdescs)}個の記述子",
                                )
                            with _sc_b:
                                if st.button("読込", key=f"load_set_{_sname}"):
                                    st.session_state["adv_desc"] = _sdescs
                                    st.rerun()
                            with _sc_c:
                                if st.button("🗑️", key=f"del_set_{_sname}"):
                                    del _desc_sets[_sname]
                                    st.session_state["_desc_sets"] = _desc_sets
                                    st.rerun()

                        # 一括比較フラグ
                        st.markdown("---")
                        _use_multi = st.checkbox(
                            "🔄 パイプライン実行時に全登録セットを一括比較する",
                            value=st.session_state.get("_use_multi_desc_sets", False),
                            key="chk_multi_desc",
                            help="ONにすると、パイプライン実行時に各記述子セットで個別にモデルを構築・評価し、結果を比較できます。",
                        )
                        st.session_state["_use_multi_desc_sets"] = _use_multi
                        if _use_multi:
                            st.info(f"✅ {len(_desc_sets)}セットを一括比較します。パイプライン実行時間は約{len(_desc_sets)}倍になります。")

        # ══════════════════════════════════════════════════════════════
        # サブタブ3: SMILES特徴量設計
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
                    selected_desc_preview = st.session_state.get("adv_desc", [])
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
                from frontend_streamlit.components.smiles_hover import render_smiles_table
                _smiles_col_prev = smiles_col_eda if smiles_col_eda and smiles_col_eda in df_preview.columns else None
                if _smiles_col_prev:
                    render_smiles_table(df_preview, smiles_col=_smiles_col_prev, max_rows=100, height=420)
                else:
                    st.dataframe(df_preview, use_container_width=True, height=400)

                # 統計量サマリー
                st.markdown("### 📈 統計量サマリー")
                try:
                    _stat_rows = []
                    for _col in df_preview.columns:
                        _s = df_preview[_col]
                        _n_missing = int(_s.isna().sum())
                        _n_total   = len(_s)
                        _missing_rate = _n_missing / _n_total * 100 if _n_total > 0 else 0.0
                        _cardinality  = int(_s.nunique(dropna=True))
                        _row = {
                            "列名": _col,
                            "ユニーク数": _cardinality,
                            "欠損数": _n_missing,
                            "欠損率(%)": f"{_missing_rate:.1f}",
                        }
                        _numeric_s = pd.to_numeric(_s, errors="coerce")
                        if _numeric_s.notna().any():
                            _row["最小値"] = f"{_numeric_s.min():.4g}"
                            _row["最大値"] = f"{_numeric_s.max():.4g}"
                            _row["平均値"] = f"{_numeric_s.mean():.4g}"
                            _row["標準偏差"] = f"{_numeric_s.std():.4g}"
                        else:
                            _row["最小値"] = _row["最大値"] = _row["平均値"] = _row["標準偏差"] = "—"
                        _stat_rows.append(_row)
                    _stat_df = pd.DataFrame(_stat_rows)
                    st.dataframe(_stat_df, use_container_width=True, hide_index=True, height=min(600, 40 + len(_stat_rows) * 35))
                    if added_smiles_cols:
                        st.caption(f"🟢 SMILES展開で追加された列: **{len(added_smiles_cols)}個**")
                except Exception as _e_stat:
                    st.warning(f"統計量計算エラー: {_e_stat}")

                # TypeDetector結果
                st.markdown("---")
                st.markdown("### 🔍 変数型の自動判定結果 (TypeDetector)")
                dr_eda = st.session_state.get("detection_result")
                if dr_eda:
                    if added_smiles_cols:
                        from backend.data.type_detector import TypeDetector
                        tmp_dr = TypeDetector().detect(df_preview.drop(columns=[target_col_eda], errors="ignore") if target_col_eda else df_preview)
                    else:
                        tmp_dr = dr_eda
                    type_data = []
                    for c in df_preview.columns:
                        t = "❓不明"
                        if target_col_eda and c == target_col_eda:
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
                    st.dataframe(pd.DataFrame(type_data), use_container_width=True, hide_index=True)
                else:
                    st.info("型判定結果がありません")

                # EDAタブでは記述子選択は行わない（SMILES特徴量設計タブで設定）
                if st.session_state.get("smiles_col"):
                    if st.session_state.get("precalc_done"):
                        n_sel = len(st.session_state.get("adv_desc", []))
                        st.info(f"🔬 記述子の選択は「⚗️ SMILES特徴量設計」タブで行えます。現在 **{n_sel}件** 選択中。")
                    else:
                        st.info("💡 「⚗️ SMILES特徴量設計」タブで記述子を計算・選択してください。")

                # ══════════════════════════════════════════════════════════════
                # 🧹 データクリーニング
                # ══════════════════════════════════════════════════════════════
                st.markdown("---")
                st.markdown(
                    '<div class="section-header">🧹 データクリーニング</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "EDAの分析結果を見ながら、データを直接加工できます。"
                    "変更は即座に反映され、Undoで元に戻せます。"
                )

                from backend.data.data_cleaner import (
                    drop_columns, drop_rows_with_missing,
                    remove_constant_columns, clip_outliers,
                    remove_duplicates, preview_missing_impact,
                    preview_outlier_impact, get_cleaning_summary,
                )

                # 初期化: Undo用スタック & ログ
                if "_df_history" not in st.session_state:
                    st.session_state["_df_history"] = []
                if "_cleaning_log" not in st.session_state:
                    st.session_state["_cleaning_log"] = []

                # ── 現在のデータ概要 ──
                _cs = get_cleaning_summary(df)
                _cs_c1, _cs_c2, _cs_c3, _cs_c4 = st.columns(4)
                with _cs_c1:
                    st.metric("行数", f"{len(df):,}")
                with _cs_c2:
                    st.metric("列数", f"{df.shape[1]}")
                with _cs_c3:
                    st.metric("欠損率", f"{_cs['total_missing_rate']:.1%}")
                with _cs_c4:
                    _issues = _cs["n_const_cols"] + _cs["n_dup_rows"] + _cs["n_all_missing_cols"]
                    st.metric("検出問題", f"{_issues}件")

                # ── 操作履歴 & Undo ──
                _log = st.session_state.get("_cleaning_log", [])
                if _log:
                    with st.expander(f"操作履歴（{len(_log)}件）", expanded=False):
                        for _li, _la in enumerate(reversed(_log)):
                            st.markdown(
                                f'<div class="card" style="padding:0.5rem 0.8rem; margin-bottom:0.3rem;">'
                                f'<span style="color:#00d4ff;">●</span> '
                                f'<b>{_la.description}</b> '
                                f'<span style="font-size:0.78rem; color:#8888aa;">'
                                f'({_la.rows_before}→{_la.rows_after}行, '
                                f'{_la.cols_before}→{_la.cols_after}列)</span></div>',
                                unsafe_allow_html=True,
                            )

                    if st.button("⏪ 直前の操作をUndoする", key="undo_cleaning",
                                 use_container_width=True):
                        _hist = st.session_state.get("_df_history", [])
                        if _hist:
                            st.session_state["df"] = _hist.pop()
                            st.session_state["_df_history"] = _hist
                            _log_list = st.session_state.get("_cleaning_log", [])
                            if _log_list:
                                _log_list.pop()
                                st.session_state["_cleaning_log"] = _log_list
                            st.success("⏪ Undo完了！直前の操作を取り消しました。")
                            st.rerun()
                        else:
                            st.warning("Undo履歴がありません。")

                # ── クリーニングアクション ──
                _clean_tabs = st.tabs([
                    "列の除外",
                    "欠損行削除",
                    "定数列除去",
                    "外れ値クリップ",
                    "重複行除去",
                ])

                # --- 1. 列の除外 ---
                with _clean_tabs[0]:
                    _target_col_clean = st.session_state.get("target_col")
                    _smiles_col_clean = st.session_state.get("smiles_col")
                    _protect = [c for c in [_target_col_clean, _smiles_col_clean] if c]
                    _droppable = [c for c in df.columns if c not in _protect]

                    if not _droppable:
                        st.info("除外可能な列がありません。")
                    else:
                        _cols_to_drop = st.multiselect(
                            "除外する列を選択",
                            options=_droppable,
                            key="clean_drop_cols",
                            help="目的変数・SMILES列は保護されます",
                        )
                        if _cols_to_drop:
                            st.caption(f"選択中: {len(_cols_to_drop)}列 → 残り{df.shape[1] - len(_cols_to_drop)}列")
                            if st.button("選択列を除外する", key="btn_drop_cols",
                                         type="primary", use_container_width=True):
                                st.session_state["_df_history"].append(df.copy())
                                _new_df, _action = drop_columns(df, _cols_to_drop)
                                st.session_state["df"] = _new_df
                                st.session_state["_cleaning_log"].append(_action)
                                st.success(f"✅ {_action.description}")
                                st.rerun()

                # --- 2. 欠損行削除 ---
                with _clean_tabs[1]:
                    _miss_threshold = st.slider(
                        "欠損率の閾値",
                        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                        key="clean_miss_thresh",
                        help="各行が持つ欠損率がこの値以上の行を削除（0.0=欠損が1つでもあれば削除）",
                    )
                    _miss_impact = preview_missing_impact(df, threshold=_miss_threshold)
                    if _miss_impact > 0:
                        st.warning(f"{_miss_impact:,}行（全体の{_miss_impact/len(df):.1%}）が削除されます。")
                    else:
                        st.success("削除対象の行はありません。")

                    if _miss_impact > 0:
                        if st.button("欠損行を削除する", key="btn_drop_miss",
                                     type="primary", use_container_width=True):
                            st.session_state["_df_history"].append(df.copy())
                            _new_df, _action = drop_rows_with_missing(df, threshold=_miss_threshold)
                            st.session_state["df"] = _new_df
                            st.session_state["_cleaning_log"].append(_action)
                            st.success(f"✅ {_action.description}")
                            st.rerun()

                # --- 3. 定数列除去 ---
                with _clean_tabs[2]:
                    if _cs["n_const_cols"] > 0:
                        st.warning(
                            f"定数列が **{_cs['n_const_cols']}列** 見つかりました: "
                            f"{', '.join(_cs['const_cols'][:8])}"
                            + (f" 他{_cs['n_const_cols']-8}列" if _cs['n_const_cols'] > 8 else "")
                        )
                        if st.button("定数列を一括除去する", key="btn_rm_const",
                                     type="primary", use_container_width=True):
                            st.session_state["_df_history"].append(df.copy())
                            _new_df, _action = remove_constant_columns(df)
                            st.session_state["df"] = _new_df
                            st.session_state["_cleaning_log"].append(_action)
                            st.success(f"✅ {_action.description}")
                            st.rerun()
                    else:
                        st.success("定数列は見つかりませんでした。")

                # --- 4. 外れ値クリッピング ---
                with _clean_tabs[3]:
                    _num_cols = list(df.select_dtypes(include="number").columns)
                    if not _num_cols:
                        st.info("数値列がないため外れ値クリッピングは利用できません。")
                    else:
                        _iqr_mult = st.slider(
                            "IQR倍率",
                            min_value=1.0, max_value=5.0, value=1.5, step=0.1,
                            key="clean_iqr_mult",
                            help="Q1 - IQR*倍率 〜 Q3 + IQR*倍率 の範囲にクリップ",
                        )
                        _outlier_cols = st.multiselect(
                            "対象列（空=全数値列）",
                            options=_num_cols,
                            key="clean_outlier_cols",
                        )
                        _target_outlier_cols = _outlier_cols if _outlier_cols else None
                        _outlier_preview = preview_outlier_impact(
                            df, iqr_multiplier=_iqr_mult, columns=_target_outlier_cols
                        )
                        _total_outliers = sum(_outlier_preview.values())
                        if _total_outliers > 0:
                            st.warning(
                                f"{len(_outlier_preview)}列で計{_total_outliers:,}値の外れ値を検出。"
                            )
                            with st.expander("列別の詳細"):
                                for _oc, _on in sorted(_outlier_preview.items(), key=lambda x: -x[1]):
                                    st.markdown(f"- **{_oc}**: {_on}値")
                            if st.button("外れ値をクリップする", key="btn_clip_outliers",
                                         type="primary", use_container_width=True):
                                st.session_state["_df_history"].append(df.copy())
                                _new_df, _action = clip_outliers(
                                    df, iqr_multiplier=_iqr_mult, columns=_target_outlier_cols
                                )
                                st.session_state["df"] = _new_df
                                st.session_state["_cleaning_log"].append(_action)
                                st.success(f"✅ {_action.description}")
                                st.rerun()
                        else:
                            st.success("外れ値は検出されませんでした。")

                # --- 5. 重複行除去 ---
                with _clean_tabs[4]:
                    if _cs["n_dup_rows"] > 0:
                        st.warning(
                            f"**{_cs['n_dup_rows']:,}件** の重複行が見つかりました。"
                        )
                        if st.button("重複行を除去する", key="btn_rm_dup",
                                     type="primary", use_container_width=True):
                            st.session_state["_df_history"].append(df.copy())
                            _new_df, _action = remove_duplicates(df)
                            st.session_state["df"] = _new_df
                            st.session_state["_cleaning_log"].append(_action)
                            st.success(f"✅ {_action.description}")
                            st.rerun()
                    else:
                        st.success("重複行は見つかりませんでした。")



        # ══════════════════════════════════════════════════════════════
        # サブタブ5: パイプライン設計
        # ══════════════════════════════════════════════════════════════
        with ds_tab5:
            df = st.session_state.get("df")
            if df is None:
                st.warning("⚠️ まず「📂 データ読込」タブでデータを読み込んでください。")
            else:
                # CV設定（コンパクト）
                with st.expander("⚙️ 交差検証・その他の基本設定", expanded=False):
                    c_cv1, c_cv2, c_cv3 = st.columns(3)
                    with c_cv1:
                        cv_folds  = st.slider("CV分割数", 2, 10, st.session_state.get("_adv_cv_folds", 5), key="adv_cv")
                    with c_cv2:
                        timeout   = st.slider("タイムアウト(秒)", 30, 3600, st.session_state.get("_adv_timeout", 300), key="adv_to")
                    with c_cv3:
                        scaler    = st.selectbox("スケーラー", ["auto","standard","robust","minmax","none"], key="adv_sc")
                    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
                    with c_p1: do_eda  = st.checkbox("EDA", value=True, key="adv_eda")
                    with c_p2: do_prep = st.checkbox("前処理", value=True, key="adv_prep")
                    with c_p3: do_eval = st.checkbox("評価", value=True, key="adv_eval")
                    with c_p4: do_pca  = st.checkbox("PCA", value=True, key="adv_pca")
                    do_shap = st.checkbox("SHAP解析", value=True, key="adv_shap")

                # モデル選択（コンパクト）
                with st.expander("🤖 使用するモデルを選ぶ", expanded=True):
                    from backend.models.factory import list_models, get_default_automl_models, get_model_registry
                    import inspect
                    _tmp_task = st.session_state.get("task", "auto")
                    if _tmp_task == "auto":
                        _tc = st.session_state.get("target_col")
                        _tmp_task = "regression" if (_tc and pd.api.types.is_float_dtype(df[_tc])) else "classification"

                    available_models = list_models(task=_tmp_task, available_only=True)
                    default_models   = get_default_automl_models(task=_tmp_task)

                    def _get_category(mkey, mname):
                        k = mkey.lower() + mname.lower()
                        if any(x in k for x in ["linear","ridge","lasso","elastic","logistic","ard","huber","theilsen","ransac","pls","sgd"]): return "線形系"
                        if any(x in k for x in ["svr","svc","support","rbf","kernel","gaussian"]): return "カーネル系"
                        if any(x in k for x in ["tree","forest","boost","gbm","gradient"]): return "決定木系"
                        return "その他"

                    categories = {"線形系": [], "カーネル系": [], "決定木系": [], "その他": []}
                    for m in available_models:
                        categories[_get_category(m["key"], m["name"])].append(m)

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

                # パイプライン全設定UI （7ステップタブ）
                st.markdown("### ⚙️ パイプライン構成の設定")
                st.caption(
                    "各ステップで複数選択すると全組み合わせを自動評価します。\n\n"
                    "**組合せ数** = `cont_imp × scaler × cat_imp × cat_enc × bin_imp × bin_enc × engineer × selector × estimator`"
                )
                try:
                    from frontend_streamlit.components.pipeline_config_ui import render_pipeline_config_ui as _render_pg_ui
                except ImportError:
                    from components.pipeline_config_ui import render_pipeline_config_ui as _render_pg_ui

                _all_cols_pg = list(df.columns) if df is not None else []
                _task_pg     = st.session_state.get("task", "regression")
                _target_pg   = st.session_state.get("target_col")
                _smiles_pg   = st.session_state.get("smiles_col")

                _pg_cfg = _render_pg_ui(
                    all_cols=_all_cols_pg,
                    target_col=_target_pg,
                    smiles_col=_smiles_pg,
                    task=_task_pg,
                )
                st.session_state["_pipeline_full_config"] = _pg_cfg

                # ── per-feature 単調性制約 UI ───────────────────────────
                st.markdown("---")
                with st.expander("📐 単調性制約（特徴量ごとに設定）", expanded=False):
                    try:
                        from frontend_streamlit.components.pipeline_config_ui import render_monotonic_constraints_ui as _rmui
                    except ImportError:
                        from components.pipeline_config_ui import render_monotonic_constraints_ui as _rmui

                    # 数値列（目的変数・SMILES除く）のみ対象
                    _feat_cols_mono = [
                        c for c in _all_cols_pg
                        if c not in (_target_pg, _smiles_pg)
                        and df is not None
                        and pd.api.types.is_numeric_dtype(df[c])
                    ] if df is not None else []

                    if _feat_cols_mono:
                        _mono_constraints = _rmui(
                            feature_cols=_feat_cols_mono,
                            n_cols=4,
                        )
                        st.session_state["_monotonic_constraints_dict"] = _mono_constraints
                    else:
                        st.info("ℹ️ 数値列が見つかりません。データを読み込んでから設定してください。")
                        st.session_state["_monotonic_constraints_dict"] = {}

                # 詳細設定の値をセッションに保存
                st.session_state["_adv"] = dict(
                    cv_folds=cv_folds, models=selected_models, timeout=timeout,
                    scaler=scaler,
                    do_eda=do_eda, do_prep=do_prep, do_eval=do_eval,
                    do_pca=do_pca, do_shap=do_shap,
                    selected_descriptors=st.session_state.get("adv_desc", []),
                )

        # ── 実行ボタン（データがある場合に常時表示） ─────────────────
        df = st.session_state.get("df")
        if df is not None:
            st.markdown("---")
            existing_result = st.session_state.get("pipeline_result")

            if existing_result is None:
                c_l, c_m, c_r = st.columns([1, 3, 1])
                with c_m:
                    if st.button(
                        "🚀 解析開始  （EDA → AutoML → 評価 → SHAP まで自動実行）",
                        use_container_width=True,
                        key="home_run",
                        type="primary",
                    ):
                        adv = st.session_state.get("_adv", {})

                        # リーケージチェック自動実行（数値列が2列以上ある場合）
                        _target_lk = st.session_state.get("target_col")
                        _smiles_lk = st.session_state.get("smiles_col")
                        _excl_lk = st.session_state.get("col_role_exclude", [])
                        _feat_lk = [c for c in df.columns if c not in [_target_lk, _smiles_lk] + _excl_lk]
                        _X_lk = df[_feat_lk].select_dtypes(include="number")
                        _leakage_group_labels = None
                        _leakage_recommended_cv = None
                        if _X_lk.shape[1] >= 2 and len(df) <= 5000:
                            try:
                                from backend.data.leakage_detector import detect_leakage
                                _lk_report = detect_leakage(_X_lk, df[_target_lk], method="auto")
                                if _lk_report.risk_level in ("medium", "high") and _lk_report.group_labels is not None:
                                    _leakage_group_labels = _lk_report.group_labels
                                    _leakage_recommended_cv = _lk_report.recommended_cv
                                st.session_state["leakage_report"] = _lk_report
                            except Exception:
                                pass

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
                            monotonic_constraints_dict = st.session_state.get("_monotonic_constraints_dict", {}),
                            leakage_group_labels = _leakage_group_labels,
                            leakage_recommended_cv = _leakage_recommended_cv,
                            cv_groups_col = st.session_state.get("col_role_group"),
                            exclude_cols = st.session_state.get("col_role_exclude", []),
                            col_role_time = st.session_state.get("col_role_time"),
                            sample_weight_col = st.session_state.get("col_role_weight"),
                        )
                        st.session_state["active_tab_idx"] = 1
                        st.rerun()
            else:
                ar = st.session_state.get("automl_result")
                if ar:
                    st.success(
                        f"✅ 解析完了！ 最良モデル: **{ar.best_model_key}** | "
                        f"スコア: `{ar.best_score:.4f}` | "
                        f"所要時間: {existing_result.elapsed:.1f}秒"
                    )
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("📊 結果タブへ", use_container_width=True, key="view_res"):
                        st.session_state["active_tab_idx"] = 2
                        st.rerun()
                with cc2:
                    if st.button("🔄 別データで再解析", use_container_width=True, key="reset"):
                        for k in ["df","file_name","automl_result","pipeline_result",
                                  "target_col","detection_result","step_eda_done",
                                  "step_preprocess_done","_run_config"]:
                            st.session_state[k] = None if k not in ("step_eda_done","step_preprocess_done") else False
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
            c_re1, c_re2 = st.columns([2, 1])
            with c_re2:
                if st.button("🔄 設定を変えて再解析", key="tab3_rerun", use_container_width=True):
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

            # ── 結果確認サブタブ ────────────────────────────────────────
            res_tab1, res_tab2, res_tab_bo = st.tabs(["📈 モデル評価", "🔬 モデル解釈性", "🎯 実験計画(BO)"])

            with res_tab1:
                try:
                    from frontend_streamlit.pages.pipeline import evaluation_page
                    evaluation_page.render()
                except Exception as e:
                    st.error(f"❌ 評価ページの読み込みエラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            with res_tab2:
                try:
                    from frontend_streamlit.components.interpretability_ui import render_interpretability_ui
                    ar = st.session_state.get("automl_result")
                    if ar is None:
                        st.info("解析結果がありません。")
                    else:
                        _model  = getattr(ar, "best_pipeline", None) or getattr(ar, "best_model", None)
                        _X_test = getattr(ar, "X_test", None) or getattr(ar, "X_train", None)
                        _y_test = getattr(ar, "y_test", None) or getattr(ar, "y_train", None)
                        _target = st.session_state.get("target_col")
                        _smiles = st.session_state.get("smiles_col")
                        _df_all = st.session_state.get("df")

                        if _X_test is not None and hasattr(_X_test, "columns"):
                            _feat_names = list(_X_test.columns)
                        elif _df_all is not None:
                            _feat_names = [c for c in _df_all.columns if c not in (_target, _smiles)]
                        else:
                            _feat_names = [f"x{i}" for i in range(
                                _X_test.shape[1] if _X_test is not None and hasattr(_X_test, "shape") else 10)]

                        if _model is None:
                            st.warning("モデルオブジェクトが取得できませんでした。")
                        elif _X_test is None:
                            st.warning("テストデータが取得できませんでした。")
                        else:
                            render_interpretability_ui(
                                model=_model,
                                X=_X_test,
                                y=_y_test,
                                feature_names=_feat_names,
                                task=getattr(ar, "task", "regression"),
                            )
                except Exception as e:
                    st.error(f"❌ 解釈性UIの読み込みエラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())


            with res_tab_bo:
                from frontend_streamlit.components.bayesian_opt_ui import render_bayesian_opt_ui
                render_bayesian_opt_ui()

