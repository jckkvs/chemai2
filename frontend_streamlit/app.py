"""
frontend_streamlit/app.py

ChemAI ML Studio - Streamlit メインアプリ
Upload → Select → ワンクリック解析。初心者向けの隠蔽設定と専門家向けの詳細設定を兼備。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

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

# ── サイドバー（ステータス表示 + 詳細ツール） ─────────────────
with st.sidebar:
    st.markdown("## ⚗️ ChemAI ML Studio")

    # ホームに戻るボタン
    if st.button("🏠 ホーム", use_container_width=True, key="go_home"):
        st.session_state["page"] = "home"
        st.rerun()

    st.markdown("---")

    # ── データステータス ──────────────────────────────────
    has_data   = st.session_state["df"] is not None
    has_target = bool(st.session_state.get("target_col"))
    has_result = st.session_state["automl_result"] is not None

    st.markdown(
        '<div style="font-size:0.75rem; color:#8888aa; text-transform:uppercase; '
        'letter-spacing:0.05em; margin-bottom:8px;">現在の状態</div>',
        unsafe_allow_html=True,
    )

    def _status_line(dot_cls: str, text: str) -> None:
        st.markdown(
            f'<div style="font-size:0.82rem; margin:4px 0;">'
            f'<span class="{dot_cls}"></span>{text}</div>',
            unsafe_allow_html=True,
        )

    if has_data:
        _df = st.session_state["df"]
        _status_line("status-dot-green",
                     f"データ: {st.session_state['file_name']}")
        _status_line("status-dot-green",
                     f"{_df.shape[0]:,}行 × {_df.shape[1]}列")
        if has_target:
            _status_line("status-dot-green",
                         f"目的変数: {st.session_state['target_col']}")
    else:
        _status_line("status-dot-gray", "データ未読み込み")

    if has_result:
        r = st.session_state["automl_result"]
        _status_line("status-dot-green",
                     f"最良モデル: {r.best_model_key}")
        _status_line("status-dot-green",
                     f"スコア: {r.best_score:.4f}")
    elif has_data:
        _status_line("status-dot-yellow", "未解析（解析開始を押してください）")
    else:
        _status_line("status-dot-gray", "解析未実行")

    # ── 詳細ツール（専門家向け） ──────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem; color:#8888aa; text-transform:uppercase; '
        'letter-spacing:0.05em; margin-bottom:8px;">詳細ツール</div>',
        unsafe_allow_html=True,
    )
    expert_pages = [
        ("📂", "データ詳細",     "data_load",    has_data),
        ("🔍", "EDA 詳細",       "eda",           has_data),
        ("⚙️", "前処理設定",     "preprocess",    has_data),
        ("📊", "モデル評価",     "evaluation",    has_result),
        ("📐", "次元削減",       "dim_reduction", has_data),
        ("💡", "SHAP 解釈",      "interpret",     has_result),
        ("🧬", "化合物解析",     "chem",          True),
    ]
    for icon, label, pkey, enabled in expert_pages:
        cur = st.session_state["page"] == pkey
        if enabled:
            btn_label = f"{icon} {label}"
            if cur:
                btn_label = f"▶ {btn_label}"
            if st.button(btn_label, key=f"exp_{pkey}", use_container_width=True):
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
# Home ページ（アップロード + 解析設定 + ワンクリック実行）
# ===============================================================
if page == "home":
    st.markdown('<div class="hero-title">⚗️ ChemAI ML Studio</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">ファイルをアップロードして目的変数を選ぶだけ。'
        'あとは自動でEDA・机械学習・評価・SHAP解析まで完結します。</div>',
        unsafe_allow_html=True,
    )

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
        c1, c2, c3 = st.columns(3)
        def _make_sample(name: str, df: pd.DataFrame) -> None:
            st.session_state["df"]          = df
            st.session_state["file_name"]   = name
            st.session_state["automl_result"]  = None
            st.session_state["pipeline_result"] = None
            detector = TypeDetector()
            dr = detector.detect(df)
            st.session_state["detection_result"] = dr
            if dr.smiles_columns:
                st.session_state["smiles_col"] = dr.smiles_columns[0]
            # デフォルト目的変数: 最終列
            st.session_state["target_col"] = df.columns[-1]

        with c1:
            if st.button("🧪 回帰サンプル", use_container_width=True, key="demo_reg"):
                np.random.seed(42); n = 200
                _make_sample("sample_regression.csv", pd.DataFrame({
                    "temperature": np.random.uniform(20, 80, n),
                    "pressure":    np.random.exponential(5, n),
                    "catalyst":    np.random.choice(["A型","B型","C型"], n),
                    "time_h":      np.random.uniform(1, 24, n),
                    "is_active":   np.random.randint(0, 2, n),
                    "yield":       np.random.randn(n) * 10 + 75,
                }))
                st.rerun()
        with c2:
            if st.button("🏷️ 分類サンプル", use_container_width=True, key="demo_cls"):
                np.random.seed(42); n = 200
                _make_sample("sample_classification.csv", pd.DataFrame({
                    "feature_1": np.random.randn(n),
                    "feature_2": np.random.randn(n),
                    "category":  np.random.choice(["低","中","高"], n),
                    "numeric":   np.random.randint(1, 100, n),
                    "label":     np.random.randint(0, 2, n),
                }))
                st.rerun()
        with c3:
            if st.button("🧬 SMILESサンプル", use_container_width=True, key="demo_smi"):
                smis = ["CCO","C","CC","CCC","CCCC","c1ccccc1",
                        "c1ccccc1O","c1ccccc1N","CC(=O)O","CCN",
                        "c1ccc(O)cc1","CC(C)O","CCOCC","ClCCl","BrC"]
                sols = [-0.77,0.0,-0.63,-1.5,-2.1,-1.9,-0.5,-0.8,-0.3,
                        -1.1,-0.7,-0.9,-1.3,-1.0,-0.4]
                _make_sample("sample_smiles.csv", pd.DataFrame({
                    "smiles":      smis * 10,
                    "solubility":  sols * 10,
                }))
                st.rerun()

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
            detector = TypeDetector()
            dr = detector.detect(df_new)
            st.session_state["detection_result"] = dr
            if dr.smiles_columns:
                st.session_state["smiles_col"] = dr.smiles_columns[0]
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
            st.session_state["target_col"] = target
        with col_task:
            task_opt = st.selectbox(
                "📋 タスク種別",
                ["auto（自動）", "regression（回帰）", "classification（分類）"],
                key="home_task",
            )
            st.session_state["task"] = task_opt.split("（")[0]

        # ── 詳細設定（折り畳み：初心者には見えない） ────────
        with st.expander("⚙️ 詳細設定（任意）", expanded=False):
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("**ML設定**")
                cv_folds   = st.slider("CV分割数", 2, 10, 5, key="adv_cv")
                max_models = st.slider("試すモデル数", 1, 15, 8, key="adv_max")
                timeout    = st.slider("タイムアウト(秒)", 30, 3600, 300, key="adv_to")
            with cb:
                st.markdown("**前処理設定**")
                scaler    = st.selectbox("数値スケーラー",
                    ["auto","standard","robust","minmax","none"], key="adv_sc")
                smiles_raw = st.selectbox("SMILES列",
                    ["なし"] + df.columns.tolist(), key="adv_sm")
                st.session_state["smiles_col"] = None if smiles_raw == "なし" else smiles_raw
            with cc:
                st.markdown("**実行フェーズ**")
                do_eda  = st.checkbox("EDA", value=True, key="adv_eda")
                do_prep = st.checkbox("前処理確認", value=True, key="adv_prep")
                do_eval = st.checkbox("評価", value=True, key="adv_eval")
                do_pca  = st.checkbox("次元削減(PCA)", value=True, key="adv_pca")
                do_shap = st.checkbox("SHAP解析", value=True, key="adv_shap")

            # 詳細設定の値をセッションに保存
            st.session_state["_adv"] = dict(
                cv_folds=cv_folds, max_models=max_models, timeout=timeout,
                scaler=scaler,
                do_eda=do_eda, do_prep=do_prep, do_eval=do_eval,
                do_pca=do_pca, do_shap=do_shap,
            )

        # ── 実行ボタン（主役） ───────────────────────────────
        st.markdown("")
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
                    # 詳細設定がない場合はデフォルト値を使用
                    adv = st.session_state.get("_adv", {})
                    st.session_state["_run_config"] = dict(
                        target_col = st.session_state["target_col"],
                        smiles_col = st.session_state.get("smiles_col"),
                        task       = st.session_state.get("task", "auto"),
                        cv_folds   = adv.get("cv_folds", 5),
                        max_models = adv.get("max_models", 8),
                        timeout    = adv.get("timeout", 300),
                        scaler     = adv.get("scaler", "auto"),
                        do_eda     = adv.get("do_eda", True),
                        do_prep    = adv.get("do_prep", True),
                        do_eval    = adv.get("do_eval", True),
                        do_pca     = adv.get("do_pca", True),
                        do_shap    = adv.get("do_shap", True),
                    )
                    st.session_state["page"] = "automl"
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

            # データプレビュー（小さめ）
            with st.expander("📄 データプレビュー", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                if st.button("📊 結果を見る", use_container_width=True, key="view_res"):
                    st.session_state["page"] = "automl"
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

# ===============================================================
# AutoML 実行ページ（結果表示を兼ねる）
# ===============================================================
elif page == "automl":
    from frontend_streamlit.pages import automl_page
    # _run_config があれば渡す（ホームからの一括実行）
    rc = st.session_state.pop("_run_config", None)
    if rc is not None:
        automl_page.render(run_config=rc)
    else:
        automl_page.render()

# ===============================================================
# 詳細ツール群（専門家向け）
# ===============================================================
elif page == "data_load":
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
