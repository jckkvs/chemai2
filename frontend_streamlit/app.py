"""
frontend_streamlit/app.py

ChemAI ML Studio - Streamlit メインアプリ
AutoML と専門家モードを切り替えられるマルチページ構成のML GUIアプリ。
"""
from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

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
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* サイドバー */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: white;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #b0afd0 !important;
}

/* メイン背景 */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0f0;
}

/* カード */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* ヒーローセクション */
.hero-title {
    background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5rem;
}

.hero-sub {
    text-align: center;
    color: #8888aa;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* バッジ */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-blue { background: rgba(0,150,255,0.2); color: #4db8ff; border: 1px solid #4db8ff; }
.badge-purple { background: rgba(123,47,247,0.2); color: #c084fc; border: 1px solid #c084fc; }
.badge-green { background: rgba(0,200,100,0.2); color: #4ade80; border: 1px solid #4ade80; }
.badge-orange { background: rgba(255,160,0,0.2); color: #fbbf24; border: 1px solid #fbbf24; }

/* メトリクスカード */
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.85rem;
    color: #8888aa;
    margin-top: 0.3rem;
}

/* ステップ進捗 */
.step-bar {
    display: flex;
    justify-content: space-around;
    margin: 1.5rem 0;
}
.step-item {
    text-align: center;
    flex: 1;
}
.step-circle {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 6px;
    font-weight: 700;
    font-size: 0.9rem;
}
.step-active { background: linear-gradient(135deg,#00d4ff,#7b2ff7); color:white; }
.step-done { background: #4ade80; color: #0f0c29; }
.step-pending { background: rgba(255,255,255,0.1); color: #666; border: 1px solid #333; }
.step-label { font-size: 0.7rem; color: #8888aa; }

/* Streamlitデフォルト上書き */
.stButton>button {
    background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 2rem;
    transition: all 0.3s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,212,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# ── セッションステート初期化 ──────────────────────────────────
def _init_session():
    defaults = {
        "mode": "AutoML",
        "page": "home",
        "df": None,
        "file_name": None,
        "detection_result": None,
        "automl_result": None,
        "pipeline_result": None,   # フルパイプライン結果
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

# ── サイドバー ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ ChemAI ML Studio")
    st.markdown("---")

    # モード切替
    mode = st.radio(
        "🔧 実行モード",
        ["AutoML", "専門家モード"],
        index=0 if st.session_state["mode"] == "AutoML" else 1,
        key="mode_radio",
    )
    st.session_state["mode"] = mode
    st.markdown("---")

    # ── ワークフローステップナビゲーション ──────────────────
    has_data   = st.session_state["df"] is not None
    has_target = bool(st.session_state.get("target_col"))
    has_result = st.session_state["automl_result"] is not None

    # ステップ定義: (ステップ番号, アイコン, ラベル, page_key, 有効条件, 完了条件)
    workflow_steps = [
        (1, "📂", "データ読み込み", "data_load", True,          has_data),
        (2, "🔍", "EDA",            "eda",        has_data,      st.session_state["step_eda_done"]),
        (3, "⚙️", "前処理設定",     "preprocess", has_data,      st.session_state["step_preprocess_done"]),
        (4, "🤖", "AutoML 実行",    "automl",     has_data and has_target, has_result),
    ]

    st.markdown(
        '<div style="font-size:0.75rem; color:#8888aa; margin-bottom:6px; '
        'text-transform:uppercase; letter-spacing:0.05em;">ワークフロー</div>',
        unsafe_allow_html=True,
    )

    for step_num, icon, label, pkey, enabled, done in workflow_steps:
        cur = st.session_state["page"] == pkey
        if done:
            badge = '<span style="color:#4ade80; font-size:0.7rem;">✓ 完了</span>'
        elif cur:
            badge = '<span style="color:#00d4ff; font-size:0.7rem;">▶ 実行中</span>'
        else:
            badge = ""

        if enabled:
            label_html = (
                f'<span style="color:{"#00d4ff" if cur else "#e0e0f0"}; font-weight:{"700" if cur else "400"};">'
                f'STEP {step_num}  {icon} {label}</span>'
            )
            btn_key = f"wf_{pkey}"
            col_l, col_r = st.columns([4, 2])
            with col_l:
                st.markdown(label_html, unsafe_allow_html=True)
            with col_r:
                st.markdown(badge, unsafe_allow_html=True)
            if st.button("→", key=btn_key, use_container_width=True):
                st.session_state["page"] = pkey
                st.rerun()
        else:
            st.markdown(
                f'<span style="color:#444466; font-size:0.9rem;">STEP {step_num}  {icon} {label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)

    # ── 詳細分析ツール ──────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem; color:#8888aa; margin-bottom:6px; '
        'text-transform:uppercase; letter-spacing:0.05em;">詳細分析ツール</div>',
        unsafe_allow_html=True,
    )
    detail_pages = [
        ("📊", "モデル評価",  "evaluation",    has_result),
        ("📐", "次元削減",    "dim_reduction", has_data),
        ("💡", "SHAP 解釈",   "interpret",     has_result),
        ("🧬", "化合物解析",  "chem",          True),
    ]
    for icon, label, pkey, enabled in detail_pages:
        cur = st.session_state["page"] == pkey
        color = "#00d4ff" if cur else ("#b0afd0" if enabled else "#444466")
        if enabled and st.button(
            f"{icon} {label}", key=f"nav_{pkey}", use_container_width=True
        ):
            st.session_state["page"] = pkey
            st.rerun()
        elif not enabled:
            st.markdown(
                f'<span style="color:#444466; font-size:0.9rem;">{icon} {label}</span>',
                unsafe_allow_html=True,
            )

    # ── データ状態サマリー ──────────────────────────────────
    st.markdown("---")
    if has_data:
        _df = st.session_state["df"]
        st.markdown(
            f'<div style="font-size:0.78rem; color:#4ade80;">'
            f'✅ {st.session_state["file_name"]}<br>'
            f'📏 {_df.shape[0]:,}行 × {_df.shape[1]}列'
            f'{"<br>🎯 目的変数: " + st.session_state["target_col"] if has_target else ""}'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.78rem; color:#fbbf24;">⚠️ データ未読み込み</div>',
            unsafe_allow_html=True,
        )
    if has_result:
        r = st.session_state["automl_result"]
        st.markdown(
            f'<div style="font-size:0.78rem; color:#c084fc; margin-top:0.5rem;">'
            f'🏆 {r.best_model_key}<br>📈 {r.best_score:.4f}</div>',
            unsafe_allow_html=True,
        )

# ── ページルーティング ────────────────────────────────────────
page = st.session_state["page"]

# ===============================================================
# Home ページ
# ===============================================================
if page == "home":
    st.markdown('<div class="hero-title">⚗️ ChemAI ML Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AutoMLと専門家モードで化学・材料・バイオデータを解析</div>',
                unsafe_allow_html=True)

    # 機能バッジ
    st.markdown("""
<div style="text-align:center; margin-bottom:2rem;">
<span class="badge badge-blue">🤖 AutoML</span>
<span class="badge badge-purple">🧬 SMILES対応</span>
<span class="badge badge-green">📊 SHAP/SRI解析</span>
<span class="badge badge-orange">🔬 化合物記述子</span>
<span class="badge badge-blue">⚗️ RDKit統合</span>
<span class="badge badge-purple">📈 MLflow追跡</span>
<span class="badge badge-green">🔄 クロスバリデーション</span>
<span class="badge badge-orange">🎯 ハイパーパラメータ最適化</span>
</div>""", unsafe_allow_html=True)

    # ワークフォローステップ
    st.markdown("### 🚀 ワークフロー")
    steps = [
        ("1", "データ読み込み", "done" if st.session_state["df"] is not None else "active"),
        ("2", "型判定・EDA", "done" if st.session_state["detection_result"] is not None else
         ("active" if st.session_state["df"] is not None else "pending")),
        ("3", "前処理設定", "pending"),
        ("4", "モデル学習", "done" if st.session_state["automl_result"] is not None else "pending"),
        ("5", "評価・解釈", "done" if st.session_state["automl_result"] is not None else "pending"),
    ]
    cols = st.columns(5)
    for col, (num, label, status) in zip(cols, steps):
        css = {"done": "step-done", "active": "step-active", "pending": "step-pending"}[status]
        icon = {"done": "✓", "active": num, "pending": num}[status]
        with col:
            st.markdown(f"""
<div class="step-item">
  <div class="step-circle {css}">{icon}</div>
  <div class="step-label">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 機能カード
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div class="card">
<h4>📂 対応データ形式</h4>
<ul style="color:#b0afd0; font-size:0.9rem; line-height:1.8;">
<li>CSV / Excel / Parquet / JSON</li>
<li>SQLite データベース</li>
<li>SMILES含有CSV / SDFファイル</li>
</ul>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="card">
<h4>🤖 対応モデル</h4>
<ul style="color:#b0afd0; font-size:0.9rem; line-height:1.8;">
<li>回帰: 25種類以上</li>
<li>分類: 18種類以上</li>
<li>XGBoost / LightGBM / CatBoost</li>
</ul>
</div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="card">
<h4>🧬 化合物特徴量</h4>
<ul style="color:#b0afd0; font-size:0.9rem; line-height:1.8;">
<li>RDKit 物理化学記述子</li>
<li>Morgan / RDKit FP</li>
<li>Mordred / MACE (オプション)</li>
</ul>
</div>""", unsafe_allow_html=True)

    # クイックスタートボタン
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🚀 データを読み込んで開始する", use_container_width=True):
            st.session_state["page"] = "data_load"
            st.rerun()

# ===============================================================
# データ読み込みページ
# ===============================================================
elif page == "data_load":
    from frontend_streamlit.pages import data_load_page
    data_load_page.render()

# ===============================================================
# EDAページ
# ===============================================================
elif page == "eda":
    from frontend_streamlit.pages import eda_page
    eda_page.render()

# ===============================================================
# 前処理設定ページ
# ===============================================================
elif page == "preprocess":
    from frontend_streamlit.pages import preprocess_page
    preprocess_page.render()

# ===============================================================
# AutoML 実行ページ
# ===============================================================
elif page == "automl":
    from frontend_streamlit.pages import automl_page
    automl_page.render()

# ===============================================================
# モデル評価ページ
# ===============================================================
elif page == "evaluation":
    from frontend_streamlit.pages import evaluation_page
    evaluation_page.render()

# ===============================================================
# 次元削減ページ
# ===============================================================
elif page == "dim_reduction":
    from frontend_streamlit.pages import dim_reduction_page
    dim_reduction_page.render()

# ===============================================================
# 化合物解析ページ
# ===============================================================
elif page == "chem":
    from frontend_streamlit.pages import chem_page
    chem_page.render()

# ===============================================================
# 解釈・SHAPページ
# ===============================================================
elif page == "interpret":
    from frontend_streamlit.pages import interpret_page
    interpret_page.render()
