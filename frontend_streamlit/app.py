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
        ("🔬", "パイプライン",   "pipeline",      has_data),
        ("📊", "モデル評価",     "evaluation",    has_result),
        ("📐", "次元削減",       "dim_reduction", has_data),
        ("💡", "SHAP 解釈",      "interpret",     has_result),
        ("🧬", "化合物解析",     "chem",          True),
        ("📚", "推奨変数ヘルプ", "help_page",     True),
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
            st.session_state["smiles_col"]     = None  # 必ずリセット
            detector = TypeDetector()
            dr = detector.detect(df)
            st.session_state["detection_result"] = dr
            if dr.smiles_columns:
                st.session_state["smiles_col"] = dr.smiles_columns[0]
            else:
                for col in df.columns:
                    if col.lower() == "smiles":
                        st.session_state["smiles_col"] = col
                        break
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
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("**ML設定**")
                cv_folds   = st.slider("CV分割数", 2, 10, 5, key="adv_cv")
                timeout    = st.slider("タイムアウト(秒)", 30, 3600, 300, key="adv_to")
                
                # タスクに応じた利用可能モデルの取得
                from backend.models.factory import list_models, get_default_automl_models
                _tmp_task = st.session_state.get("task", "auto")
                if _tmp_task == "auto":
                    _tmp_task = "regression" if pd.api.types.is_float_dtype(df[st.session_state.get("target_col")]) else "classification"
                
                available_models = list_models(task=_tmp_task, available_only=True)
                model_options = {m["key"]: f'{m["name"]} {"" if m["available"] else "(未インストール)"}' for m in available_models}
                default_models = get_default_automl_models(task=_tmp_task)
                
                selected_models = st.multiselect(
                    "使用するモデル",
                    options=list(model_options.keys()),
                    default=default_models,
                    format_func=lambda x: model_options.get(x, x),
                    key="adv_models",
                    help="AutoMLで評価するモデルを選択します"
                )
            with cb:
                st.markdown("**前処理設定**")
                scaler    = st.selectbox("数値スケーラー",
                    ["auto","standard","robust","minmax","none"], key="adv_sc")
                
                smiles_options = ["なし"] + df.columns.tolist()
                smiles_default_idx = 0
                for i, col in enumerate(df.columns):
                    if col.lower() == "smiles":
                        smiles_default_idx = i + 1
                        break
                        
                smiles_raw = st.selectbox("SMILES列",
                    smiles_options, index=smiles_default_idx, key="adv_sm")
                st.session_state["smiles_col"] = None if smiles_raw == "なし" else smiles_raw
            with cc:
                st.markdown("**実行フェーズ**")
                do_eda  = st.checkbox("EDA", value=True, key="adv_eda")
                do_prep = st.checkbox("前処理確認", value=True, key="adv_prep")
                do_eval = st.checkbox("評価", value=True, key="adv_eval")
                do_pca  = st.checkbox("次元削減(PCA)", value=True, key="adv_pca")
                do_shap = st.checkbox("SHAP解析", value=True, key="adv_shap")

            st.markdown("---")
            st.markdown("**🧪 記述子計算設定（任意）**")

            from backend.chem import RDKitAdapter, XTBAdapter, CosmoAdapter, UniPkaAdapter, GroupContribAdapter, MordredAdapter
            from backend.chem.recommender import (
                get_target_recommendation_by_name,
                get_target_categories,
                get_targets_by_category,
                get_all_descriptor_categories,
                get_descriptors_by_category,
                get_all_target_recommendations
            )

            all_adapters = [RDKitAdapter(compute_fp=True), MordredAdapter(selected_only=True), XTBAdapter(), CosmoAdapter(), UniPkaAdapter(), GroupContribAdapter()]
            
            # 辞書化してライブラリごとの記述子を持っておく（タブ3用）
            lib_descriptors = {}
            all_available_descriptors = []
            for adp in all_adapters:
                if adp.is_available():
                    lib_name = adp.__class__.__name__.replace("Adapter", "")
                    names = adp.get_descriptor_names()
                    lib_descriptors[lib_name] = names
                    all_available_descriptors.extend(names)

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

            tab1, tab2, tab3 = st.tabs(["目的変数の系統から選ぶ", "記述子の意味から選ぶ", "計算ライブラリから選ぶ"])

            with tab1:
                st.markdown("予測したい目的変数の系統（光、強度など）に合わせて、推奨される記述子のセットを一括で追加・削除できます。")
                target_col_val = st.session_state.get("target_col", "")
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

            with tab2:
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
                            
                        # 個別のチェックボックス
                        for d in valid_descs:
                            is_checked = d.name in current_selected
                            changed = st.checkbox(f"**{d.name}** ({d.library}): {d.meaning}", value=is_checked, key=f"chk_mean_{dcat}_{d.name}")
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
                        new_lib_selected = st.multiselect(
                            f"{lib} の出力記述子",
                            options=d_names,
                            default=lib_selected,
                            key=f"ms_lib_{lib}",
                            label_visibility="collapsed"
                        )
                        
                        # 差分があれば更新
                        if set(new_lib_selected) != set(lib_selected):
                            diff_add = set(new_lib_selected) - set(lib_selected)
                            diff_remove = set(lib_selected) - set(new_lib_selected)
                            current_selected.update(diff_add)
                            current_selected.difference_update(diff_remove)
                            st.session_state["adv_desc"] = list(current_selected)

            selected_desc = list(current_selected)
            st.caption(f"✅ 現在 {len(selected_desc)} 件の記述子が選択されています。")
            if len(selected_desc) > 0:
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
                            # データベースに明示的な意味登録がないその他記述子
                            st.markdown(f"- **{d_name}** <span style='color:#888;font-size:0.85em;'>(全般的な記述子)</span>", unsafe_allow_html=True)
                    
                    if hidden_count > 0:
                        st.markdown(f"- ...他 **{hidden_count}** 件（省略）")

            # 詳細設定の値をセッションに保存
            st.session_state["_adv"] = dict(
                cv_folds=cv_folds, models=selected_models, timeout=timeout,
                scaler=scaler,
                do_eda=do_eda, do_prep=do_prep, do_eval=do_eval,
                do_pca=do_pca, do_shap=do_shap,
                selected_descriptors=selected_desc,
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
                        models     = adv.get("models", []),
                        timeout    = adv.get("timeout", 300),
                        scaler     = adv.get("scaler", "auto"),
                        do_eda     = adv.get("do_eda", True),
                        do_prep    = adv.get("do_prep", True),
                        do_eval    = adv.get("do_eval", True),
                        do_pca     = adv.get("do_pca", True),
                        do_shap    = adv.get("do_shap", True),
                        selected_descriptors = adv.get("selected_descriptors", None),
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

elif page == "pipeline":
    from frontend_streamlit.pages.pipeline import pipeline_page
    pipeline_page.render()

elif page == "help_page":
    from frontend_streamlit.pages import help_page
    help_page.render_help_page()
