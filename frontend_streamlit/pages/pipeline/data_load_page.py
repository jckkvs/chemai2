"""
frontend_streamlit/pages/data_load_page.py
データ読み込みページ。ファイルアップロード・型判定・プレビューを提供。
"""
from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from backend.data.loader import load_from_bytes, get_supported_extensions
from backend.data.type_detector import TypeDetector


def render() -> None:
    st.markdown("## 📂 データ読み込み")

    # ── ファイルアップロード ──────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ファイルを選択")

    ext_list = ", ".join(get_supported_extensions())
    uploaded = st.file_uploader(
        f"対応フォーマット: {ext_list}",
        type=[e.lstrip(".") for e in get_supported_extensions()],
        help="CSVファイルはエンコーディングが自動判定されます。",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
<div style="text-align:center; padding:3rem; color:#555; border:2px dashed #333; border-radius:12px;">
<div style="font-size:3rem;">📄</div>
<div style="margin-top:1rem;">ファイルをドラッグ＆ドロップ、またはブラウズして選択</div>
</div>""", unsafe_allow_html=True)

        # サンプルデータ生成ボタン
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🧪 回帰サンプルデータを生成", use_container_width=True):
                import numpy as np
                np.random.seed(42)
                n = 200
                df = pd.DataFrame({
                    "temperature": np.random.uniform(20, 80, n),
                    "pressure": np.random.exponential(5, n),
                    "catalyst": np.random.choice(["A型", "B型", "C型"], n),
                    "time_h": np.random.uniform(1, 24, n),
                    "is_active": np.random.randint(0, 2, n),
                    "yield": np.random.randn(n) * 10 + 75,
                })
                _store_df(df, "sample_regression.csv")
                st.rerun()
        with c2:
            if st.button("🏷️ 分類サンプルデータを生成", use_container_width=True):
                import numpy as np
                np.random.seed(42)
                n = 200
                df = pd.DataFrame({
                    "feature_1": np.random.randn(n),
                    "feature_2": np.random.randn(n),
                    "category": np.random.choice(["低", "中", "高"], n),
                    "numeric": np.random.randint(1, 100, n),
                    "label": np.random.randint(0, 2, n),
                })
                _store_df(df, "sample_classification.csv")
                st.rerun()
        with c3:
            if st.button("🧬 SMILES サンプルデータを生成", use_container_width=True):
                df = pd.DataFrame({
                    "smiles": ["CCO", "C", "CC", "CCC", "CCCC", "c1ccccc1",
                               "c1ccccc1O", "c1ccccc1N", "CC(=O)O", "CCN",
                               "c1ccc(O)cc1", "CC(C)O", "CCOCC", "ClCCl", "BrC"] * 10,
                    "solubility": [-0.77, 0.0, -0.63, -1.5, -2.1, -1.9,
                                   -0.5, -0.8, -0.3, -1.1, -0.7, -0.9,
                                   -1.3, -1.0, -0.4] * 10,
                    "source": ["ref"] * 150,
                })
                _store_df(df, "sample_smiles.csv")
                st.rerun()
        return

    # ── ファイル読み込み ──────────────────────────────────────
    try:
        with st.spinner("ファイルを読み込み中..."):
            raw = uploaded.read()
            df = load_from_bytes(raw, uploaded.name)
        _store_df(df, uploaded.name)
        st.success(f"✅ `{uploaded.name}` を読み込みました。")
    except Exception as e:
        st.error(f"❌ 読み込みエラー: {e}")
        return

    _show_data_overview()


def _store_df(df: pd.DataFrame, name: str) -> None:
    """DataFrameをセッションに保存し型判定を実行する。"""
    st.session_state["df"] = df
    st.session_state["file_name"] = name
    st.session_state["automl_result"] = None  # リセット

    # 型判定（SMILES列候補を探して設定）
    detector = TypeDetector()
    result = detector.detect(df)
    st.session_state["detection_result"] = result

    if result.smiles_columns:
        st.session_state["smiles_col"] = result.smiles_columns[0]


def _show_data_overview() -> None:
    """データ概要・型判定結果・プレビューを表示する。"""
    df = st.session_state.get("df")
    result = st.session_state.get("detection_result")
    if df is None:
        return

    # 基本メトリクス
    col1, col2, col3, col4 = st.columns(4)
    cols_metrics = [
        (col1, str(df.shape[0]), "行数"),
        (col2, str(df.shape[1]), "列数"),
        (col3, f"{df.isna().mean().mean():.1%}", "欠損率（全体）"),
        (col4, str(df.duplicated().sum()), "重複行数"),
    ]
    for col, val, label in cols_metrics:
        with col:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{val}</div>
  <div class="metric-label">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # タブ: プレビュー / 型判定 / 統計
    tab1, tab2, tab3 = st.tabs(["📄 データプレビュー", "🔍 列型判定結果", "📊 基本統計"])

    with tab1:
        rows = st.slider("表示行数", 5, min(100, len(df)), 10)
        st.dataframe(df.head(rows), use_container_width=True)

    with tab2:
        if result:
            summary = result.summary_table()
            # 色付き表示
            def color_type(val: str) -> str:
                colors = {
                    "NUMERIC_NORMAL": "#4db8ff",
                    "NUMERIC_LOG": "#60a5fa",
                    "BINARY": "#4ade80",
                    "CATEGORY_LOW": "#fbbf24",
                    "CATEGORY_HIGH": "#f97316",
                    "SMILES": "#c084fc",
                    "DATETIME": "#a78bfa",
                    "CONSTANT": "#6b7280",
                }
                color = colors.get(str(val), "#e0e0f0")
                return f"color: {color}; font-weight: bold;"
            styled = summary.style.applymap(color_type, subset=["col_type"])
            st.dataframe(styled, use_container_width=True, height=350)

            if result.smiles_columns:
                st.info(f"🧬 SMILES列を検出: **{', '.join(result.smiles_columns)}**")

    with tab3:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if numeric_cols:
            st.dataframe(
                df[numeric_cols].describe().round(4),
                use_container_width=True,
            )
        else:
            st.info("数値列がありません。")

    # ── 目的変数の選択 ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 目的変数（ターゲット）の選択")
    col_a, col_b = st.columns(2)
    with col_a:
        target = st.selectbox(
            "目的変数列を選択",
            options=df.columns.tolist(),
            index=len(df.columns) - 1,
            key="target_col_select",
        )
        st.session_state["target_col"] = target
    with col_b:
        task = st.selectbox(
            "タスク種別",
            ["auto（自動判定）", "regression（回帰）", "classification（分類）"],
            key="task_select",
        )
        st.session_state["task"] = task.split("（")[0]

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 EDA を実行", use_container_width=True):
            st.session_state["page"] = "eda"
            st.rerun()
    with c2:
        if st.button("🤖 AutoML を実行", use_container_width=True):
            st.session_state["page"] = "automl"
            st.rerun()
