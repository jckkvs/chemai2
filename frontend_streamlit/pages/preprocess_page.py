"""frontend_streamlit/pages/preprocess_page.py - 前処理設定ページ"""
from __future__ import annotations
import streamlit as st
from backend.data.preprocessor import PreprocessConfig


def render() -> None:
    st.markdown("## ⚙️ 前処理設定")
    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
        return

    st.markdown("### 数値スケーラー")
    scaler = st.selectbox("スケーラー選択",
        ["auto（自動）", "standard（StandardScaler）", "robust（外れ値対応）",
         "minmax（0-1正規化）", "power_yj（YeoJohnson）", "quantile_normal（分位数）", "none（変換なし）"])

    st.markdown("### カテゴリエンコーダー")
    col1, col2 = st.columns(2)
    with col1:
        enc_low = st.selectbox("低cardinality (< 20ユニーク)", ["onehot", "ordinal", "target"])
    with col2:
        enc_high = st.selectbox("高cardinality (≥ 20ユニーク)", ["ordinal", "target", "hashing", "binary"])

    st.markdown("### 欠損値補完")
    col3, col4 = st.columns(2)
    with col3:
        imputer_num = st.selectbox("数値欠損", ["mean", "median", "knn", "iterative"])
    with col4:
        imputer_cat = st.selectbox("カテゴリ欠損", ["most_frequent", "constant"])

    st.markdown("### 除外設定")
    col5, col6, col7 = st.columns(3)
    with col5:
        excl_smiles = st.checkbox("SMILES列を除外", value=True)
    with col6:
        excl_dt = st.checkbox("DateTime列を除外", value=True)
    with col7:
        excl_const = st.checkbox("定数列を除外", value=True)

    if st.button("✅ 設定を保存", use_container_width=True):
        cfg = PreprocessConfig(
            numeric_scaler=scaler.split("（")[0],
            cat_low_encoder=enc_low,
            cat_high_encoder=enc_high,
            numeric_imputer=imputer_num,
            categorical_imputer=imputer_cat,
            exclude_smiles=excl_smiles,
            exclude_datetime=excl_dt,
            exclude_constant=excl_const,
        )
        st.session_state["preprocess_config"] = cfg
        st.success("✅ 前処理設定を保存しました。AutoML実行時に反映されます。")
