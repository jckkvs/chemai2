"""
frontend_streamlit/pages/dim_reduction_page.py
次元削減（PCA / t-SNE / UMAP）可視化ページ。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render() -> None:
    st.markdown("## 📐 次元削減")

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
        if st.button("📂 データ読み込みへ"):
            st.session_state["page"] = "data_load"
            st.rerun()
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.error("次元削減には2列以上の数値列が必要です。")
        return

    target_col = st.session_state.get("target_col")

    # ─── サイドパネル設定 ─────────────────────────────────────────────
    with st.expander("⚙️ 設定", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("手法", ["PCA", "t-SNE", "UMAP"], key="dr_method")
            use_scale = st.checkbox("事前スケーリング (StandardScaler)", value=True)
            color_by = st.selectbox(
                "色付け列",
                ["なし"] + df.columns.tolist(),
                index=0 if not target_col else df.columns.tolist().index(target_col) + 1
                if target_col in df.columns else 0,
                key="dr_color",
            )
        with col2:
            n_components = st.slider("次元数", 2, 3, 2, key="dr_n_comp")
            if method == "t-SNE":
                perplexity = st.slider("パープレキシティ", 5.0, 50.0, 30.0, 1.0)
                n_iter = st.number_input("最大反復数", 250, 5000, 1000, 250)
            elif method == "UMAP":
                n_neighbors = st.slider("近傍数 (n_neighbors)", 2, 100, 15)
                min_dist = st.slider("最小距離 (min_dist)", 0.0, 1.0, 0.1, 0.05)

        exclude_cols = st.multiselect(
            "除外する列", numeric_cols,
            default=[target_col] if target_col and target_col in numeric_cols else []
        )

    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    if len(feature_cols) < 2:
        st.warning("特徴量列が2列以上必要です。除外列を減らしてください。")
        return

    st.info(f"✅ 使用する特徴量: **{len(feature_cols)}列** / サンプル数: **{len(df):,}件**")

    # ─── 実行ボタン ─────────────────────────────────────────────────
    if st.button(f"▶️ {method} を実行", type="primary"):
        with st.spinner(f"{method} を計算中..."):
            try:
                from backend.data.dim_reduction import DimReductionConfig, DimReducer

                sub_df = df[feature_cols].dropna()
                idx = sub_df.index

                if method == "PCA":
                    cfg = DimReductionConfig(method="pca", n_components=n_components, scale=use_scale)
                elif method == "t-SNE":
                    safe_perp = min(perplexity, (len(sub_df) - 1) / 3)
                    cfg = DimReductionConfig(
                        method="tsne", n_components=n_components, scale=use_scale,
                        perplexity=safe_perp, tsne_n_iter=int(n_iter),  # type:ignore[reportPossiblyUnbound]
                    )
                else:  # UMAP
                    cfg = DimReductionConfig(
                        method="umap", n_components=n_components, scale=use_scale,
                        n_neighbors=n_neighbors, min_dist=min_dist,  # type:ignore[reportPossiblyUnbound]
                    )

                reducer = DimReducer(cfg)
                embedding = reducer.fit_transform(sub_df)

                dim_labels = (
                    [f"PC{i+1}" for i in range(n_components)] if method == "PCA"
                    else [f"{method}{i+1}" for i in range(n_components)]
                )
                emb_df = pd.DataFrame(embedding, columns=dim_labels, index=idx)

                # color列をマージ
                if color_by != "なし" and color_by in df.columns:
                    emb_df["_color"] = df.loc[idx, color_by].values
                    color_col_name = "_color"
                else:
                    color_col_name = None

                st.session_state["_dr_result"] = emb_df
                st.session_state["_dr_dim_labels"] = dim_labels
                st.session_state["_dr_method"] = method
                st.session_state["_dr_color"] = color_col_name

                # PCA寄与率
                if method == "PCA" and reducer.explained_variance_ratio_ is not None:
                    st.session_state["_dr_evr"] = reducer.explained_variance_ratio_

            except ImportError as e:
                st.error(f"❌ {e}")
                return
            except Exception as e:
                st.error(f"❌ 計算エラー: {e}")
                return

    # ─── 結果可視化 ──────────────────────────────────────────────────
    emb_df = st.session_state.get("_dr_result")
    if emb_df is None:
        return

    dim_labels = st.session_state.get("_dr_dim_labels", [])
    dr_method = st.session_state.get("_dr_method", "")
    color_col_name = st.session_state.get("_dr_color")
    evr = st.session_state.get("_dr_evr")

    import plotly.express as px

    # PCA寄与率バーチャート
    if dr_method == "PCA" and evr is not None:
        st.markdown("### 📊 PCA 寄与率")
        col1, col2 = st.columns([2, 1])
        with col1:
            evr_df = pd.DataFrame({
                "主成分": [f"PC{i+1}" for i in range(len(evr))],
                "寄与率": evr * 100,
                "累積寄与率": evr.cumsum() * 100,
            })
            fig_evr = px.bar(evr_df, x="主成分", y="寄与率",
                             text=evr_df["寄与率"].map("{:.1f}%".format),
                             color="寄与率", color_continuous_scale="Blues",
                             template="plotly_dark")
            fig_evr.add_scatter(x=evr_df["主成分"], y=evr_df["累積寄与率"],
                                mode="lines+markers", name="累積",
                                line=dict(color="#ff6b9d", width=2))
            fig_evr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0"), coloraxis_showscale=False,
            )
            st.plotly_chart(fig_evr, use_container_width=True)
        with col2:
            st.dataframe(evr_df.round(2), use_container_width=True)

    # 2D / 3D 散布図
    st.markdown(f"### 🔵 {dr_method} 埋め込み可視化")
    if len(dim_labels) >= 3:
        view_mode = st.radio("表示モード", ["2D", "3D"], horizontal=True)
    else:
        view_mode = "2D"

    color_data = emb_df[color_col_name] if color_col_name and color_col_name in emb_df.columns else None

    if view_mode == "2D":
        fig = px.scatter(
            emb_df, x=dim_labels[0], y=dim_labels[1],
            color=color_data,
            opacity=0.75,
            color_continuous_scale="Viridis" if color_data is not None and pd.api.types.is_numeric_dtype(color_data) else None,
            template="plotly_dark",
            title=f"{dr_method} 2D埋め込み (n={len(emb_df):,})",
        )
    else:
        fig = px.scatter_3d(
            emb_df, x=dim_labels[0], y=dim_labels[1], z=dim_labels[2],
            color=color_data,
            opacity=0.7,
            color_continuous_scale="Viridis" if color_data is not None and pd.api.types.is_numeric_dtype(color_data) else None,
            template="plotly_dark",
            title=f"{dr_method} 3D埋め込み (n={len(emb_df):,})",
        )
        fig.update_traces(marker=dict(size=3))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 埋め込み結果ダウンロード
    st.download_button(
        "💾 埋め込み結果をCSVでダウンロード",
        emb_df.drop(columns=["_color"], errors="ignore").to_csv(index=True),
        file_name=f"{dr_method.lower()}_embedding.csv",
        mime="text/csv",
    )
