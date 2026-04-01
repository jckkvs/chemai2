"""
frontend_nicegui/components/interpretation_panel.py

モデル解釈性パネル: 予測実測プロット・データ点表・回帰係数・SHAP・SAGE・SRI
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui, run

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# メインエントリポイント
# ═══════════════════════════════════════════════════════════════
def render_interpretation_panel(ar: Any, state: dict) -> None:
    """解釈性パネル全体を描画する。"""

    model = getattr(ar, "best_pipeline", None)
    X: Any = getattr(ar, "processed_X", None)
    y: Any = getattr(ar, "y_train", None)
    y_true_oof: Any = getattr(ar, "oof_true", None)
    y_pred_oof: Any = getattr(ar, "oof_predictions", None)

    if model is None or X is None:
        ui.label("⚠️ モデルまたはデータが取得できませんでした").classes("text-amber")
        return

    feat_names: list[str] = (
        list(X.columns) if hasattr(X, "columns")
        else [f"feature_{i}" for i in range(X.shape[1])]
    )
    X_arr: np.ndarray = X.values if hasattr(X, "values") else np.asarray(X)

    # ── サブタブ構成 ──
    with ui.tabs().classes("full-width").props(
        "dense active-color=cyan indicator-color=cyan scrollable"
    ) as interp_tabs:
        t_pred    = ui.tab("pred",    label="📈 予測実測プロット")
        t_table   = ui.tab("table",   label="📋 データ点表")
        t_coef    = ui.tab("coef",    label="📊 回帰係数/特徴量重要度")
        t_shap    = ui.tab("shap",    label="🔍 SHAP")
        t_sage    = ui.tab("sage",    label="🌿 SAGE")
        t_sri     = ui.tab("sri",     label="🔬 SRI分解")

    with ui.tab_panels(interp_tabs, value=t_pred).classes("full-width bg-transparent"):

        # ════════════════════════════════════════════════════
        # 予測実測プロット
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_pred):
            _render_pred_actual(ar, y_true_oof, y_pred_oof, feat_names, X_arr)

        # ════════════════════════════════════════════════════
        # データ点ごとの X / y / y_pred 表
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_table):
            _render_sample_table(ar, X, feat_names, y_true_oof, y_pred_oof)

        # ════════════════════════════════════════════════════
        # 回帰係数・Feature Importance
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_coef):
            _render_coefficients(ar, model, X, feat_names)

        # ════════════════════════════════════════════════════
        # SHAP
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_shap):
            _render_shap_panel(ar, model, X, X_arr, feat_names, y)

        # ════════════════════════════════════════════════════
        # SAGE
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_sage):
            _render_sage_panel(ar, model, X, X_arr, feat_names, y)

        # ════════════════════════════════════════════════════
        # SRI 分解
        # ════════════════════════════════════════════════════
        with ui.tab_panel(t_sri):
            _render_sri_panel(ar, model, X, X_arr, feat_names)


# ═══════════════════════════════════════════════════════════════
# 予測実測プロット
# ═══════════════════════════════════════════════════════════════
def _render_pred_actual(ar, y_true, y_pred, feat_names, X_arr) -> None:
    """予測 vs 実測の散布図（OOF）を描画する。"""
    import plotly.graph_objects as go
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    ui.label("📈 予測値 vs 実測値プロット (OOF)").classes("text-subtitle1 text-bold q-mb-xs")
    ui.label(
        "Out-of-Fold (OOF) 予測: 学習に使われていないデータに対する予測値と実測値を比較します。"
        "完璧なモデルは全点が y=x 直線上に乗ります。"
    ).classes("text-caption text-grey-5 q-mb-md")

    if y_true is None or y_pred is None:
        ui.label("⚠️ OOFデータが利用できません").classes("text-amber")
        return

    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    residuals = y_t - y_p

    # 指標
    try:
        r2   = r2_score(y_t, y_p)
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        mae  = float(mean_absolute_error(y_t, y_p))
    except Exception:
        r2 = rmse = mae = float("nan")

    with ui.row().classes("q-gutter-md q-mb-md"):
        for val, lbl, color in [
            (f"{r2:.4f}", "R² (OOF)",   "cyan"),
            (f"{rmse:.4f}", "RMSE (OOF)", "amber"),
            (f"{mae:.4f}",  "MAE (OOF)",  "green"),
        ]:
            with ui.card().classes("q-pa-sm").style(
                f"min-width:90px; background:rgba(0,0,0,0.2); border-radius:8px;"
                f"border:1px solid rgba(0,212,255,0.15);"
            ):
                ui.label(val).classes(f"text-h6 text-bold text-{color}")
                ui.label(lbl).classes("text-caption text-grey-5")

    # 散布図
    rng = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    n_pts = len(y_t)
    sample_idx = np.arange(n_pts)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rng, y=rng, mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1.5),
        name="y = x (完璧な予測)",
    ))
    fig.add_trace(go.Scatter(
        x=y_t, y=y_p,
        mode="markers",
        marker=dict(
            size=7,
            color=residuals,
            colorscale="RdBu_r",
            showscale=True,
            colorbar=dict(title="残差"),
            opacity=0.75,
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        text=[f"Sample {i}<br>実測: {y_t[i]:.4f}<br>予測: {y_p[i]:.4f}<br>残差: {residuals[i]:.4f}"
              for i in sample_idx],
        hovertemplate="%{text}<extra></extra>",
        name="データ点",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="実測値",
        yaxis_title="予測値",
        title=f"予測値 vs 実測値 (OOF, n={n_pts})",
        legend=dict(orientation="h", y=1.05),
    )
    ui.plotly(fig).classes("full-width")

    # 残差 vs 予測値
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=y_p, y=residuals, mode="markers",
        marker=dict(size=5, color="rgba(74,222,128,0.6)"),
        text=[f"Sample {i}<br>予測: {y_p[i]:.4f}<br>残差: {residuals[i]:.4f}" for i in sample_idx],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)", height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="予測値", yaxis_title="残差 (実測 - 予測)",
        title="残差プロット",
    )
    ui.plotly(fig2).classes("full-width q-mt-md")


# ═══════════════════════════════════════════════════════════════
# データ点ごとの X / y / y_pred 表
# ═══════════════════════════════════════════════════════════════
def _render_sample_table(ar, X, feat_names, y_true, y_pred) -> None:
    """全サンプルの X・y・y_pred を横並びの表で表示する。"""

    ui.label("📋 データ点別 予測結果表").classes("text-subtitle1 text-bold q-mb-xs")
    ui.label(
        "各サンプルの全特徴量値 (X)・実測値 (y)・OOF予測値 (y_pred)・残差を一覧表示します。"
    ).classes("text-caption text-grey-5 q-mb-md")

    if y_true is None or y_pred is None:
        ui.label("⚠️ OOFデータが利用できません").classes("text-amber")
        return

    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    residuals = y_t - y_p

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    n = min(len(y_t), X_arr.shape[0])
    display_feats = feat_names[:20]  # 最大20列

    # 検索・ソート用フィルタ
    search_input = ui.input(
        placeholder="🔍 サンプルインデックスで絞り込み...",
        on_change=lambda e: _update_table(e.value),
    ).props("outlined dense clearable").classes("q-mb-sm full-width")

    table_container = ui.column().classes("full-width")

    def _update_table(query: str = ""):
        table_container.clear()
        with table_container:
            # フィルタ
            indices = list(range(n))
            if query.strip().isdigit():
                indices = [i for i in indices if str(i) == query.strip()]

            rows = []
            for i in indices[:500]:  # 最大500行
                row = {"#": i, "y_実測": f"{y_t[i]:.4f}", "y_予測": f"{y_p[i]:.4f}",
                       "残差": f"{residuals[i]:.4f}",
                       "残差%": f"{abs(residuals[i] / y_t[i] * 100):.1f}%" if abs(y_t[i]) > 1e-12 else "inf"}
                for fi, fn in enumerate(display_feats):
                    if fi < X_arr.shape[1]:
                        v = X_arr[i, fi]
                        row[fn] = f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
                rows.append(row)

            fixed_cols = [
                {"name": "#",      "label": "#",       "field": "#",      "align": "center", "sortable": True},
                {"name": "y_実測", "label": "y_実測",  "field": "y_実測", "align": "right",  "sortable": True},
                {"name": "y_予測", "label": "y_予測",  "field": "y_予測", "align": "right",  "sortable": True},
                {"name": "残差",   "label": "残差",     "field": "残差",   "align": "right",  "sortable": True},
                {"name": "残差%",  "label": "|残差|%", "field": "残差%",  "align": "right",  "sortable": True},
            ]
            feat_cols = [
                {"name": fn, "label": fn[:12], "field": fn, "align": "right", "sortable": True}
                for fn in display_feats
            ]

            ui.table(
                columns=fixed_cols + feat_cols,
                rows=rows,
                pagination={"rowsPerPage": 20, "sortBy": "#"},
            ).classes("full-width").props("dense flat bordered virtual-scroll")

            if n > 500:
                ui.label(f"... 先頭500件を表示中（全{n}件）").classes("text-caption text-grey-6 q-mt-xs")

    _update_table()

    # CSVダウンロード
    def _download_csv():
        y_t2 = np.asarray(y_true).ravel()
        y_p2 = np.asarray(y_pred).ravel()
        res2 = y_t2 - y_p2
        X_a2 = X.values if hasattr(X, "values") else np.asarray(X)
        df = pd.DataFrame(X_a2[:, :len(feat_names)], columns=feat_names)
        df.insert(0, "y_実測", y_t2)
        df.insert(1, "y_予測", y_p2)
        df.insert(2, "残差", res2)
        csv_bytes = df.to_csv(index=True, index_label="#").encode("utf-8-sig")
        ui.download(csv_bytes, "prediction_table.csv")
        ui.notify("📥 CSVダウンロード開始", type="positive")

    ui.button("📥 全データをCSV出力", on_click=_download_csv).props(
        "outline color=cyan size=sm no-caps"
    ).classes("q-mt-sm")


# ═══════════════════════════════════════════════════════════════
# 回帰係数・Feature Importance
# ═══════════════════════════════════════════════════════════════
def _render_coefficients(ar, model, X, feat_names: list[str]) -> None:
    """線形モデルの回帰係数 / ツリー系の feature_importances_ を描画する。"""
    import plotly.graph_objects as go

    ui.label("📊 回帰係数 / Feature Importance").classes("text-subtitle1 text-bold q-mb-xs")

    try:
        estimator = model
        if hasattr(model, "steps"):
            estimator = model.steps[-1][1]
            if hasattr(estimator, "steps"):
                estimator = estimator.steps[-1][1]

        # ── ツリー系: feature_importances_ ──
        if hasattr(estimator, "feature_importances_"):
            imp = estimator.feature_importances_
            fi_len = len(imp)
            names = feat_names[:fi_len] if len(feat_names) >= fi_len else [f"f{i}" for i in range(fi_len)]
            idx = np.argsort(imp)[::-1]
            top = min(30, len(idx))

            ui.label("ツリー系モデルの Impurity-based Feature Importance (Top 30)").classes(
                "text-caption text-grey-5 q-mb-sm"
            )
            fig = go.Figure(go.Bar(
                x=imp[idx[:top]][::-1],
                y=[names[i] if i < len(names) else f"f{i}" for i in idx[:top]][::-1],
                orientation="h",
                marker=dict(
                    color=imp[idx[:top]][::-1],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="重要度"),
                ),
                text=[f"{v:.4f}" for v in imp[idx[:top]][::-1]],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.1)",
                height=max(350, 22 * top), margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Feature Importance (不純度減少の平均)",
                title=f"Feature Importance ({ar.best_model_key})",
            )
            ui.plotly(fig).classes("full-width")

            # 表
            rows = [
                {"順位": i + 1,
                 "特徴量": names[idx[i]] if idx[i] < len(names) else f"f{idx[i]}",
                 "重要度": f"{imp[idx[i]]:.6f}",
                 "累積": f"{imp[idx[:i+1]].sum():.4f}"}
                for i in range(top)
            ]
            cols = [{"name": k, "label": k, "field": k,
                     "align": "center" if k != "特徴量" else "left", "sortable": True}
                    for k in ["順位", "特徴量", "重要度", "累積"]]
            with ui.expansion("📋 数値テーブル", icon="table_chart").classes("full-width q-mt-md"):
                ui.table(columns=cols, rows=rows).classes("full-width").props("dense flat bordered")

        # ── 線形モデル: coef_ ──
        elif hasattr(estimator, "coef_"):
            coefs = estimator.coef_.ravel()
            coef_len = len(coefs)
            names = feat_names[:coef_len] if len(feat_names) >= coef_len else [f"f{i}" for i in range(coef_len)]
            idx = np.argsort(np.abs(coefs))[::-1]
            top = min(30, len(idx))

            ui.label("線形モデルの回帰係数 (|係数| 上位30)").classes("text-caption text-grey-5 q-mb-sm")

            colors = ["rgba(74,222,128,0.7)" if coefs[i] > 0 else "rgba(248,113,113,0.7)"
                      for i in idx[:top]]
            fig = go.Figure(go.Bar(
                x=coefs[idx[:top]][::-1],
                y=[names[i] if i < len(names) else f"f{i}" for i in idx[:top]][::-1],
                orientation="h",
                marker_color=colors[::-1],
                text=[f"{coefs[i]:+.4f}" for i in idx[:top]][::-1],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.1)",
                height=max(350, 22 * top), margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="回帰係数",
                title=f"回帰係数 ({ar.best_model_key})",
            )
            ui.plotly(fig).classes("full-width")
            ui.label("🟢 正の係数 = 目的変数を増加させる方向  🔴 負の係数 = 減少させる方向").classes(
                "text-caption text-grey-5 q-mt-xs"
            )

            # Intercept 表示
            if hasattr(estimator, "intercept_"):
                ui.label(
                    f"切片 (intercept): {float(estimator.intercept_):.6f}"
                ).classes("text-caption text-cyan q-mt-xs")

        else:
            ui.label("ℹ️ このモデルは回帰係数/Feature Importanceを直接取得できません。").classes("text-grey-5")
            ui.label("SHAP タブを利用してください。").classes("text-caption text-grey-6")

    except Exception as ex:
        ui.label(f"❌ 取得エラー: {ex}").classes("text-red text-caption")


# ═══════════════════════════════════════════════════════════════
# SHAP パネル
# ═══════════════════════════════════════════════════════════════
def _render_shap_panel(ar, model, X, X_arr, feat_names, y) -> None:
    """SHAP Summary Bar・Beeswarm・Waterfall・Dependence を描画する。"""
    import plotly.graph_objects as go

    ui.label("🔍 SHAP解析").classes("text-subtitle1 text-bold q-mb-xs")
    ui.label(
        "Shapley Additive exPlanations: 各特徴量の予測への平均的な寄与を測定します。"
    ).classes("text-caption text-grey-5 q-mb-md")

    shap_container = ui.column().classes("full-width")

    async def _run_shap():
        shap_container.clear()
        with shap_container:
            with ui.card().classes("q-pa-sm glass-card full-width"):
                prog = ui.linear_progress(value=0, show_value=False).props("color=cyan rounded")
                lbl  = ui.label("⏳ SHAP値を計算中...").classes("text-grey-5 text-caption")

        try:
            from backend.interpret.shap_explainer import ShapExplainer

            def _compute():
                exp = ShapExplainer()
                return exp.explain(model, X, feature_names=feat_names)

            prog.value = 0.3
            lbl.text = "⏳ SHAP Explainer を初期化中..."
            shap_result = await run.io_bound(_compute)
            prog.value = 0.9

            sv = shap_result.shap_values
            if sv.ndim == 3:
                sv = sv[:, :, 0]

            fi_df = shap_result.feature_importance()
            top20 = fi_df.head(20)
            exp_val = float(shap_result.expected_value) if not hasattr(shap_result.expected_value, "__len__") \
                else float(shap_result.expected_value[0])

            shap_container.clear()
            with shap_container:
                # 1. Bar Plot
                ui.label("📊 Feature Importance (平均|SHAP値| Top 20)").classes("text-subtitle2 q-mb-xs")
                fig_bar = go.Figure(go.Bar(
                    x=top20["importance"].values[::-1],
                    y=top20["feature"].values[::-1],
                    orientation="h",
                    marker=dict(
                        color=top20["importance"].values[::-1],
                        colorscale="Blues",
                        showscale=True,
                        colorbar=dict(title="|SHAP|"),
                    ),
                    text=[f"{v:.4f}" for v in top20["importance"].values[::-1]],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=max(300, 22 * len(top20)),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="平均|SHAP値|",
                    title="SHAP Feature Importance",
                )
                ui.plotly(fig_bar).classes("full-width")

                # 2. Beeswarm (Top 10)
                ui.separator()
                ui.label("🐝 Beeswarm Plot (Top 10)").classes("text-subtitle2 q-mt-md q-mb-xs")
                top10_feats = fi_df.head(10)["feature"].tolist()
                fig_bee = go.Figure()
                for feat in reversed(top10_feats):
                    fi = feat_names.index(feat) if feat in feat_names else -1
                    if fi < 0 or fi >= sv.shape[1]:
                        continue
                    sv_col = sv[:, fi]
                    fv_col = X_arr[:, fi]
                    fig_bee.add_trace(go.Scatter(
                        x=sv_col, y=[feat] * len(sv_col), mode="markers",
                        marker=dict(
                            size=4, color=fv_col,
                            colorscale="RdBu_r", opacity=0.55,
                            showscale=False,
                        ),
                        name=feat, showlegend=False,
                    ))
                fig_bee.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=max(300, 35 * len(top10_feats)),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="SHAP値", title="Beeswarm Plot",
                )
                ui.plotly(fig_bee).classes("full-width")

                # 3. Waterfall (サンプル 0)
                ui.separator()
                ui.label("💧 Waterfall Plot (サンプル #0)").classes("text-subtitle2 q-mt-md q-mb-xs")
                sv0 = sv[0]
                top_w_idx = np.argsort(np.abs(sv0))[::-1][:15]
                w_feats = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in top_w_idx]
                w_vals  = [sv0[i] for i in top_w_idx]
                fig_wf = go.Figure(go.Waterfall(
                    orientation="h", y=w_feats[::-1], x=w_vals[::-1],
                    connector=dict(line=dict(color="rgba(255,255,255,0.15)")),
                    increasing=dict(marker=dict(color="rgba(74,222,128,0.75)")),
                    decreasing=dict(marker=dict(color="rgba(248,113,113,0.75)")),
                    base=exp_val,
                ))
                fig_wf.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=max(300, 25 * len(w_feats)),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="予測への寄与",
                    title=f"Waterfall (ベースライン: {exp_val:.4f})",
                )
                ui.plotly(fig_wf).classes("full-width")

                # 4. Dependence (Top 1)
                if len(top10_feats) > 0:
                    ui.separator()
                    t1 = top10_feats[0]
                    ti = feat_names.index(t1) if t1 in feat_names else 0
                    ui.label(f"📈 Dependence Plot: {t1}").classes("text-subtitle2 q-mt-md q-mb-xs")
                    fig_dep = go.Figure(go.Scatter(
                        x=X_arr[:, ti], y=sv[:, ti], mode="markers",
                        marker=dict(size=5, color=sv[:, ti], colorscale="RdBu_r",
                                    opacity=0.7, showscale=True,
                                    colorbar=dict(title="SHAP値")),
                    ))
                    fig_dep.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.1)", height=300,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title=t1, yaxis_title=f"SHAP値 ({t1})",
                        title=f"Dependence Plot: {t1}",
                    )
                    ui.plotly(fig_dep).classes("full-width")

                ui.notify("✅ SHAP解析完了", type="positive")

        except ImportError as ie:
            shap_container.clear()
            with shap_container:
                ui.label(f"⚠️ {ie}").classes("text-amber text-caption")
        except Exception as ex:
            shap_container.clear()
            with shap_container:
                ui.label(f"❌ SHAP計算エラー: {ex}").classes("text-red text-caption")

    ui.button("🔍 SHAP解析を実行", on_click=_run_shap).props(
        "unelevated color=cyan size=sm no-caps"
    )
    shap_container


# ═══════════════════════════════════════════════════════════════
# SAGE パネル
# ═══════════════════════════════════════════════════════════════
def _render_sage_panel(ar, model, X, X_arr, feat_names, y) -> None:
    """SAGE (Shapley Additive Global importancE) を計算・描画する。

    参考文献: Covert, Lundberg & Lee, "Understanding Global Feature Contributions
    with Additive Importance Measures", NeurIPS 2020.
    https://arxiv.org/abs/2004.00668

    SAGE は全サンプルを使ったマージナル期待値の差分でグローバル重要度を計算する。
    sklearn-compatible な推定器であれば直接対応する。
    """
    import plotly.graph_objects as go

    ui.label("🌿 SAGE解析").classes("text-subtitle1 text-bold q-mb-xs")
    ui.label(
        "SAGE (Shapley Additive Global importancE): "
        "各特徴量を隠した場合の予測損失増加をShapley値で公平に配分します。"
        "高い値 = その特徴量が全体の予測精度に大きく貢献。"
    ).classes("text-caption text-grey-5 q-mb-xs")
    ui.label(
        "⚠️ SAGE は全サンプル×全特徴量の組み合わせを評価するため計算に時間がかかります。"
    ).classes("text-caption text-amber q-mb-md")

    sage_container = ui.column().classes("full-width")

    async def _run_sage():
        sage_container.clear()
        with sage_container:
            with ui.card().classes("q-pa-sm glass-card full-width"):
                prog = ui.linear_progress(value=0, show_value=False).props("color=green rounded")
                lbl  = ui.label("⏳ SAGE値を計算中（数十秒かかる場合があります）...").classes(
                    "text-grey-5 text-caption"
                )

        try:
            import sage as sage_pkg  # pip install sage-importance
            from sklearn.metrics import mean_squared_error as mse_fn

            if y is None:
                raise ValueError("y_train が取得できません")

            y_arr = np.asarray(y).ravel()
            X_np  = X_arr.copy()
            top_n_sage = min(30, X_np.shape[1])

            def _compute_sage():
                prog.value = 0.1
                # SAGEの損失関数: MSE
                imputer = sage_pkg.MarginalImputer(model, X_np)
                estimator = sage_pkg.PermutationEstimator(imputer, "mse")
                sage_values = estimator(X_np, y_arr, thresh=0.01)
                return sage_values

            sage_values = await run.io_bound(_compute_sage)
            prog.value = 0.95

            vals = np.asarray(sage_values.values)
            names_full = feat_names[:len(vals)]
            idx = np.argsort(np.abs(vals))[::-1]
            top_idx = idx[:top_n_sage]

            sage_container.clear()
            with sage_container:
                ui.label(f"🌿 SAGE Feature Importance (Top {top_n_sage})").classes(
                    "text-subtitle2 q-mb-xs"
                )
                colors = ["rgba(74,222,128,0.75)" if vals[i] > 0 else "rgba(248,113,113,0.75)"
                          for i in top_idx[::-1]]
                fig = go.Figure(go.Bar(
                    x=vals[top_idx][::-1],
                    y=[names_full[i] if i < len(names_full) else f"f{i}" for i in top_idx][::-1],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{vals[i]:.4f}" for i in top_idx][::-1],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=max(350, 22 * top_n_sage),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="SAGE値 (予測誤差への寄与)",
                    title="SAGE Feature Importance",
                )
                ui.plotly(fig).classes("full-width")

                rows = [
                    {"順位": r + 1,
                     "特徴量": names_full[top_idx[r]] if top_idx[r] < len(names_full) else f"f{top_idx[r]}",
                     "SAGE値": f"{vals[top_idx[r]]:+.6f}"}
                    for r in range(top_n_sage)
                ]
                cols = [{"name": k, "label": k, "field": k,
                         "align": "center" if k != "特徴量" else "left", "sortable": True}
                        for k in ["順位", "特徴量", "SAGE値"]]
                with ui.expansion("📋 数値テーブル", icon="table_chart").classes("full-width q-mt-md"):
                    ui.table(columns=cols, rows=rows).classes("full-width").props("dense flat bordered")

                ui.notify("✅ SAGE解析完了", type="positive")

        except ImportError:
            sage_container.clear()
            with sage_container:
                ui.label("⚠️ sage-importance パッケージが必要です").classes("text-amber text-caption")
                ui.label("pip install sage-importance").classes("text-caption text-grey-6 font-mono")
                ui.separator()
                _render_sage_fallback(ar, model, X, X_arr, feat_names, y)
        except Exception as ex:
            sage_container.clear()
            with sage_container:
                ui.label(f"❌ SAGE計算エラー: {ex}").classes("text-red text-caption")
                ui.separator()
                _render_sage_fallback(ar, model, X, X_arr, feat_names, y)

    ui.button("🌿 SAGE解析を実行", on_click=_run_sage).props(
        "unelevated color=green size=sm no-caps"
    )
    sage_container


def _render_sage_fallback(ar, model, X, X_arr, feat_names, y) -> None:
    """sage パッケージ未インストール時のフォールバック:
    Permutation Importance で同等の計算を行う。
    """
    import plotly.graph_objects as go
    from sklearn.inspection import permutation_importance

    ui.label("🔄 代替: Permutation Importance (SAGE に相当)").classes(
        "text-subtitle2 text-amber q-mt-sm q-mb-xs"
    )
    ui.label(
        "Permutation Importanceは各特徴量をランダムシャッフルしたときの性能低下を測定します。"
        "SAGE のShapley値計算の近似として利用できます。"
    ).classes("text-caption text-grey-5 q-mb-sm")

    perm_container = ui.column().classes("full-width")

    async def _calc_perm():
        perm_container.clear()
        with perm_container:
            ui.label("⏳ Permutation Importance 計算中...").classes("text-grey-5 text-caption")
        try:
            scoring = "r2" if ar.task == "regression" else "accuracy"
            y_arr = np.asarray(y).ravel() if y is not None else None
            if y_arr is None:
                raise ValueError("y_train が取得できません")

            def _compute():
                return permutation_importance(
                    model, X, y_arr, n_repeats=5, random_state=42, scoring=scoring,
                )

            perm_res = await run.io_bound(_compute)
            sorted_idx = np.argsort(perm_res.importances_mean)[::-1]
            top = min(20, len(sorted_idx))

            perm_container.clear()
            with perm_container:
                fig = go.Figure(go.Bar(
                    x=perm_res.importances_mean[sorted_idx[:top]][::-1],
                    y=[feat_names[i] if i < len(feat_names) else f"f{i}"
                       for i in sorted_idx[:top]][::-1],
                    orientation="h",
                    error_x=dict(
                        type="data",
                        array=perm_res.importances_std[sorted_idx[:top]][::-1],
                        visible=True,
                        color="rgba(255,255,255,0.5)",
                    ),
                    marker_color="rgba(74,222,128,0.7)",
                    text=[f"{perm_res.importances_mean[i]:.4f} ± {perm_res.importances_std[i]:.4f}"
                          for i in sorted_idx[:top]][::-1],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=max(300, 22 * top),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title=f"性能低下 ({scoring})",
                    title=f"Permutation Importance (±std, n_repeats=5)",
                )
                ui.plotly(fig).classes("full-width")
                ui.notify("✅ Permutation Importance 完了", type="positive")

        except Exception as ex:
            perm_container.clear()
            with perm_container:
                ui.label(f"❌ エラー: {ex}").classes("text-red text-caption")

    ui.button("🔄 Permutation Importance を計算", on_click=_calc_perm).props(
        "outline color=green size=sm no-caps"
    )
    perm_container


# ═══════════════════════════════════════════════════════════════
# SRI パネル
# ═══════════════════════════════════════════════════════════════
def _render_sri_panel(ar, model, X, X_arr, feat_names) -> None:
    """SHAP SRI (Synergy / Redundancy / Independence) 分解を描画する。

    参考文献: Ittner et al. "Feature Synergy, Redundancy, and Independence
    in Global Model Explanations using SHAP Vector Decomposition"
    arXiv:2107.12436 (2021)
    """
    import plotly.graph_objects as go

    ui.label("🔬 SRI分解 (Synergy / Redundancy / Independence)").classes(
        "text-subtitle1 text-bold q-mb-xs"
    )
    ui.label(
        "Ittner et al. (2021) の手法によりSHAPベクトルを3成分に分解します。\n"
        "・Synergy (相乗): 2特徴量が組み合わさって予測に寄与する成分\n"
        "・Redundancy (冗長): 2特徴量が同じ情報を重複して持つ成分\n"
        "・Independence (独立): 他特徴量と無関係に単独で寄与する成分"
    ).classes("text-caption text-grey-5 q-mb-md")

    sri_container = ui.column().classes("full-width")

    async def _run_sri():
        sri_container.clear()
        with sri_container:
            with ui.card().classes("q-pa-sm glass-card full-width"):
                prog = ui.linear_progress(value=0, show_value=False).props("color=purple rounded")
                lbl  = ui.label("⏳ SHAP → SRI分解を実行中...").classes("text-grey-5 text-caption")

        try:
            from backend.interpret.shap_explainer import ShapExplainer
            from backend.interpret.sri import SRIDecomposer

            def _compute():
                exp = ShapExplainer()
                shap_res = exp.explain(model, X, feature_names=feat_names)
                decomposer = SRIDecomposer(center=True)
                return shap_res, decomposer.decompose(shap_res)

            prog.value = 0.15
            lbl.text = "⏳ SHAP値を計算中..."
            shap_res, sri_res = await run.io_bound(_compute)
            prog.value = 0.9

            summary = sri_res.summary_df()
            top_n_sri = min(20, len(summary))
            df_top = summary.head(top_n_sri)

            total_syn, total_red, total_ind = sri_res.total_sri

            sri_container.clear()
            with sri_container:

                # サマリーカード
                with ui.row().classes("q-gutter-md q-mb-md"):
                    for val, lbl2, color in [
                        (f"{total_syn:.4f}",  "Synergy 合計",    "amber"),
                        (f"{total_red:.4f}",  "Redundancy 合計", "red"),
                        (f"{total_ind:.4f}",  "Independence 合計", "cyan"),
                    ]:
                        with ui.card().classes("q-pa-sm").style(
                            f"background:rgba(0,0,0,0.2); border-radius:8px; min-width:100px;"
                        ):
                            ui.label(val).classes(f"text-h6 text-bold text-{color}")
                            ui.label(lbl2).classes("text-caption text-grey-5")

                # スタック棒グラフ (Top 20)
                ui.label("📊 特徴量ごとの SRI 成分 (Top 20)").classes("text-subtitle2 q-mb-xs")
                fig_sri = go.Figure()
                fig_sri.add_trace(go.Bar(
                    y=df_top["feature"].values[::-1],
                    x=df_top["independence"].values[::-1],
                    orientation="h", name="Independence",
                    marker_color="rgba(0,212,255,0.7)",
                ))
                fig_sri.add_trace(go.Bar(
                    y=df_top["feature"].values[::-1],
                    x=df_top["synergy"].values[::-1],
                    orientation="h", name="Synergy",
                    marker_color="rgba(250,204,21,0.7)",
                ))
                fig_sri.add_trace(go.Bar(
                    y=df_top["feature"].values[::-1],
                    x=df_top["redundancy"].values[::-1],
                    orientation="h", name="Redundancy",
                    marker_color="rgba(248,113,113,0.7)",
                ))
                fig_sri.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    barmode="stack",
                    height=max(350, 25 * top_n_sri),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="SRI成分スコア",
                    title="SHAP SRI分解 (Independence / Synergy / Redundancy)",
                    legend=dict(orientation="h", y=1.05),
                )
                ui.plotly(fig_sri).classes("full-width")

                # Synergy ヒートマップ (Top 12)
                ui.separator()
                ui.label("🔥 Synergy ヒートマップ (Top 12ペア)").classes("text-subtitle2 q-mt-md q-mb-xs")
                top12 = summary.head(12)["feature"].tolist()
                syn_mat = sri_res.synergy_matrix
                fn = sri_res.feature_names
                fi_top = [fn.index(f) if f in fn else 0 for f in top12]
                sub_syn = syn_mat[np.ix_(fi_top, fi_top)]

                fig_heat = go.Figure(go.Heatmap(
                    z=sub_syn.tolist(),
                    x=top12,
                    y=top12,
                    colorscale="RdBu_r",
                    zmid=0,
                    colorbar=dict(title="Synergy"),
                ))
                fig_heat.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    height=450, margin=dict(l=10, r=10, t=30, b=10),
                    title="Synergy Matrix (Top 12特徴量)",
                    xaxis=dict(tickangle=-30),
                )
                ui.plotly(fig_heat).classes("full-width")

                # 数値テーブル
                with ui.expansion("📋 SRI 数値テーブル", icon="table_chart").classes(
                    "full-width q-mt-md"
                ):
                    rows = []
                    for _, row in df_top.iterrows():
                        rows.append({
                            "特徴量":       row["feature"],
                            "Independence": f"{row['independence']:.4f}",
                            "Synergy":      f"{row['synergy']:.4f}",
                            "Redundancy":   f"{row['redundancy']:.4f}",
                        })
                    cols = [
                        {"name": k, "label": k, "field": k,
                         "align": "left" if k == "特徴量" else "center", "sortable": True}
                        for k in ["特徴量", "Independence", "Synergy", "Redundancy"]
                    ]
                    ui.table(columns=cols, rows=rows).classes("full-width").props(
                        "dense flat bordered"
                    )

                ui.notify("✅ SRI分解完了", type="positive")

        except Exception as ex:
            sri_container.clear()
            with sri_container:
                ui.label(f"❌ SRI計算エラー: {ex}").classes("text-red text-caption")

    ui.button("🔬 SRI分解を実行", on_click=_run_sri).props(
        "unelevated color=purple size=sm no-caps"
    )
    sri_container
