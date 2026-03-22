"""
frontend_nicegui/components/results_tab.py

結果確認タブ：モデル比較・Fold別スコア・前処理後データ・SHAP解釈性
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui


def render_results_tab(state: dict[str, Any]) -> None:
    """結果確認タブ全体を描画する。"""

    ar = state.get("automl_result")

    if ar is None:
        with ui.card().classes("glass-card q-pa-xl full-width"):
            ui.icon("analytics", color="grey-7", size="xl").classes("q-mb-md")
            ui.label("解析結果がまだありません").classes("text-h6 text-grey-5")
            ui.label(
                "「📂 データ設定」タブでデータを読み込み、画面上部の「🚀 解析開始」ボタンを押してください。"
            ).classes("text-grey-6 q-mt-sm")
        return

    # ── 結果サマリーカード（ファーストビュー） ──
    with ui.card().classes("glass-card q-pa-md full-width q-mb-md"):
        # 行1: 最良モデル + スコア
        with ui.row().classes("items-center q-gutter-md"):
            ui.icon("emoji_events", color="amber", size="lg")
            ui.label(f"最良モデル: {ar.best_model_key}").classes("text-h5 text-bold hero-gradient")
            ui.badge(f"{ar.best_score:.4f}", color="cyan").props("floating")

        # 行2: 統計カード群
        scores = ar.model_scores if hasattr(ar, "model_scores") else {}
        n_models = len(scores)
        proc_X = getattr(ar, "processed_X", None)
        n_feats = proc_X.shape[1] if proc_X is not None and hasattr(proc_X, "shape") else "?"

        # 次点モデル差分
        runner_up_text = ""
        if n_models >= 2:
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            runner_up_key, runner_up_score = sorted_models[1]
            diff = ar.best_score - runner_up_score
            runner_up_text = f"2位: {runner_up_key} ({runner_up_score:.4f}, 差: {diff:+.4f})"

        with ui.row().classes("q-gutter-sm q-mt-sm"):
            for val, lbl, icon_name in [
                (ar.task, "タスク", "category"),
                (f"{ar.elapsed_seconds:.1f}秒", "所要時間", "timer"),
                (f"{n_models}個", "比較モデル数", "compare_arrows"),
                (str(n_feats), "特徴量数", "functions"),
            ]:
                with ui.card().classes("q-pa-xs").style(
                    "min-width: 90px; background: rgba(0,212,255,0.08); border-radius: 8px;"
                ):
                    with ui.row().classes("items-center q-gutter-xs"):
                        ui.icon(icon_name, size="xs", color="cyan")
                        ui.label(str(val)).classes("text-subtitle2 text-bold")
                    ui.label(lbl).classes("text-caption text-grey-5").style("font-size: 0.65rem;")

        if runner_up_text:
            ui.label(runner_up_text).classes("text-caption text-grey-5 q-mt-xs")

        # 行3: エクスポートボタン群
        with ui.row().classes("q-gutter-sm q-mt-sm"):
            async def _export_csv():
                """モデル比較表 + OOF予測をCSVでダウンロード。"""
                import io
                import csv

                buf = io.StringIO()
                writer = csv.writer(buf)

                # モデル比較
                writer.writerow(["=== モデル比較 ==="])
                writer.writerow(["モデル", "スコア"])
                for mk, ms in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    writer.writerow([mk, f"{ms:.6f}"])
                writer.writerow([])

                # OOF予測
                y_true = getattr(ar, "oof_true", None)
                y_pred = getattr(ar, "oof_predictions", None)
                if y_true is not None and y_pred is not None:
                    writer.writerow(["=== OOF予測 ==="])
                    writer.writerow(["実測値", "予測値", "残差"])
                    yt = np.asarray(y_true).ravel()
                    yp = np.asarray(y_pred).ravel()
                    for t, p in zip(yt, yp):
                        writer.writerow([f"{t:.6f}", f"{p:.6f}", f"{t - p:.6f}"])

                csv_text = buf.getvalue()
                ui.download(csv_text.encode("utf-8-sig"), f"chemai_results_{ar.best_model_key}.csv")
                ui.notify("📥 CSVダウンロードを開始しました", type="positive")

            ui.button("📥 結果CSV", on_click=_export_csv).props(
                "outline color=cyan size=sm no-caps icon=download"
            ).tooltip("モデル比較表 + OOF予測値をCSVダウンロード")

    # ── 警告 ──
    if ar.warnings:
        with ui.expansion(f"⚠️ 警告 ({len(ar.warnings)}件)", icon="warning").classes("full-width q-mb-md"):
            for w in ar.warnings:
                ui.label(f"⚠️ {w}").classes("text-amber text-caption")

    # ── 結果サブタブ ──
    with ui.tabs().classes("full-width").props("dense active-color=cyan indicator-color=cyan") as res_tabs:
        tab_eval = ui.tab("eval", label="📈 モデル評価", icon="leaderboard")
        tab_data = ui.tab("data", label="📊 前処理後データ", icon="table_chart")
        tab_interp = ui.tab("interp", label="🔬 モデル解釈性", icon="psychology")
        tab_batch = ui.tab("batch", label="🔮 バッチ予測", icon="batch_prediction")
        tab_report = ui.tab("report", label="📝 レポート", icon="summarize")

    with ui.tab_panels(res_tabs, value=tab_eval).classes("full-width"):

        # ════════════════════════════════════════════════════
        # サブタブ: モデル評価
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_eval):
            _render_model_evaluation(ar)

        # ════════════════════════════════════════════════════
        # サブタブ: 前処理後データ
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_data):
            _render_processed_data(ar)

        # ════════════════════════════════════════════════════
        # サブタブ: モデル解釈性
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_interp):
            _render_interpretability(ar, state)

        # ════════════════════════════════════════════════════
        # サブタブ: バッチ予測
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_batch):
            from frontend_nicegui.components.batch_predict_tab import render_batch_predict_tab
            render_batch_predict_tab(state)

        # ════════════════════════════════════════════════════
        # サブタブ: レポート生成
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_report):
            from frontend_nicegui.components.report_generator import render_report_tab
            render_report_tab(state)


# ================================================================
# モデル評価
# ================================================================
def _render_model_evaluation(ar) -> None:
    """モデルスコア比較テーブルとFold別スコア"""

    ui.label("🏆 モデル比較").classes("text-subtitle1")
    ui.label(f"スコアリング: {ar.scoring}").classes("text-caption text-grey-5 q-mb-md")

    # ── スコア比較テーブル ──
    rows = []
    for key, score in sorted(ar.model_scores.items(), key=lambda x: -x[1]):
        detail = ar.model_details.get(key, {})
        is_best = key == ar.best_model_key
        rows.append({
            "モデル": f"🏆 {key}" if is_best else key,
            "平均スコア": f"{score:.4f}",
            "標準偏差": f"±{detail.get('std', 0):.4f}",
            "学習時間(秒)": f"{detail.get('fit_time', 0):.2f}",
            "状態": "🏆 最良" if is_best else "✅",
        })

    columns = [
        {"name": c, "label": c, "field": c,
         "align": "left" if c in ("モデル", "状態") else "center",
         "sortable": True}
        for c in ["モデル", "平均スコア", "標準偏差", "学習時間(秒)", "状態"]
    ]
    ui.table(columns=columns, rows=rows).classes("full-width").props("dense flat bordered")

    # ── Fold別スコア ──
    ui.separator()
    with ui.expansion("📊 Fold別スコア詳細", icon="bar_chart").classes("full-width q-mt-md"):
        for key, detail in ar.model_details.items():
            fold_scores = detail.get("fold_scores", [])
            if fold_scores:
                with ui.card().classes("glass-card q-pa-sm q-mb-sm"):
                    ui.label(f"{'🏆 ' if key == ar.best_model_key else ''}{key}").classes(
                        "text-subtitle2 text-bold" if key == ar.best_model_key else "text-subtitle2"
                    )
                    fold_text = " | ".join(
                        f"Fold{i+1}: {s:.4f}" for i, s in enumerate(fold_scores)
                    )
                    ui.label(fold_text).classes("text-caption text-grey-5")

    # ── OOF予測サマリー ──
    if ar.oof_predictions is not None and ar.oof_true is not None:
        ui.separator()
        ui.label("📈 Out-of-Fold予測サマリー").classes("text-subtitle2 q-mt-md")
        try:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            if ar.task == "regression":
                r2 = r2_score(ar.oof_true, ar.oof_predictions)
                rmse = mean_squared_error(ar.oof_true, ar.oof_predictions, squared=False)
                mae = mean_absolute_error(ar.oof_true, ar.oof_predictions)
                with ui.row().classes("q-gutter-md"):
                    for val, lbl in [
                        (f"{r2:.4f}", "R² (OOF)"),
                        (f"{rmse:.4f}", "RMSE (OOF)"),
                        (f"{mae:.4f}", "MAE (OOF)"),
                    ]:
                        with ui.card().classes("glass-card q-pa-sm"):
                            ui.label(val).classes("text-h6 text-bold hero-gradient")
                            ui.label(lbl).classes("text-caption text-grey-5")
            else:
                from sklearn.metrics import accuracy_score, f1_score
                acc = accuracy_score(ar.oof_true, ar.oof_predictions)
                f1 = f1_score(ar.oof_true, ar.oof_predictions, average="weighted", zero_division=0)
                with ui.row().classes("q-gutter-md"):
                    for val, lbl in [
                        (f"{acc:.4f}", "Accuracy (OOF)"),
                        (f"{f1:.4f}", "F1-weighted (OOF)"),
                    ]:
                        with ui.card().classes("glass-card q-pa-sm"):
                            ui.label(val).classes("text-h6 text-bold hero-gradient")
                            ui.label(lbl).classes("text-caption text-grey-5")
        except Exception as ex:
            ui.label(f"OOFメトリクス計算エラー: {ex}").classes("text-caption text-red")

        # ── 残差分析プロット ──
        if ar.task == "regression":
            ui.separator()
            with ui.expansion("📉 残差分析（OOF）", icon="scatter_plot").classes("full-width q-mt-sm"):
                _render_residual_analysis(ar)


# ================================================================
# 前処理後データ
# ================================================================
def _render_processed_data(ar) -> None:
    """前処理後のデータテーブルと統計量"""

    proc_X = getattr(ar, "processed_X", None)
    if proc_X is None or not hasattr(proc_X, "shape"):
        ui.label("⚠️ 前処理後データが取得できませんでした").classes("text-amber")
        return

    ui.label("📊 モデルに入力された最終データ").classes("text-subtitle1")
    ui.label(
        "カテゴリエンコーディング・欠損補完・スケーリング・変数選択などが完了した後の、"
        "実際にモデルに渡された数値データです。"
    ).classes("text-caption text-grey-5 q-mb-md")

    # メトリクスカード
    with ui.row().classes("q-gutter-md"):
        for val, lbl in [
            (f"{proc_X.shape[0]:,}", "サンプル数"),
            (f"{proc_X.shape[1]:,}", "特徴量数"),
            (f"{int(proc_X.isnull().sum().sum()):,}" if hasattr(proc_X, "isnull") else "0", "欠損値"),
        ]:
            with ui.card().classes("glass-card q-pa-sm"):
                ui.label(val).classes("text-h6 text-bold hero-gradient")
                ui.label(lbl).classes("text-caption text-grey-5")

    # データプレビュー
    ui.separator()
    ui.label("🔍 データプレビュー（先頭50行）").classes("text-subtitle2 q-mt-md")
    preview = proc_X.head(50)
    columns = [
        {"name": col, "label": col, "field": col, "align": "left", "sortable": True}
        for col in preview.columns[:20]  # 表示は20列まで
    ]
    rows = []
    for _, row in preview.iterrows():
        row_dict = {}
        for col in preview.columns[:20]:
            v = row[col]
            if pd.isna(v):
                row_dict[col] = "—"
            elif isinstance(v, float):
                row_dict[col] = f"{v:.4g}"
            else:
                row_dict[col] = str(v)
        rows.append(row_dict)
    ui.table(columns=columns, rows=rows).classes("full-width").props("dense flat bordered")

    if proc_X.shape[1] > 20:
        ui.label(f"... 他 {proc_X.shape[1] - 20} 列").classes("text-caption text-grey-6")

    # 基本統計量
    with ui.expansion("📐 基本統計量", icon="calculate").classes("full-width q-mt-md"):
        desc = proc_X.describe().T.round(4).reset_index()
        desc.rename(columns={"index": "特徴量"}, inplace=True)
        desc_cols = [
            {"name": c, "label": c, "field": c, "align": "left" if c == "特徴量" else "center", "sortable": True}
            for c in desc.columns
        ]
        desc_rows = desc.head(50).to_dict("records")
        for row in desc_rows:
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = f"{v:.4g}"
        ui.table(columns=desc_cols, rows=desc_rows).classes("full-width").props("dense flat bordered")

    # CSVダウンロード
    csv_data = proc_X.to_csv(index=False)
    ui.button(
        "📥 前処理後データをCSVダウンロード",
        on_click=lambda: ui.download(csv_data.encode("utf-8"), "processed_features.csv"),
    ).props("outline color=cyan").classes("q-mt-md")


# ================================================================
# モデル解釈性
# ================================================================
def _render_interpretability(ar, state: dict) -> None:
    """SHAP・Feature Importance等"""

    model = getattr(ar, "best_pipeline", None)
    X = getattr(ar, "X_train", None)
    y = getattr(ar, "y_train", None)

    if model is None or X is None:
        ui.label("⚠️ モデルまたはデータが取得できませんでした").classes("text-amber")
        return

    ui.label("🔬 モデル解釈性").classes("text-subtitle1")
    ui.label("Feature Importanceとモデルの重要特徴量を表示します。").classes("text-caption text-grey-5 q-mb-md")

    # ── Feature Importance (tree-based models) ──
    try:
        # パイプラインの最終モデルからfeature importanceを取得
        estimator = model
        if hasattr(model, "steps"):
            estimator = model.steps[-1][1]
            if hasattr(estimator, "steps"):
                estimator = estimator.steps[-1][1]

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_

            # 特徴量名の取得
            try:
                feat_names = model[:-1].get_feature_names_out().tolist()
            except Exception:
                if hasattr(X, "columns"):
                    feat_names = list(X.columns)
                else:
                    feat_names = [f"feature_{i}" for i in range(len(importances))]

            # 長さの調整
            if len(feat_names) != len(importances):
                feat_names = [f"feature_{i}" for i in range(len(importances))]

            # ソート
            indices = np.argsort(importances)[::-1]
            top_n = min(20, len(indices))

            ui.label("📊 Feature Importance (Top 20)").classes("text-subtitle2 q-mt-md")

            rows = []
            for i in range(top_n):
                idx = indices[i]
                rows.append({
                    "順位": i + 1,
                    "特徴量": feat_names[idx] if idx < len(feat_names) else f"feature_{idx}",
                    "重要度": f"{importances[idx]:.4f}",
                    "バー": "█" * int(importances[idx] / max(importances) * 20),
                })

            fi_columns = [
                {"name": "順位", "label": "#", "field": "順位", "align": "center"},
                {"name": "特徴量", "label": "特徴量", "field": "特徴量", "align": "left"},
                {"name": "重要度", "label": "重要度", "field": "重要度", "align": "center"},
                {"name": "バー", "label": "", "field": "バー", "align": "left"},
            ]
            ui.table(columns=fi_columns, rows=rows).classes("full-width").props("dense flat bordered")

        elif hasattr(estimator, "coef_"):
            # 線形モデル
            coefs = estimator.coef_.ravel() if hasattr(estimator.coef_, "ravel") else estimator.coef_

            try:
                feat_names = model[:-1].get_feature_names_out().tolist()
            except Exception:
                feat_names = list(X.columns) if hasattr(X, "columns") else [f"feature_{i}" for i in range(len(coefs))]

            if len(feat_names) != len(coefs):
                feat_names = [f"feature_{i}" for i in range(len(coefs))]

            indices = np.argsort(np.abs(coefs))[::-1]
            top_n = min(20, len(indices))

            ui.label("📊 回帰係数 (Top 20)").classes("text-subtitle2 q-mt-md")

            rows = []
            for i in range(top_n):
                idx = indices[i]
                rows.append({
                    "順位": i + 1,
                    "特徴量": feat_names[idx] if idx < len(feat_names) else f"feature_{idx}",
                    "係数": f"{coefs[idx]:.4f}",
                    "絶対値": f"{abs(coefs[idx]):.4f}",
                })

            coef_columns = [
                {"name": "順位", "label": "#", "field": "順位", "align": "center"},
                {"name": "特徴量", "label": "特徴量", "field": "特徴量", "align": "left"},
                {"name": "係数", "label": "係数", "field": "係数", "align": "center"},
                {"name": "絶対値", "label": "|係数|", "field": "絶対値", "align": "center"},
            ]
            ui.table(columns=coef_columns, rows=rows).classes("full-width").props("dense flat bordered")

        else:
            ui.label("ℹ️ このモデルタイプはFeature Importance / 回帰係数を直接表示できません。").classes("text-grey-5")
            ui.label("SHAP解析を利用してください。").classes("text-caption text-grey-6")

    except Exception as ex:
        ui.label(f"Feature Importance取得エラー: {ex}").classes("text-caption text-red")

    # ── Permutation Importance ──
    ui.separator()
    with ui.expansion("🔀 Permutation Importance", icon="shuffle").classes("full-width q-mt-md"):
        ui.label(
            "Permutation Importanceは計算に時間がかかるため、ボタンクリックで実行します。"
        ).classes("text-caption text-grey-5 q-mb-sm")

        perm_container = ui.column().classes("full-width")

        async def _calc_perm_importance():
            perm_container.clear()
            with perm_container:
                ui.label("⏳ 計算中...").classes("text-grey-5")
            try:
                from sklearn.inspection import permutation_importance
                proc_X = getattr(ar, "processed_X", X)
                scoring = "r2" if ar.task == "regression" else "accuracy"
                perm_result = permutation_importance(
                    model, proc_X, y, n_repeats=5, random_state=42, scoring=scoring
                )
                sorted_idx = perm_result.importances_mean.argsort()[::-1]

                try:
                    feat_names_p = list(proc_X.columns) if hasattr(proc_X, "columns") else [
                        f"feature_{i}" for i in range(proc_X.shape[1])
                    ]
                except Exception:
                    feat_names_p = [f"feature_{i}" for i in range(len(perm_result.importances_mean))]

                perm_container.clear()
                with perm_container:
                    top_n = min(15, len(sorted_idx))
                    rows = []
                    for i in range(top_n):
                        idx = sorted_idx[i]
                        rows.append({
                            "順位": i + 1,
                            "特徴量": feat_names_p[idx] if idx < len(feat_names_p) else f"feature_{idx}",
                            "平均重要度": f"{perm_result.importances_mean[idx]:.4f}",
                            "標準偏差": f"±{perm_result.importances_std[idx]:.4f}",
                        })
                    pi_columns = [
                        {"name": c, "label": c, "field": c, "align": "left" if c == "特徴量" else "center"}
                        for c in ["順位", "特徴量", "平均重要度", "標準偏差"]
                    ]
                    ui.table(columns=pi_columns, rows=rows).classes("full-width").props("dense flat bordered")

            except Exception as ex:
                perm_container.clear()
                with perm_container:
                    ui.label(f"エラー: {ex}").classes("text-red text-caption")

        ui.button(
            "🔀 Permutation Importance を計算", on_click=_calc_perm_importance
        ).props("outline color=purple size=sm")

    # ── SHAP 解析 ──
    ui.separator()
    with ui.expansion("🔍 SHAP 解析", icon="insights").classes("full-width q-mt-md"):
        ui.label(
            "SHAP値で各特徴量のモデル予測への寄与を可視化します。"
            "計算にはshapライブラリが必要です。"
        ).classes("text-caption text-grey-5 q-mb-sm")

        shap_container = ui.column().classes("full-width")

        async def _calc_shap():
            shap_container.clear()
            with shap_container:
                ui.label("⏳ SHAP値を計算中...").classes("text-grey-5")
            try:
                from backend.interpret.shap_explainer import ShapExplainer, ShapResult
                import plotly.graph_objects as go  # noqa: F811

                proc_X = getattr(ar, "processed_X", X)
                if hasattr(proc_X, "values"):
                    proc_X_arr = proc_X.values
                else:
                    proc_X_arr = np.asarray(proc_X)

                feat_names_shap = list(proc_X.columns) if hasattr(proc_X, "columns") else [
                    f"f{i}" for i in range(proc_X_arr.shape[1])
                ]

                explainer = ShapExplainer()
                shap_result = explainer.explain(model, proc_X, feature_names=feat_names_shap)

                # 特徴量重要度（SHAP ベース）
                fi_df = shap_result.feature_importance()
                top_features = fi_df.head(20)

                shap_container.clear()
                with shap_container:
                    # ── SHAP Summary Bar Plot ──
                    ui.label("📊 SHAP Feature Importance (Top 20)").classes("text-subtitle2 q-mb-sm")
                    fig_bar = go.Figure(go.Bar(
                        x=top_features["importance"].values[::-1],
                        y=top_features["feature"].values[::-1],
                        orientation="h",
                        marker_color="rgba(0,212,255,0.7)",
                    ))
                    fig_bar.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=max(300, 20 * len(top_features)),
                        margin=dict(l=10, r=10, t=30, b=10),
                        title="平均|SHAP値|",
                        xaxis_title="平均|SHAP値|",
                    )
                    ui.plotly(fig_bar).classes("full-width")

                    # ── SHAP Beeswarm (dot plot approximation via scatter) ──
                    ui.separator()
                    ui.label("🐝 SHAP Beeswarm Plot (Top 10)").classes("text-subtitle2 q-mt-md q-mb-sm")
                    top10_feats = fi_df.head(10)["feature"].tolist()
                    sv = shap_result.shap_values
                    if sv.ndim == 3:
                        sv = sv[:, :, 0]

                    fig_bee = go.Figure()
                    for i, feat in enumerate(reversed(top10_feats)):
                        feat_idx = feat_names_shap.index(feat) if feat in feat_names_shap else i
                        if feat_idx < sv.shape[1]:
                            shap_vals = sv[:, feat_idx]
                            feat_vals = proc_X_arr[:, feat_idx]
                            fig_bee.add_trace(go.Scatter(
                                x=shap_vals,
                                y=[feat] * len(shap_vals),
                                mode="markers",
                                marker=dict(
                                    size=4,
                                    color=feat_vals,
                                    colorscale="RdBu_r",
                                    opacity=0.6,
                                    showscale=(i == 0),
                                    colorbar=dict(title="特徴量値") if i == 0 else None,
                                ),
                                name=feat,
                                showlegend=False,
                            ))
                    fig_bee.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=max(300, 35 * len(top10_feats)),
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title="SHAP値",
                    )
                    ui.plotly(fig_bee).classes("full-width")

                    # ── Waterfall (サンプル0) ──
                    ui.separator()
                    ui.label("💧 Waterfall Plot (サンプル #0)").classes("text-subtitle2 q-mt-md q-mb-sm")
                    sample_sv = sv[0]
                    sorted_idx_w = np.argsort(np.abs(sample_sv))[::-1]
                    top_w = min(15, len(sorted_idx_w))

                    waterfall_feats = [feat_names_shap[sorted_idx_w[i]] if sorted_idx_w[i] < len(feat_names_shap) else f"f{sorted_idx_w[i]}" for i in range(top_w)]
                    waterfall_vals = [sample_sv[sorted_idx_w[i]] for i in range(top_w)]

                    exp_val = shap_result.expected_value
                    if hasattr(exp_val, "__len__"):
                        exp_val = float(exp_val[0]) if len(exp_val) > 0 else 0.0
                    else:
                        exp_val = float(exp_val)

                    fig_wf = go.Figure(go.Waterfall(
                        name="SHAP",
                        orientation="h",
                        y=waterfall_feats[::-1],
                        x=waterfall_vals[::-1],
                        connector=dict(line=dict(color="rgba(255,255,255,0.2)")),
                        increasing=dict(marker=dict(color="rgba(74,222,128,0.7)")),
                        decreasing=dict(marker=dict(color="rgba(248,113,113,0.7)")),
                        base=exp_val,
                    ))
                    fig_wf.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=max(300, 25 * top_w),
                        margin=dict(l=10, r=10, t=30, b=10),
                        title=f"ベースライン: {exp_val:.4f}",
                        xaxis_title="予測への寄与",
                    )
                    ui.plotly(fig_wf).classes("full-width")

                    # ── Dependence Plot (Top 1特徴量) ──
                    if len(top10_feats) > 0:
                        ui.separator()
                        top1_feat = top10_feats[0]
                        top1_idx = feat_names_shap.index(top1_feat) if top1_feat in feat_names_shap else 0
                        ui.label(f"📈 Dependence Plot: {top1_feat}").classes("text-subtitle2 q-mt-md q-mb-sm")

                        fig_dep = go.Figure(go.Scatter(
                            x=proc_X_arr[:, top1_idx],
                            y=sv[:, top1_idx],
                            mode="markers",
                            marker=dict(
                                size=5,
                                color=sv[:, top1_idx],
                                colorscale="RdBu_r",
                                opacity=0.7,
                                showscale=True,
                                colorbar=dict(title="SHAP値"),
                            ),
                        ))
                        fig_dep.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=350,
                            margin=dict(l=10, r=10, t=30, b=10),
                            xaxis_title=top1_feat,
                            yaxis_title=f"SHAP値 ({top1_feat})",
                        )
                        ui.plotly(fig_dep).classes("full-width")

                    ui.notify("✅ SHAP解析完了", type="positive")

            except ImportError as ie:
                shap_container.clear()
                with shap_container:
                    ui.label(f"⚠️ {ie}").classes("text-amber text-caption")
                    ui.label("pip install shap でインストールしてください。").classes("text-caption text-grey-6")
            except Exception as ex:
                shap_container.clear()
                with shap_container:
                    ui.label(f"SHAP計算エラー: {ex}").classes("text-red text-caption")

        ui.button(
            "🔍 SHAP 解析を実行", on_click=_calc_shap
        ).props("outline color=cyan size=sm no-caps")

    # ── PDP (Partial Dependence Plot) ──
    ui.separator()
    with ui.expansion("📉 Partial Dependence Plot (PDP)", icon="timeline").classes("full-width q-mt-md"):
        ui.label(
            "特定の特徴量が予測にどう影響するかを可視化します（他の特徴量を平均化）。"
        ).classes("text-caption text-grey-5 q-mb-sm")

        pdp_container = ui.column().classes("full-width")

        async def _calc_pdp():
            pdp_container.clear()
            with pdp_container:
                ui.label("⏳ PDP計算中...").classes("text-grey-5")
            try:
                from sklearn.inspection import partial_dependence
                import plotly.graph_objects as go  # noqa: F811

                proc_X = getattr(ar, "processed_X", X)
                feat_names_pdp = list(proc_X.columns) if hasattr(proc_X, "columns") else [
                    f"f{i}" for i in range(proc_X.shape[1])
                ]

                # Feature Importanceが高い上位4特徴量
                if hasattr(estimator, "feature_importances_"):
                    imp = estimator.feature_importances_
                    top_idx = np.argsort(imp)[::-1][:4]
                else:
                    top_idx = list(range(min(4, len(feat_names_pdp))))

                pdp_container.clear()
                with pdp_container:
                    ui.label("📉 PDP (Top 4 特徴量)").classes("text-subtitle2 q-mb-sm")

                    for idx in top_idx:
                        feat_name = feat_names_pdp[idx] if idx < len(feat_names_pdp) else f"f{idx}"
                        try:
                            pdp_result = partial_dependence(
                                model, proc_X, features=[idx],
                                grid_resolution=50, kind="average",
                            )
                            grid = pdp_result["grid_values"][0]
                            avg_pred = pdp_result["average"][0]

                            fig_pdp = go.Figure(go.Scatter(
                                x=grid, y=avg_pred,
                                mode="lines",
                                line=dict(color="rgba(0,212,255,0.8)", width=2),
                                fill="tozeroy",
                                fillcolor="rgba(0,212,255,0.08)",
                            ))
                            fig_pdp.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                height=250,
                                margin=dict(l=10, r=10, t=30, b=10),
                                title=f"PDP: {feat_name}",
                                xaxis_title=feat_name,
                                yaxis_title="予測値",
                            )
                            ui.plotly(fig_pdp).classes("full-width q-mb-sm")
                        except Exception:
                            pass

                    ui.notify("✅ PDP計算完了", type="positive")

            except ImportError:
                pdp_container.clear()
                with pdp_container:
                    ui.label("⚠️ sklearn.inspection が必要です").classes("text-amber text-caption")
            except Exception as ex:
                pdp_container.clear()
                with pdp_container:
                    ui.label(f"PDP計算エラー: {ex}").classes("text-red text-caption")

        ui.button(
            "📉 PDP を計算", on_click=_calc_pdp
        ).props("outline color=teal size=sm no-caps")


# ================================================================
# 残差分析（OOF予測）
# ================================================================
def _render_residual_analysis(ar) -> None:
    """OOF実測vs予測の残差分析プロット群。"""
    y_true = ar.oof_true
    y_pred = ar.oof_predictions

    if y_true is None or y_pred is None:
        ui.label("OOFデータが利用できません").classes("text-grey")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from sklearn.metrics import mean_absolute_percentage_error

        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        residuals = y_t - y_p

        # ── 3プロットを横並び ──
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["実測 vs 予測", "残差ヒストグラム", "残差 vs 予測値"],
        )

        # 1. 実測 vs 予測 散布図
        fig.add_trace(
            go.Scatter(
                x=y_t, y=y_p, mode="markers",
                marker=dict(size=4, color="rgba(0,212,255,0.6)"),
                name="データ点",
            ),
            row=1, col=1,
        )
        # y=x 基準線
        rng = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        fig.add_trace(
            go.Scatter(
                x=rng, y=rng, mode="lines",
                line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
                name="y=x",
            ),
            row=1, col=1,
        )

        # 2. 残差ヒストグラム
        fig.add_trace(
            go.Histogram(
                x=residuals, nbinsx=30,
                marker_color="rgba(123,47,247,0.6)",
                name="残差分布",
            ),
            row=1, col=2,
        )

        # 3. 残差 vs 予測値
        fig.add_trace(
            go.Scatter(
                x=y_p, y=residuals, mode="markers",
                marker=dict(size=4, color="rgba(74,222,128,0.6)"),
                name="残差",
            ),
            row=1, col=3,
        )
        # 零線
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=1, col=3)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=10, r=10, t=40, b=30),
            showlegend=False,
        )
        fig.update_xaxes(title_text="実測値", row=1, col=1)
        fig.update_yaxes(title_text="予測値", row=1, col=1)
        fig.update_xaxes(title_text="残差", row=1, col=2)
        fig.update_xaxes(title_text="予測値", row=1, col=3)
        fig.update_yaxes(title_text="残差", row=1, col=3)

        ui.plotly(fig).classes("full-width")

        # ── 統計量カード ──
        try:
            mape = mean_absolute_percentage_error(y_t, y_p) * 100
        except Exception:
            mape = float("nan")
        max_res = float(np.max(np.abs(residuals)))
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))

        # 正規性検定
        try:
            from scipy.stats import shapiro
            if len(residuals) <= 5000:
                _, p_sw = shapiro(residuals)
            else:
                _, p_sw = shapiro(np.random.choice(residuals, 5000, replace=False))
            normality_text = f"p={p_sw:.4f} ({'正規分布' if p_sw > 0.05 else '非正規分布'})"
        except Exception:
            normality_text = "計算不可"

        with ui.row().classes("q-gutter-sm q-mt-sm"):
            for val, lbl in [
                (f"{mape:.1f}%", "MAPE"),
                (f"{max_res:.4g}", "最大|残差|"),
                (f"{mean_res:.4g}", "残差平均"),
                (f"{std_res:.4g}", "残差σ"),
                (normality_text, "Shapiro-Wilk"),
            ]:
                with ui.card().classes("glass-card q-pa-xs"):
                    ui.label(str(val)).classes("text-subtitle2 text-bold hero-gradient")
                    ui.label(lbl).classes("text-caption text-grey-5").style("font-size: 0.7rem;")

    except ImportError:
        ui.label("Plotlyが必要です: pip install plotly").classes("text-amber")
    except Exception as ex:
        ui.label(f"残差分析エラー: {ex}").classes("text-red text-caption")
