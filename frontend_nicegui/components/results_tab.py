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

    # ── サマリーヘッダー ──
    with ui.card().classes("glass-card q-pa-md full-width q-mb-md"):
        with ui.row().classes("items-center q-gutter-md"):
            ui.icon("emoji_events", color="amber", size="lg")
            ui.label(f"最良モデル: {ar.best_model_key}").classes("text-h5 text-bold hero-gradient")
            ui.badge(f"{ar.best_score:.4f}", color="cyan").props("floating")
            ui.label(f"タスク: {ar.task}").classes("text-grey-5")
            ui.label(f"所要時間: {ar.elapsed_seconds:.1f}秒").classes("text-grey-5")

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
