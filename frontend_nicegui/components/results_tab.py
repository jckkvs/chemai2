"""
frontend_nicegui/components/results_tab.py

結果確認タブ：モデル比較・Fold別スコア・前処理後データ・SHAP解釈性
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

from frontend_nicegui.components.results_tab_extras import (
    _render_model_overview,
    _render_per_model_tabs,
    _render_pred_actual_inline,
    _render_sample_table_inline,
    _render_extra_visualizations,
)


def render_results_tab(state: dict[str, Any]) -> None:
    """結果確認タブ全体を描画する。"""

    # ── 複数セット対応: セット選択UI ──
    all_results = state.get("automl_results", {})
    single_ar = state.get("automl_result")

    # 結果が全くない場合
    if not all_results and single_ar is None:
        with ui.card().classes("glass-card q-pa-xl full-width animate-slide-up items-center justify-center text-center").props('data-testid="no-results-card"'):
            ui.icon("analytics", color="grey-7", size="xl").classes("q-mb-md").props('aria-hidden="true"')
            ui.label("解析結果がまだありません").classes("text-h6 text-grey-5").props('role="heading" aria-level="2"')
            ui.label(
                "「📂 データ設定」タブでデータを読み込み、画面上部の「🚀 解析開始」ボタンを押してください。"
            ).classes("text-grey-6 q-mt-sm")
        return

    # 成功したセットのみ抽出
    success_results = {k: v for k, v in all_results.items() if v is not None}
    if not success_results and single_ar:
        success_results = {"デフォルト": single_ar}

    # ── セット切替タブ（2セット以上の場合） ──
    set_names = list(success_results.keys())
    if not set_names:
        return

    current_view = state.get("_viewing_set", state.get("best_set_name", set_names[0]))
    if current_view not in success_results:
        current_view = set_names[0]

    if len(set_names) >= 2:
        # セット選択タブ
        with ui.card().classes("glass-card q-pa-md full-width q-mb-md"):
            with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
                ui.icon("layers", color="cyan")
                ui.label("記述子セット別 結果").classes("text-h6")
                ui.badge(f"{len(set_names)}セット", color="teal").props("dense")

            set_tab_keys = []
            with ui.tabs().classes("full-width").props(
                "dense no-caps active-color=cyan indicator-color=cyan scrollable"
            ) as set_result_tabs:
                for sn in set_names:
                    sr = success_results[sn]
                    is_best = (sn == state.get("best_set_name", ""))
                    label = f"🏆 {sn} ({sr.best_score:.4f})" if is_best else f"{sn} ({sr.best_score:.4f})"
                    key = f"res_set_{sn}"
                    set_tab_keys.append(key)
                    ui.tab(key, label=label)

            best_tab_key = f"res_set_{current_view}"
            with ui.tab_panels(set_result_tabs, value=best_tab_key).classes("full-width bg-transparent"):
                for sn in set_names:
                    key = f"res_set_{sn}"
                    with ui.tab_panel(key):
                        _render_single_result(success_results[sn], state)
    else:
        ar = success_results.get(current_view, single_ar)
        _render_single_result(ar, state)
        return

    # ── 複数セットの場合はここでreturn済み（タブ内に_render_single_resultが呼ばれている） ──
    return


def _render_single_result(ar, state: dict) -> None:
    """単一セットの結果詳細を描画する。"""
    scores = ar.model_scores if hasattr(ar, "model_scores") else {}
    with ui.card().classes("glass-card q-pa-md full-width q-mb-md animate-slide-up best-model-glow").props('data-testid="best-model-summary-card"'):
        # 行1: 最良モデル + スコア
        with ui.row().classes("items-center q-gutter-md"):
            ui.icon("emoji_events", color="amber", size="lg").props('aria-label="最良モデル"')
            ui.label(f"最良モデル: {ar.best_model_key}").classes("text-h5 text-bold hero-gradient").props('role="heading" aria-level="2" id="best-model-title"')
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
            for i, (val, lbl, icon_name) in enumerate([
                (ar.task, "タスク", "category"),
                (f"{ar.elapsed_seconds:.1f}秒", "所要時間", "timer"),
                (f"{n_models}個", "比較モデル数", "compare_arrows"),
                (str(n_feats), "特徴量数", "functions"),
            ]):
                delay_class = f"delay-{(i+1)*100}"
                with ui.card().classes(f"q-pa-xs animate-slide-up hover-bounce {delay_class}").style(
                    "min-width: 90px; background: rgba(0,212,255,0.08); border-radius: 8px;"
                ):
                    with ui.row().classes("items-center q-gutter-xs"):
                        ui.icon(icon_name, size="xs", color="cyan")
                        ui.label(str(val)).classes("text-subtitle2 text-bold")
                    ui.label(lbl).classes("text-caption text-grey-5").style("font-size: 0.82rem;")

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
                'outline color=cyan size=sm no-caps icon=download data-testid="export-csv-btn" aria-label="結果CSVをダウンロード"'
            ).tooltip("モデル比較表 + OOF予測値をCSVダウンロード")

            # ── タスク3-2: 逆解析CTAボタン ──
            def _go_inverse():
                """逆解析タブに遷移し、最良モデルを自動設定する。"""
                # 逆解析設定にモデル情報を事前セット
                if "_inv" not in state:
                    state["_inv"] = {
                        "target_mode": "range",
                        "target_min": None,
                        "target_max": None,
                        "constraints": {},
                        "method": "random",
                        "method_params": {},
                        "results": None,
                    }
                best_key = ar.best_model_key if hasattr(ar, "best_model_key") else None
                if best_key:
                    state["_inv"]["selected_model"] = best_key
                # タブ遷移
                switch_fn = state.get("_switch_to_inverse")
                if switch_fn:
                    switch_fn()
                    # 逆解析タブを再描画して最新モデル情報を反映
                    refresh_inv = state.get("_refresh_inverse")
                    if refresh_inv:
                        try:
                            refresh_inv()
                        except Exception:
                            pass
                    ui.notify(
                        f"🔮 {best_key} で逆解析を設定できます",
                        type="info", timeout=3000,
                    )
                else:
                    ui.notify("🔮 逆解析タブに移動してください", type="info")

            ui.button("🔮 このモデルで逆解析", on_click=_go_inverse).props(
                'unelevated color=purple size=sm no-caps icon=find_replace '
                'data-testid="inverse-cta-btn" aria-label="逆解析を開始"'
            ).tooltip(
                f"{ar.best_model_key} の学習済みモデルを使って、"
                "目標物性を持つ最適な説明変数値を探索します"
            ).classes("text-bold")

    # ── 警告 ──
    if ar.warnings:
        with ui.expansion(f"⚠️ 警告 ({len(ar.warnings)}件)", icon="warning").classes("full-width q-mb-md animate-shake"):
            for w in ar.warnings:
                ui.label(f"⚠️ {w}").classes("text-amber text-caption")

    # ── 結果サブタブ ──
    with ui.tabs().classes("full-width").props(
        "dense active-color=cyan indicator-color=cyan scrollable"
    ) as res_tabs:
        tab_overview = ui.tab("overview", label="🏆 全モデル概要",    icon="leaderboard")
        tab_permodel = ui.tab("permodel", label="📊 モデル別詳細",    icon="analytics")
        tab_pred     = ui.tab("pred",     label="📈 予測実測プロット", icon="scatter_plot")
        tab_table    = ui.tab("table",    label="📋 データ点表",       icon="table_rows")
        tab_tuning   = ui.tab("tuning",   label="🎯 チューニング",     icon="tune")
        tab_data     = ui.tab("data",     label="🔢 前処理後データ",   icon="table_chart")
        tab_interp   = ui.tab("interp",   label="🔬 解釈性・重要度",   icon="psychology")
        tab_extra    = ui.tab("extra",    label="🎨 追加可視化",       icon="bar_chart")
        tab_batch    = ui.tab("batch",    label="🔮 バッチ予測",       icon="batch_prediction")
        tab_report   = ui.tab("report",   label="📝 レポート",         icon="summarize")

    with ui.tab_panels(res_tabs, value=tab_overview).classes("full-width"):

        # ════════════════════════════════════════════════════
        # 全モデル概要
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_overview):
            _render_model_overview(ar)

        # ════════════════════════════════════════════════════
        # モデル別詳細
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_permodel):
            _render_per_model_tabs(ar)

        # ════════════════════════════════════════════════════
        # 予測実測プロット
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_pred):
            _render_pred_actual_inline(ar)

        # ════════════════════════════════════════════════════
        # データ点表
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_table):
            _render_sample_table_inline(ar)

        # ════════════════════════════════════════════════════
        # チューニング
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_tuning):
            from frontend_nicegui.components.tuning_tab import render_tuning_tab
            render_tuning_tab(state)

        # ════════════════════════════════════════════════════
        # 前処理後データ
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_data):
            _render_processed_data(ar)

        # ════════════════════════════════════════════════════
        # 解釈性・重要度（フル版）
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_interp):
            from frontend_nicegui.components.interpretation_panel import render_interpretation_panel
            render_interpretation_panel(ar, state)

        # ════════════════════════════════════════════════════
        # 追加可視化
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_extra):
            _render_extra_visualizations(ar, state)

        # ════════════════════════════════════════════════════
        # バッチ予測
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_batch):
            from frontend_nicegui.components.batch_predict_tab import render_batch_predict_tab
            render_batch_predict_tab(state)

        # ════════════════════════════════════════════════════
        # レポート生成
        # ════════════════════════════════════════════════════
        with ui.tab_panel(tab_report):
            from frontend_nicegui.components.report_generator import render_report_tab
            render_report_tab(state)


# ================================================================
# モデル評価
# ================================================================
def _render_model_evaluation(ar) -> None:
    """モデルスコア比較テーブルとFold別スコア"""

    # ── パイプラインフロー図 ──
    proc_X = getattr(ar, "processed_X", None)
    n_feats = proc_X.shape[1] if proc_X is not None and hasattr(proc_X, "shape") else "?"
    n_models = len(ar.model_scores) if hasattr(ar, "model_scores") else "?"

    flow_steps = [
        ("📂", "データ", f"{getattr(ar, 'n_samples', '?')}行"),
        ("⚙️", "前処理", f"{n_feats}特徴量"),
        ("🔄", f"CV({getattr(ar, 'cv_folds', '?')}fold)", f"{n_models}モデル"),
        ("🏆", ar.best_model_key, f"{ar.best_score:.4f}"),
    ]

    with ui.row().classes("items-center q-gutter-none q-mb-md full-width justify-center"):
        for i, (icon, label, detail) in enumerate(flow_steps):
            delay_class = f"delay-{(i+1)*100}"
            with ui.card().classes(f"q-pa-xs text-center animate-slide-up hover-bounce {delay_class}").style(
                "min-width: 100px; background: rgba(0,212,255,0.08); border-radius: 8px;"
                "border: 1px solid rgba(0,212,255,0.2);"
            ):
                ui.label(icon).style("font-size: 1.2rem;")
                ui.label(label).classes("text-caption text-bold").style("font-size: 0.75rem;")
                ui.label(detail).classes("text-caption text-grey").style("font-size: 0.82rem;")
            if i < len(flow_steps) - 1:
                ui.label("→").classes("text-grey-5 q-mx-xs").style("font-size: 1.2rem;")

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
                with ui.card().classes("glass-card q-pa-sm q-mb-sm hover-bounce"):
                    ui.label(f"{'🏆 ' if key == ar.best_model_key else ''}{key}").classes(
                        "text-subtitle2 text-bold" if key == ar.best_model_key else "text-subtitle2"
                    )
                    fold_text = " | ".join(
                        f"Fold{i+1}: {s:.4f}" for i, s in enumerate(fold_scores)
                    )
                    ui.label(fold_text).classes("text-caption text-grey-5")

    # ── モデル間統計検定 ──
    _render_model_significance(ar)

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
                    for i, (val, lbl) in enumerate([
                        (f"{r2:.4f}", "R² (OOF)"),
                        (f"{rmse:.4f}", "RMSE (OOF)"),
                        (f"{mae:.4f}", "MAE (OOF)"),
                    ]):
                        delay_class = f"delay-{(i+1)*100}"
                        with ui.card().classes(f"glass-card q-pa-sm animate-slide-up hover-bounce {delay_class}"):
                            ui.label(val).classes("text-h6 text-bold hero-gradient")
                            ui.label(lbl).classes("text-caption text-grey-5")
            else:
                from sklearn.metrics import accuracy_score, f1_score
                acc = accuracy_score(ar.oof_true, ar.oof_predictions)
                f1 = f1_score(ar.oof_true, ar.oof_predictions, average="weighted", zero_division=0)
                with ui.row().classes("q-gutter-md"):
                    for i, (val, lbl) in enumerate([
                        (f"{acc:.4f}", "Accuracy (OOF)"),
                        (f"{f1:.4f}", "F1-weighted (OOF)"),
                    ]):
                        delay_class = f"delay-{(i+1)*100}"
                        with ui.card().classes(f"glass-card q-pa-sm animate-slide-up hover-bounce {delay_class}"):
                            ui.label(val).classes("text-h6 text-bold hero-gradient")
                            ui.label(lbl).classes("text-caption text-grey-5")
        except Exception as ex:
            ui.label(f"OOFメトリクス計算エラー: {ex}").classes("text-caption text-red")

        # ── 残差分析プロット ──
        if ar.task == "regression":
            ui.separator()
            with ui.expansion("📉 残差分析（OOF）", icon="scatter_plot").classes("full-width q-mt-sm"):
                _render_residual_analysis(ar)

    # ── 学習曲線 ──
    ui.separator()
    with ui.expansion("📈 学習曲線 (Learning Curve)", icon="trending_up").classes("full-width q-mt-sm"):
        _render_learning_curve(ar)

    # ── 分類タスク専用: 混同行列・ROC ──
    if ar.task in ("classification", "multiclass"):
        ui.separator()
        with ui.expansion("🔢 混同行列・ROC曲線", icon="grid_on").classes("full-width q-mt-sm"):
            _render_classification_metrics(ar)


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
        for i, (val, lbl) in enumerate([
            (f"{proc_X.shape[0]:,}", "サンプル数"),
            (f"{proc_X.shape[1]:,}", "特徴量数"),
            (f"{int(proc_X.isnull().sum().sum()):,}" if hasattr(proc_X, "isnull") else "0", "欠損値"),
        ]):
            delay_class = f"delay-{(i+1)*100}"
            with ui.card().classes(f"glass-card q-pa-sm animate-slide-up hover-bounce {delay_class}"):
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
# モデル間統計検定
# ================================================================
def _render_model_significance(ar) -> None:
    """最良モデルと他モデルの対応t検定（Fold間スコア）。"""
    best_key = ar.best_model_key
    best_detail = ar.model_details.get(best_key, {})
    best_folds = best_detail.get("fold_scores", [])

    if len(best_folds) < 3 or len(ar.model_details) < 2:
        return

    ui.separator()
    with ui.expansion("📐 モデル間統計検定（対応t検定）", icon="science").classes("full-width q-mt-sm"):
        ui.label(f"基準モデル: 🏆 {best_key}").classes("text-caption text-grey q-mb-sm")

        rows = []
        for key, detail in ar.model_details.items():
            if key == best_key:
                continue
            other_folds = detail.get("fold_scores", [])
            if len(other_folds) != len(best_folds):
                continue

            try:
                from scipy.stats import ttest_rel
                t_stat, p_value = ttest_rel(best_folds, other_folds)

                # Cohen's d (paired)
                diffs = [b - o for b, o in zip(best_folds, other_folds)]
                mean_diff = sum(diffs) / len(diffs)
                std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
                cohens_d = mean_diff / std_diff if std_diff > 0 else 0

                sig = "✅ 有意差あり" if p_value < 0.05 else "⚠️ 有意差なし"
                sig_color = "text-green" if p_value < 0.05 else "text-amber"
                effect = "大" if abs(cohens_d) > 0.8 else ("中" if abs(cohens_d) > 0.5 else "小")

                rows.append({
                    "vs_model": key,
                    "p_value": p_value,
                    "t_stat": t_stat,
                    "cohens_d": cohens_d,
                    "sig": sig,
                    "sig_color": sig_color,
                    "effect": effect,
                })
            except Exception:
                continue

        if rows:
            for r in rows:
                with ui.card().classes("full-width q-pa-xs q-mb-xs glass-card"):
                    with ui.row().classes("items-center full-width justify-between"):
                        ui.label(f"🏆 {best_key} vs {r['vs_model']}").classes("text-caption text-bold")
                        ui.label(r["sig"]).classes(f"text-caption {r['sig_color']}")
                    with ui.row().classes("q-gutter-sm"):
                        for val, lbl in [
                            (f"p={r['p_value']:.4f}", "p値"),
                            (f"t={r['t_stat']:.3f}", "t統計量"),
                            (f"d={r['cohens_d']:.3f} ({r['effect']})", "Cohen's d"),
                        ]:
                            ui.label(f"{lbl}: {val}").classes("text-caption text-grey").style("font-size: 0.82rem;")
        else:
            ui.label("Fold数が一致するモデルペアがありません").classes("text-caption text-grey")


# ================================================================
# 学習曲線
# ================================================================
def _render_learning_curve(ar) -> None:
    """交差検証ベースの学習曲線（Train vs Validation スコア vs サンプル数）。"""
    model = getattr(ar, "best_pipeline", None)
    X = getattr(ar, "processed_X", None)
    y = getattr(ar, "y_train", None)

    if model is None or X is None or y is None:
        ui.label("⚠️ モデルまたはデータが取得できません").classes("text-amber text-caption")
        return

    lc_container = ui.column().classes("full-width")

    async def _calc_lc():
        lc_container.clear()
        with lc_container:
            ui.label("⏳ 学習曲線を計算中...").classes("text-grey-5")
        try:
            from sklearn.model_selection import learning_curve
            import plotly.graph_objects as go
            import numpy as np

            cv_folds = getattr(ar, "cv_folds", 5)
            scoring = "r2" if ar.task == "regression" else "accuracy"

            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=np.linspace(0.1, 1.0, 8),
                cv=cv_folds,
                scoring=scoring,
                n_jobs=1,
            )

            train_mean = train_scores.mean(axis=1)
            train_std  = train_scores.std(axis=1)
            val_mean   = val_scores.mean(axis=1)
            val_std    = val_scores.std(axis=1)

            fig = go.Figure()
            # Train帯
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
                fill="toself", fillcolor="rgba(0,212,255,0.1)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_mean, mode="lines+markers",
                line=dict(color="#00d4ff", width=2), name="Train スコア",
                marker=dict(size=6),
            ))
            # Val帯
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                fill="toself", fillcolor="rgba(74,222,128,0.1)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes, y=val_mean, mode="lines+markers",
                line=dict(color="#4ade80", width=2), name="Validation スコア",
                marker=dict(size=6),
            ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="学習サンプル数",
                yaxis_title=scoring,
                title=f"学習曲線 ({scoring})",
                legend=dict(orientation="h", y=1.05),
            )

            lc_container.clear()
            with lc_container:
                ui.plotly(fig).classes("full-width")
                gap = float(train_mean[-1] - val_mean[-1])
                gap_label = "⚠️ 過学習の可能性あり" if gap > 0.1 else "✅ 汎化性能良好"
                gap_color = "text-amber" if gap > 0.1 else "text-green"
                ui.label(f"Train-Val ギャップ: {gap:+.4f}  {gap_label}").classes(f"text-caption {gap_color}")

            ui.notify("✅ 学習曲線計算完了", type="positive")

        except Exception as ex:
            lc_container.clear()
            with lc_container:
                ui.label(f"学習曲線計算エラー: {ex}").classes("text-red text-caption")

    ui.label("学習データ量と汎化性能の関係を可視化します（計算に数秒かかります）。").classes("text-caption text-grey-5 q-mb-sm")
    ui.button("📈 学習曲線を計算", on_click=_calc_lc).props("outline color=cyan size=sm no-caps")
    lc_container


# ================================================================
# 分類タスク専用: 混同行列・ROC曲線
# ================================================================
def _render_classification_metrics(ar) -> None:
    """OOFの混同行列・ROC-AUC・Classification Report を描画する。"""
    y_true = getattr(ar, "oof_true", None)
    y_pred = getattr(ar, "oof_predictions", None)

    if y_true is None or y_pred is None:
        ui.label("⚠️ OOFデータが利用できません").classes("text-amber text-caption")
        return

    try:
        import numpy as np
        import plotly.figure_factory as ff
        import plotly.graph_objects as go
        from sklearn.metrics import (
            confusion_matrix, classification_report,
            roc_auc_score, roc_curve,
        )

        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        classes = sorted(set(y_t.tolist()))

        # ── 混同行列 ──
        cm = confusion_matrix(y_t, y_p, labels=classes)
        fig_cm = ff.create_annotated_heatmap(
            z=cm.tolist(),
            x=[str(c) for c in classes],
            y=[str(c) for c in classes],
            colorscale="Blues",
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=10, r=10, t=50, b=10),
            title="混同行列 (OOF)",
            xaxis_title="予測ラベル",
            yaxis_title="正解ラベル",
        )
        ui.label("🔢 混同行列").classes("text-subtitle2 q-mt-md q-mb-sm")
        ui.plotly(fig_cm).classes("full-width")

        # ── Classification Report テーブル ──
        try:
            report_str = classification_report(y_t, y_p, output_dict=True, zero_division=0)
            rows = []
            for key, val in report_str.items():
                if isinstance(val, dict):
                    rows.append({
                        "クラス": key,
                        "Precision": f"{val.get('precision', 0):.4f}",
                        "Recall": f"{val.get('recall', 0):.4f}",
                        "F1-score": f"{val.get('f1-score', 0):.4f}",
                        "Support": str(int(val.get("support", 0))),
                    })
            cols = [{"name": c, "label": c, "field": c, "align": "center"} for c in rows[0].keys()]
            cols[0]["align"] = "left"
            ui.label("📋 分類レポート").classes("text-subtitle2 q-mt-md q-mb-sm")
            ui.table(columns=cols, rows=rows).classes("full-width").props("dense flat bordered")
        except Exception:
            pass

        # ── 2クラス限定: ROC曲線 ──
        if len(classes) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_t, y_p, pos_label=classes[1])
                auc_val = roc_auc_score(y_t, y_p)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="rgba(255,255,255,0.2)", dash="dash"),
                    name="Random",
                ))
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    line=dict(color="#00d4ff", width=2),
                    name=f"ROC (AUC={auc_val:.4f})",
                    fill="tozeroy",
                    fillcolor="rgba(0,212,255,0.06)",
                ))
                fig_roc.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    title=f"ROC曲線 (AUC = {auc_val:.4f})",
                )
                ui.label("📈 ROC曲線").classes("text-subtitle2 q-mt-md q-mb-sm")
                ui.plotly(fig_roc).classes("full-width")
            except Exception:
                pass

    except ImportError as ie:
        ui.label(f"⚠️ {ie}").classes("text-amber text-caption")
    except Exception as ex:
        ui.label(f"分類指標計算エラー: {ex}").classes("text-red text-caption")


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
