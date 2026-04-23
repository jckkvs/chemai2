"""
frontend_nicegui/components/results_tab.py

結果確認タブ：モデル比較・Fold別スコア・前処理後データ・SHAP解釈性
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import asyncio
import logging
logger = logging.getLogger(__name__)

from nicegui import ui, app

from frontend_nicegui.components.results_tab_extras import (
    _render_model_overview,
    _render_per_model_tabs,
    _render_pred_actual_inline,
    _render_sample_table_inline,
    _render_extra_visualizations,
)
from frontend_nicegui.components.feature_comparison_dashboard import render_feature_comparison_dashboard
from frontend_nicegui.utils.plot_utils import render_plot_with_expand

# ── 【必須】ObservableDict 対応ヘルパー ──
def _safe_get(obj, key: str, default=None):
    """
    ObservableDict / dict / 通常オブジェクトのいずれからも安全に値を取得
    """
    if obj is None:
        return default
    # 1. 通常の属性アクセスを試す
    if hasattr(obj, key):
        val = getattr(obj, key)
        # callable（関数）は除外
        return None if callable(val) else val
    # 2. dict 風アクセスを試す
    if isinstance(obj, dict):
        return obj.get(key, default)
    # 3. __getitem__ 対応（ObservableDict 等）
    try:
        return obj[key]
    except (TypeError, KeyError, AttributeError):
        return default
# ────────────────────────────────


def render_results_tab(state: dict[str, Any]) -> None:
    """結果確認タブ全体を描画する。（専門家の検証フロー4タブ構成）"""

    # [追加] stateに結果がない場合、storageから復元
    if not state.get("automl_result") and not state.get("automl_results"):
        saved_result = app.storage.user.get('automl_result')
        if saved_result:
            state["automl_result"] = saved_result
            state["automl_results"] = {"デフォルト": saved_result}
            logger.info("✓ app.storage.user['automl_result'] から結果を復元しました")

    all_results = state.get("automl_results", {})
    single_ar   = state.get("automl_result")

    # [追加] 結果を再読込するボタン (指示に基づく)
    with ui.row().classes('w-full justify-end q-mb-md'):
        def _on_reload():
            # [指示に基づく修正] 統一されたキー automl_result を最優先でチェック
            results = (app.storage.user.get('automl_result') or 
                       app.storage.user.get('analysis_results') or 
                       app.storage.user.get('current_results'))
            
            logger.info(f"Retrieved results: {bool(results)}")

            if results:
                ui.notify(f"✅ 結果を復元しました。表示を更新します。", type='positive')
                # 状態のリフレッシュをトリガー
                refresh_fn = state.get("_refresh_results")
                if refresh_fn:
                    refresh_fn()
            else:
                ui.notify('⚠️ 保存された解析結果が見つかりません。', type='warning')

        ui.button('🔄 結果を再読込', on_click=_on_reload).props('outline dense icon=refresh').classes('glass-card')

    # [追加] 3秒ごとに結果をチェックするタイマー (指示に基づく)
    async def _check_for_results_poll():
        """ストレージに結果がないか定期的にチェック"""
        results = app.storage.user.get('automl_result')
        # 既に画面に結果が出ている場合（single_arが存在）は通知を抑制
        if results and not single_ar and not hasattr(_check_for_results_poll, '_displayed'):
            _check_for_results_poll._displayed = True
            ui.notify('✅ 解析結果を検出しました。', type='positive')
            # 自動更新をトリガー
            refresh_fn = state.get("_refresh_results")
            if refresh_fn:
                refresh_fn()
    
    ui.timer(3.0, _check_for_results_poll)

    # 結果が全くない場合
    if not all_results and single_ar is None:
        with ui.card().classes(
            "glass-card q-pa-xl full-width animate-slide-up items-center justify-center text-center"
        ).props('data-testid="no-results-card"'):
            ui.icon("analytics", color="grey-7", size="xl").classes("q-mb-md").props('aria-hidden="true"')
            ui.label("解析結果がまだありません").classes("text-h6 text-grey-5").props(
                'role="heading" aria-level="2"'
            )
            ui.label(
                "「📂 データ設定」タブでデータを読み込み、画面上部の「🚀 解析開始」ボタンを押してください。"
            ).classes("text-grey-6 q-mt-sm")
        return

    # 成功したセットのみ抽出
    success_results = {k: v for k, v in all_results.items() if v is not None}
    if not success_results and single_ar:
        success_results = {"デフォルト": single_ar}

    set_names = list(success_results.keys())
    if not set_names:
        return

    # 最良セットを自動選択（専門家は「最良モデル詳細」を最初に見る）
    best_set_name = state.get("best_set_name", set_names[0])
    if best_set_name not in success_results:
        best_set_name = set_names[0]
    best_ar = success_results[best_set_name]

    # ═══════════════════════════════════════════════════
    # 4タブ構成（専門家の検証フロー）
    # ═══════════════════════════════════════════════════
    with ui.tabs().classes("full-width").props(
        "dense no-caps active-color=cyan indicator-color=cyan scrollable"
    ) as main_tabs:
        tab_insight   = ui.tab("insight",  label="🏆 最良モデル詳細",   icon="star")
        tab_compare   = ui.tab("compare",  label="📊 全体比較",          icon="compare_arrows")
        tab_explorer  = ui.tab("explorer", label="🧪 モデル詳細検証",    icon="science")
        tab_data      = ui.tab("dataview", label="📥 データプレビュー",  icon="table_chart")

    with ui.tab_panels(main_tabs, value=tab_insight).classes("full-width bg-transparent"):

        # ════════ ① 最良モデル詳細 ════════
        with ui.tab_panel(tab_insight):
            _render_best_insight_tab(best_ar, state, best_set_name)

        # ════════ ② 全体比較 ════════
        with ui.tab_panel(tab_compare):
            _render_comparison_tab(success_results, state)

        # ════════ ③ モデル詳細検証 ════════
        with ui.tab_panel(tab_explorer):
            _render_model_explorer_tab(success_results, state)

        # ════════ ④ データプレビュー ════════
        with ui.tab_panel(tab_data):
            _render_data_preview_tab(state)


# ================================================================
# ヘルパー: SMILES → Base64 画像
# ================================================================
def _smiles_to_b64(smiles: str, size: tuple[int, int] = (200, 200)) -> str:
    """SMILESをRDKitで描画してBase64文字列を返す。RDKit未導入の場合は空文字。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import io, base64
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _detect_smiles_col(df: pd.DataFrame) -> str | None:
    """DataFrameの列名からSMILES列を自動検出（大文字小文字不問）。"""
    for col in df.columns:
        if "smiles" in col.lower():
            return col
    return None


# ================================================================
# ① 最良モデル詳細タブ
# ================================================================
def _render_best_insight_tab(ar, state: dict, set_name: str) -> None:
    """正方形プロット（ホバー→構造サイドパネル）＋指標カード＋Feature Importance。"""
    import plotly.graph_objects as go

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
        ui.icon("emoji_events", color="amber", size="md")
        ui.label(f"最良モデル: {_safe_get(ar, 'best_model_key', '不明')}").classes("text-h5 text-bold hero-gradient")
        ui.badge(f"セット: {set_name}", color="teal").props("dense")
        ui.badge(f"CVスコア: {_safe_get(ar, 'best_score', 0):.4f}", color="cyan").props("dense")

    model    = _safe_get(ar, "best_pipeline")
    proc_X   = _safe_get(ar, "processed_X")
    
    cv_true  = _safe_get(ar, "oof_true")
    cv_pred  = _safe_get(ar, "oof_predictions")
    train_true = _safe_get(ar, "y_train")
    train_pred = _safe_get(ar, "train_predictions")
    
    # ── データ前処理サマリー (Transparency Report) ──
    preproc_report = _safe_get(ar, "preprocess_report")
    if preproc_report:
        with ui.expansion("⚙️ 前処理レポート (Transparency)", icon="auto_fix_high").classes("full-width q-mb-md glass-card"):
            ui.label("このモデルの学習時に適用された前処理ステップの記録です。").classes("text-caption text-grey-5 q-mb-sm")
            ui.html(f"<pre style='font-size: 0.85rem; color: #a1a1aa; white-space: pre-wrap;'>{preproc_report.generate_summary()}</pre>")

    # ── 指標カード行 ──
    if cv_true is not None and cv_pred is not None:
        y_cv_t = np.asarray(cv_true).ravel()
        y_cv_p = np.asarray(cv_pred).ravel()
        cv_residuals = y_cv_t - y_cv_p

        if _safe_get(ar, "task", "regression") == "regression":
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            try:
                cv_r2   = r2_score(y_cv_t, y_cv_p)
                cv_rmse = float(np.sqrt(mean_squared_error(y_cv_t, y_cv_p)))
            except Exception:
                cv_r2 = cv_rmse = float("nan")

            train_r2 = float("nan")
            y_tr_t = None
            y_tr_p = None
            if train_true is not None and train_pred is not None:
                try:
                    y_tr_t = np.asarray(train_true).ravel()
                    y_tr_p = np.asarray(train_pred).ravel()
                    # NaN除外
                    valid = ~(np.isnan(y_tr_t) | np.isnan(y_tr_p))
                    if valid.sum() > 5:
                        y_tr_t = y_tr_t[valid]
                        y_tr_p = y_tr_p[valid]
                        train_r2 = r2_score(y_tr_t, y_tr_p)
                except Exception as _train_err:
                    import logging
                    logging.getLogger(__name__).debug(f"Train metrics計算失敗: {_train_err}")
            
            # --- 🔍 モデル診断パネル (Train と CV の R² および 過学習度) ---
            ui.label("🔍 モデル適合度診断").classes("text-h6 q-mt-md")
            with ui.row().classes("full-width q-gutter-md q-mb-md"):
                # 全データ R²
                with ui.card().classes("col q-pa-sm").style("background:rgba(40,0,60,0.2); border:1px solid rgba(123,47,247,0.2);"):
                    ui.label("全データ R²").classes("text-caption text-grey-4")
                    ui.label(f"{train_r2:.4f}" if not np.isnan(train_r2) else "N/A").classes("text-h5 text-purple")
                    ui.label("（モデルの表現力）").classes("text-xs text-grey-6")
                
                # CV R²
                with ui.card().classes("col q-pa-sm").style("background:rgba(0,40,40,0.2); border:1px solid rgba(0,212,255,0.2);"):
                    ui.label("CV R²（平均）").classes("text-caption text-grey-4")
                    ui.label(f"{cv_r2:.4f}").classes("text-h5 text-cyan")
                    ui.label("（汎化性能の見積もり）").classes("text-xs text-grey-6")

                # 過学習度（差）
                if not np.isnan(train_r2) and not np.isnan(cv_r2):
                    gap = train_r2 - cv_r2
                    gap_color = "green" if gap < 0.05 else ("amber" if gap < 0.15 else "red")
                    with ui.card().classes("col q-pa-sm").style(f"background:rgba(0,0,0,0.2); border:1px solid var(--q-{gap_color});"):
                        ui.label("過学習度（差）").classes("text-caption text-grey-4")
                        ui.label(f"{gap:+.4f}").classes(f"text-h5 text-{gap_color}")
                        ui.label("（+は過学習傾向）").classes("text-xs text-grey-6")
            
            # 自動通知
            if not np.isnan(train_r2) and not np.isnan(cv_r2) and gap > 0.15:
                 ui.notify(f"⚠️ 過学習の疑い: Train-CV差={gap:.4f}", type="warning", position="bottom-right")

            # --- 多階層メトリック評価システム ---
            ui.label("📊 多階層評価指標 (Stratified Metrics)").classes("text-h6 q-mt-md")
            
            metrics = None
            if state and "stratified_metrics" in state:
                sm = state["stratified_metrics"]
                if hasattr(sm, "to_dict"):
                    metrics = sm.to_dict()
                elif isinstance(sm, dict):
                    metrics = sm
            elif _safe_get(ar, "stratified_metrics") is not None:
                sm = _safe_get(ar, "stratified_metrics")
                if hasattr(sm, "to_dict"):
                    metrics = sm.to_dict()
            
            if metrics is None:
                ui.label("ℹ️ 解析を実行すると階層別評価指標が表示されます").classes("text-grey-6")
            else:
                from frontend_nicegui.components.metric_breakdown_panel import render_metric_breakdown_panel
                render_metric_breakdown_panel(metrics)

            # --- プロット生成用ヘルパー ---
            def _create_scatter(y_t_arr, y_p_arr, title, is_train=False):
                # 非有限値・NaNの除去（描画クラッシュ防止）
                mask = np.isfinite(y_t_arr) & np.isfinite(y_p_arr)
                yt, yp = y_t_arr[mask], y_p_arr[mask]
                if len(yt) < 2:
                    return None

                fig = go.Figure()
                color_scale = "Purples" if is_train else "RdBu_r"
                marker_color = yt - yp if not is_train else "#a78bfa"
                
                fig.add_trace(go.Scatter(
                    x=yt, y=yp, mode="markers",
                    marker=dict(size=6, color=marker_color, opacity=0.7, colorscale=color_scale if not is_train else None),
                    name="データ点",
                    hovertemplate="実測: %{x:.3f}<br>予測: %{y:.3f}<extra></extra>"
                ))

                # アスペクト比固定（range手動設定は競合するため削除）
                fig.update_yaxes(scaleanchor="x", scaleratio=1, gridcolor="rgba(255,255,255,0.08)")
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")

                # y=x 対角線
                mn, mx = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
                fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                              line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=1.5))

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.15)",
                    margin=dict(l=40, r=20, t=40, b=40),
                    xaxis_title="実測値", yaxis_title="予測値",
                    title=dict(text=title, font=dict(size=14)),
                    hovermode="closest"
                )
                return fig


            # --- プロット表示 (タブ切り替え) ---
            ui.label("📊 予測実測プロット").classes("text-subtitle2 q-mt-md")
            ui.markdown("*学習に使用した全データに対する傾向と、未知データ(CV)に対する傾向を比較します。学習データで点が直線に近く、CVでばらつく場合は過学習です。*").classes("text-caption text-grey-5 q-mb-sm")
            
            with ui.tabs().classes("w-full") as plot_tabs:
                ui.tab("CV（検証）プロット")
                ui.tab("Train（学習）プロット")
                
            with ui.tab_panels(plot_tabs, value="CV（検証）プロット").classes("w-full bg-transparent p-0"):
                with ui.tab_panel("CV（検証）プロット"):
                    ui.label("検証データでの汎化性能").classes("text-caption text-grey-5 mb-2")
                    fig_cv = _create_scatter(y_cv_t, y_cv_p, f"検証データ (CV) [n={len(y_cv_t)}]", is_train=False)
                    render_plot_with_expand(fig_cv, title="CV Plot", height="400px")
                    
                with ui.tab_panel("Train（学習）プロット"):
                    if y_tr_t is not None and y_tr_p is not None:
                        ui.label("学習データへの適合度（過学習チェック用）").classes("text-caption text-grey-5 mb-2")
                        fig_train = _create_scatter(y_tr_t, y_tr_p, f"学習データ (Train) [n={len(y_tr_t)}]", is_train=True)
                        render_plot_with_expand(fig_train, title="Train Plot", height="400px")
                    else:
                        ui.label("Trainデータプロットが利用できません。").classes("text-amber")


            # 既存の残差分析
            with ui.expansion("📉 残差分析 (CV)", icon="scatter_plot").classes("full-width q-mt-sm"):
                _render_residual_analysis(ar)

        else:
            # 分類タスク
            from sklearn.metrics import accuracy_score, f1_score
            try:
                acc = accuracy_score(y_t, y_p)
                f1  = f1_score(y_t, y_p, average="weighted", zero_division=0)
            except Exception:
                acc = f1 = float("nan")
            with ui.row().classes("q-gutter-sm q-mb-md"):
                for val, lbl, col in [
                    (f"{_safe_get(ar, 'best_score', 0):.4f}", _safe_get(ar, 'scoring', 'score'),     "cyan"),
                    (f"{acc:.4f}",           "Accuracy(OOF)","teal"),
                    (f"{f1:.4f}",            "F1-weighted",  "amber"),
                ]:
                    with ui.card().classes("q-pa-xs").style(
                        "min-width:90px; background:rgba(0,0,0,0.2); border-radius:8px;"
                        "border:1px solid rgba(0,212,255,0.15);"
                    ):
                        ui.label(val).classes(f"text-subtitle1 text-bold text-{col}")
                        ui.label(lbl).classes("text-caption text-grey-5")
            with ui.expansion("🔢 混同行列・ROC", icon="grid_on").classes("full-width q-mt-sm"):
                _render_classification_metrics(ar)

    # ── Feature Importance ──
    ui.separator().classes("q-my-md")
    ui.label("📊 Feature Importance").classes("text-subtitle1 text-bold q-mb-xs")
    if model is not None:
        import plotly.graph_objects as go
        try:
            est = model
            if hasattr(model, "steps"):
                est = model.steps[-1][1]
                if hasattr(est, "steps"):
                    est = est.steps[-1][1]
            feat_names = list(proc_X.columns) if proc_X is not None and hasattr(proc_X, "columns") else []
            if hasattr(est, "feature_importances_"):
                imp   = est.feature_importances_
                names = feat_names[:len(imp)] if len(feat_names) >= len(imp) else [f"f{i}" for i in range(len(imp))]
                idx   = np.argsort(imp)[::-1]
                top   = min(20, len(idx))
                fig_fi = go.Figure(go.Bar(
                    x=imp[idx[:top]][::-1], y=[names[i] for i in idx[:top]][::-1],
                    orientation="h",
                    marker=dict(color=imp[idx[:top]][::-1], colorscale="Teal"),
                ))
                fig_fi.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=max(300, 22 * top), margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="重要度", title=f"Feature Importance ({_safe_get(ar, 'best_model_key', '不明')})",
                )
                ui.plotly(fig_fi).classes("full-width")
            elif hasattr(est, "coef_"):
                coefs  = est.coef_.ravel()
                names  = feat_names[:len(coefs)] if len(feat_names) >= len(coefs) else [f"f{i}" for i in range(len(coefs))]
                idx    = np.argsort(np.abs(coefs))[::-1]
                top    = min(20, len(idx))
                colors = ["rgba(74,222,128,0.75)" if coefs[i] >= 0 else "rgba(248,113,113,0.75)" for i in idx[:top]]
                fig_c  = go.Figure(go.Bar(
                    x=coefs[idx[:top]][::-1], y=[names[i] for i in idx[:top]][::-1],
                    orientation="h", marker_color=colors[::-1],
                ))
                fig_c.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=max(300, 22 * top), margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="回帰係数", title=f"回帰係数 ({_safe_get(ar, 'best_model_key', '不明')})",
                )
                ui.plotly(fig_c).classes("full-width")

                # 回帰係数詳細テーブルの表示（標準化前後）
                try:
                    from backend.models.linear_utils import extract_regression_coefficients
                    # スケーラーを探す（パイプラインの場合）
                    scaler = None
                    if hasattr(model, "steps"):
                        for step_name, step_obj in model.steps:
                            if "scaler" in step_name.lower() or "scale" in step_name.lower() or "standard" in step_name.lower():
                                scaler = step_obj
                                break
                    
                    df_coef = extract_regression_coefficients(
                        est, 
                        feature_names=feat_names[:len(coefs)] if len(feat_names) >= len(coefs) else [f"f{i}" for i in range(len(coefs))],
                        X_original=proc_X,  # 厳密には標準化前ではないがインターフェースとして
                        X_scaled=proc_X,
                        scaler=scaler
                    )

                    with ui.expansion("📐 回帰係数詳細", icon="format_list_numbered").classes("full-width q-mt-sm").props('default-opened'):
                        # 状態管理
                        show_scaled_state = {"value": True}

                        # 切り替えボタン
                        with ui.row().classes("q-mb-sm"):
                            btn_scaled = ui.button("標準化後で表示", color="primary")
                            btn_original = ui.button("標準化前で表示")
                            ui.label("※標準化後：特徴量間の相対的重要度比較用 / 標準化前：実スケールでの解釈用").classes("text-caption text-grey-5")

                        @ui.refreshable
                        def coef_table_view():
                            show_scaled = show_scaled_state["value"]
                            cols = [
                                {"name": "rank", "label": "順位", "field": "rank", "align": "center"},
                                {"name": "feature", "label": "特徴量", "field": "feature", "align": "left"},
                                {"name": "coef", "label": "係数", "field": "coef_scaled" if show_scaled else "coef_original",
                                 "format": lambda v: f"{v:+.4f}" if pd.notna(v) else "-", "align": "right"},
                                {"name": "abs_coef", "label": "寄与度", "field": "abs_coef_scaled",
                                 "format": lambda v: f"{v:.4f}" if pd.notna(v) else "-", "align": "right"},
                            ]
                            if not show_scaled:
                                cols.append({"name": "coef_scaled_ref", "label": "参考 (標準化後)",
                                           "field": "coef_scaled", "format": lambda v: f"{v:+.4f}", "align": "right"})

                            df_view = df_coef.copy()
                            df_view["rank"] = range(1, len(df_view)+1)
                            rows_data = df_view.to_dict('records')
                            ui.table(rows=rows_data, columns=cols, row_key="feature").classes("full-width").props("dense flat bordered dark")

                        def update_table(show_scaled: bool):
                            show_scaled_state["value"] = show_scaled
                            coef_table_view.refresh()
                            btn_scaled.props(f'color={"primary" if show_scaled else ""}')
                            btn_original.props(f'color={"primary" if not show_scaled else ""}')

                        btn_scaled.on("click", lambda: update_table(True))
                        btn_original.on("click", lambda: update_table(False))

                        # 初期表示
                        coef_table_view()
                except Exception as ex:
                    import traceback
                    ui.label(f"回帰係数詳細表示エラー：{ex}").classes("text-red text-caption")
                    ui.label(f"詳細：{traceback.format_exc()}").classes("text-red text-caption")
            else:
                ui.label("ℹ️ SHAP解析は「🧪 モデル詳細検証」タブ → 解釈性・重要度 で確認できます").classes("text-grey-5")
        except Exception as ex:
            ui.label(f"Feature Importance取得エラー: {ex}").classes("text-red text-caption")
    else:
        ui.label("⚠️ モデルが取得できません").classes("text-amber")


# ================================================================
# ② 全体比較タブ
# ================================================================
def _render_comparison_tab(success_results: dict, state: dict) -> None:
    """ヒートマップ＋統合ランキング＋既存機能へのショートカット。"""
    set_names = list(success_results.keys())

    # ── 全セット横断比較（2セット以上）──
    if len(set_names) >= 2:
        _render_cross_set_comparison(success_results, state)
    else:
        # 1セットの場合はシンプルなスコア比較テーブル
        ar = success_results[set_names[0]]
        scores = _safe_get(ar, "model_scores", {})
        if scores:
            ui.label("📋 推定器スコア比較").classes("text-h6 q-mb-sm")
            rows = [
                {"順位": r, "モデル": mk, "CVスコア": f"{ms:.4f}", "最良": "🏆" if r == 1 else ""}
                for r, (mk, ms) in enumerate(
                    sorted(scores.items(), key=lambda x: x[1], reverse=True), 1
                )
            ]
            cols = [{"name": k, "label": k, "field": k, "sortable": True} for k in ["順位", "モデル", "CVスコア", "最良"]]
            ui.table(columns=cols, rows=rows).classes("full-width").props("dense flat bordered")

    ui.separator().classes("q-my-md")

    # ── 各セットのサブタブ詳細（既存機能へのアクセス） ──
    ui.label("📂 セット別詳細・既存機能").classes("text-h6 q-mb-sm")
    with ui.tabs().classes("full-width").props(
        "dense no-caps active-color=cyan indicator-color=cyan scrollable"
    ) as set_tabs:
        for sn in set_names:
            sr = success_results[sn]
            is_best = sn == state.get("best_set_name", set_names[0])
            ui.tab(f"cmp_{sn}", label=f"{'🏆 ' if is_best else ''}{sn}")

    first_key = f"cmp_{set_names[0]}"
    with ui.tab_panels(set_tabs, value=first_key).classes("full-width bg-transparent"):
        for sn in set_names:
            with ui.tab_panel(f"cmp_{sn}"):
                _render_single_result(success_results[sn], state)


# ================================================================
# ③ モデル詳細検証タブ
# ================================================================
def _render_model_explorer_tab(success_results: dict, state: dict) -> None:
    """セレクトボックスでモデルを選択し、最良モデル詳細と同じビューを表示。"""
    # 全セット × 全モデルの組み合わせリストを構築
    options: dict[str, tuple] = {}
    for sn, ar in success_results.items():
        scores = _safe_get(ar, "model_scores", {})
        for mk in sorted(scores.keys()):
            label = f"{sn} / {mk}"
            options[label] = (sn, mk, ar)

    if not options:
        ui.label("解析結果がありません").classes("text-grey")
        return

    ui.label("🔽 確認したいモデルを選択してください").classes("text-subtitle2 q-mb-sm")
    sel = ui.select(list(options.keys()), value=list(options.keys())[0], label="セット / モデル").props(
        "outlined dense"
    ).classes("full-width q-mb-md")

    view_container = ui.column().classes("full-width")

    def _draw():
        view_container.clear()
        key = sel.value
        if key not in options:
            return
        sn, mk, ar_full = options[key]

        # 選択モデルのスコアを表示
        scores = ar_full.model_scores if hasattr(ar_full, "model_scores") else {}
        score  = scores.get(mk, float("nan"))

        with view_container:
            with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
                ui.badge(f"モデル: {mk}", color="teal").props("dense")
                ui.badge(f"CVスコア: {score:.4f}", color="cyan").props("dense")
                ui.badge(f"セット: {sn}", color="grey-7").props("dense")

            # 最良モデルと同じビューを再利用（arはセット全体のオブジェクト）
            # OOFを含む詳細はar全体に紐付いているため、best_model_key を一時差し替え
            class _ArProxy:
                """best_model_keyのみ差し替えたプロキシ。"""
                pass

            proxy = _ArProxy()
            if isinstance(ar_full, dict):
                proxy.__dict__.update(ar_full)
            else:
                proxy.__dict__.update(ar_full.__dict__)
            proxy.best_model_key = mk

            _render_best_insight_tab(proxy, state, sn)

    sel.on_value_change(lambda: _draw())
    _draw()


# ================================================================
# ④ データプレビュータブ
# ================================================================
def _render_data_preview_tab(state: dict) -> None:
    """読み込みデータの確認。SMILES列が存在すれば化学構造画像を表示。"""
    df = state.get("df")
    if df is None:
        ui.label("⚠️ データが読み込まれていません").classes("text-amber")
        return

    smiles_col = _detect_smiles_col(df)
    has_rdkit  = False
    try:
        from rdkit import Chem  # noqa: F401
        has_rdkit = True
    except ImportError:
        pass

    with ui.row().classes("q-gutter-sm q-mb-md"):
        ui.badge(f"{df.shape[0]}行 × {df.shape[1]}列", color="teal").props("dense")
        if smiles_col:
            ui.badge(f"SMILES列: {smiles_col}", color="cyan").props("dense")
        if not has_rdkit:
            ui.badge("RDKit未導入 — 構造画像なし", color="amber").props("dense")

    # 表示行数制御
    max_rows = 50
    preview  = df.head(max_rows)

    if smiles_col and has_rdkit:
        # 構造画像付きテーブル（HTMLレンダリング）
        ui.label(f"📋 データプレビュー（先頭{max_rows}行、構造画像付き）").classes("text-subtitle2 q-mb-xs")
        ui.label("※ 構造画像はクライアント側で描画されるため、行数が多いと表示に時間がかかります").classes(
            "text-caption text-grey-5 q-mb-sm"
        )

        display_cols = [smiles_col] + [c for c in preview.columns if c != smiles_col][:8]
        columns_def  = [
            {"name": "mol_img", "label": "構造", "field": "mol_img"},
        ] + [
            {"name": c, "label": c, "field": c, "sortable": True}
            for c in display_cols
        ]

        rows_data = []
        for _, row in preview.iterrows():
            smi    = str(row.get(smiles_col, ""))
            b64    = _smiles_to_b64(smi, size=(80, 80))
            img_html = (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:80px;height:80px;border-radius:4px;" />'
                if b64 else "—"
            )
            r_dict = {"mol_img": img_html}
            for c in display_cols:
                v = row.get(c, "")
                r_dict[c] = f"{v:.4g}" if isinstance(v, float) else str(v)
            rows_data.append(r_dict)

        tbl = ui.table(columns=columns_def, rows=rows_data, row_key=smiles_col).classes(
            "full-width"
        ).props("dense flat bordered")
        tbl.add_slot(
            "body-cell-mol_img",
            '<td class="q-td"><span v-html="props.value"></span></td>',
        )

    else:
        # 通常テーブル（SMILESなし or RDKit未導入）
        ui.label(f"📋 データプレビュー（先頭{max_rows}行）").classes("text-subtitle2 q-mb-sm")
        display_cols = list(preview.columns[:20])
        columns_def  = [
            {"name": c, "label": c, "field": c, "sortable": True, "align": "left"}
            for c in display_cols
        ]
        rows_data = []
        for _, row in preview.iterrows():
            r_dict = {}
            for c in display_cols:
                v = row[c]
                r_dict[c] = "—" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))
            rows_data.append(r_dict)
        ui.table(columns=columns_def, rows=rows_data).classes("full-width").props("dense flat bordered")
        if df.shape[1] > 20:
            ui.label(f"... 他 {df.shape[1] - 20} 列").classes("text-caption text-grey-6")

    # 基本統計量
    ui.separator().classes("q-my-md")
    with ui.expansion("📐 基本統計量", icon="calculate").classes("full-width"):
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            desc = num_df.describe().T.round(4).reset_index().rename(columns={"index": "列名"})
            dcols = [{"name": c, "label": c, "field": c, "sortable": True} for c in desc.columns]
            ui.table(
                columns=dcols, rows=desc.to_dict("records"),
                pagination={"rowsPerPage": 20},
            ).classes("full-width").props("dense flat bordered")

    # ── Raw vs Encoded 比較 ──
    all_results = state.get("automl_results", {})
    single_ar = state.get("automl_result")
    best_ar = None
    for v in all_results.values():
        if v is not None:
            best_ar = v
            break
    if best_ar is None:
        best_ar = single_ar

    if best_ar is not None:
        proc_X = getattr(best_ar, "processed_X", None)
        if proc_X is not None and hasattr(proc_X, "shape"):
            ui.separator().classes("q-my-md")
            with ui.expansion("🔄 Raw → Encoded データ比較", icon="compare_arrows").classes("full-width"):
                raw_cols = df.shape[1]
                enc_cols = proc_X.shape[1]
                raw_dtypes = df.dtypes.value_counts()
                enc_dtypes = proc_X.dtypes.value_counts() if hasattr(proc_X, "dtypes") else {}

                with ui.row().classes("q-gutter-md q-mb-md"):
                    with ui.card().classes("q-pa-xs").style(
                        "min-width:100px; background:rgba(0,0,0,0.2); border-radius:8px;"
                        "border:1px solid rgba(0,212,255,0.15);"
                    ):
                        ui.label(f"{raw_cols}").classes("text-subtitle1 text-bold text-grey-4")
                        ui.label("元データ列数").classes("text-caption text-grey-5")
                    with ui.card().classes("q-pa-xs").style(
                        "min-width:100px; background:rgba(0,0,0,0.2); border-radius:8px;"
                        "border:1px solid rgba(0,212,255,0.15);"
                    ):
                        ui.label("→").classes("text-subtitle1 text-bold text-cyan")
                    with ui.card().classes("q-pa-xs").style(
                        "min-width:100px; background:rgba(0,0,0,0.2); border-radius:8px;"
                        "border:1px solid rgba(74,222,128,0.3);"
                    ):
                        ui.label(f"{enc_cols}").classes("text-subtitle1 text-bold text-green")
                        ui.label("エンコード後列数").classes("text-caption text-grey-5")
                    # 差分
                    diff = enc_cols - raw_cols
                    diff_text = f"+{diff}" if diff > 0 else str(diff)
                    diff_color = "green" if diff >= 0 else "amber"
                    with ui.card().classes("q-pa-xs").style(
                        "min-width:100px; background:rgba(0,0,0,0.2); border-radius:8px;"
                    ):
                        ui.label(diff_text).classes(f"text-subtitle1 text-bold text-{diff_color}")
                        ui.label("列数変化").classes("text-caption text-grey-5")

                # 型変化サマリー
                ui.label("データ型の変化:").classes("text-caption text-grey-5 q-mt-xs")
                with ui.row().classes("q-gutter-sm"):
                    for dtype, count in raw_dtypes.items():
                        ui.badge(f"Raw: {dtype} ({count}列)", color="grey").props("dense")
                    for dtype, count in (enc_dtypes.items() if hasattr(enc_dtypes, "items") else []):
                        ui.badge(f"Encoded: {dtype} ({count}列)", color="teal").props("dense")

                # エンコード後プレビュー
                ui.separator().classes("q-my-sm")
                ui.label("📊 エンコード後データ（先頭20行）").classes("text-caption text-grey-4")
                enc_preview = proc_X.head(20)
                enc_display = list(enc_preview.columns[:15])
                enc_cols_def = [
                    {"name": c, "label": c, "field": c, "sortable": True, "align": "left"}
                    for c in enc_display
                ]
                enc_rows = []
                for _, row in enc_preview.iterrows():
                    r = {}
                    for c in enc_display:
                        v = row[c]
                        r[c] = "—" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))
                    enc_rows.append(r)
                ui.table(columns=enc_cols_def, rows=enc_rows).classes("full-width").props("dense flat bordered")
                if proc_X.shape[1] > 15:
                    ui.label(f"... 他 {proc_X.shape[1] - 15} 列").classes("text-caption text-grey-6")


# ================================================================
# 旧 _render_single_result（② 全体比較タブ → セット別詳細 から呼ばれる）
# ================================================================
def _render_single_result(ar, state: dict) -> None:
    """単一セットの結果詳細を描画する（既存機能へのアクセスを保持）。"""
    scores = _safe_get(ar, "model_scores", {})

    # ── 警告 ──
    warnings = _safe_get(ar, "warnings")
    if warnings:
        with ui.expansion(f"⚠️ 警告 ({len(warnings)}件)", icon="warning").classes(
            "full-width q-mb-md animate-shake"
        ):
            for w in warnings:
                ui.label(f"⚠️ {w}").classes("text-amber text-caption")

    # ── サブタブ（既存8タブを残す）──
    with ui.tabs().classes("full-width").props(
        "dense active-color=cyan indicator-color=cyan scrollable"
    ) as res_tabs:
        tab_best     = ui.tab("best",     label="🏆 ベスト推定器",   icon="star")
        tab_overview = ui.tab("overview", label="📊 全モデル概要",   icon="leaderboard")
        tab_compare  = ui.tab("compare",  label="🔄 推定器比較",    icon="compare_arrows")
        tab_permodel = ui.tab("permodel", label="📈 モデル別詳細",   icon="analytics")
        tab_interp   = ui.tab("interp",   label="🔬 解釈性・重要度", icon="psychology")
        tab_extra    = ui.tab("extra",    label="🎨 追加可視化",     icon="bar_chart")
        tab_batch    = ui.tab("batch",    label="🔮 バッチ予測",     icon="batch_prediction")
        tab_report   = ui.tab("report",   label="📝 レポート",       icon="summarize")

    with ui.tab_panels(res_tabs, value=tab_best).classes("full-width"):
        with ui.tab_panel(tab_best):
            _render_best_estimator_tab(ar, state)

        # ════════ 全モデル概要 ════════
        with ui.tab_panel(tab_overview):
            _render_model_overview(ar)

        # ════════ 推定器比較（新設）════════
        with ui.tab_panel(tab_compare):
            _render_model_comparison_tab(ar, state)

        # ════════ モデル別詳細 ════════
        with ui.tab_panel(tab_permodel):
            _render_per_model_tabs(ar)

        # ════════ 解釈性・重要度（SHAP/SAGE/SRI）════════
        with ui.tab_panel(tab_interp):
            from frontend_nicegui.components.interpretation_panel import render_interpretation_panel
            render_interpretation_panel(ar, state)

        # ════════ 追加可視化 ════════
        with ui.tab_panel(tab_extra):
            _render_extra_visualizations(ar, state)

        # ════════ バッチ予測 ════════
        with ui.tab_panel(tab_batch):
            from frontend_nicegui.components.batch_predict_tab import render_batch_predict_tab
            render_batch_predict_tab(state)

        # ════════ レポート ════════
        with ui.tab_panel(tab_report):
            from frontend_nicegui.components.report_generator import render_report_tab
            render_report_tab(state)


# ================================================================
# 全セット×全推定器 クロス比較
# ================================================================
def _render_cross_set_comparison(success_results: dict, state: dict) -> None:
    """全セット×全推定器のスコアをヒートマップ・ランキングで横断比較する。"""
    import plotly.graph_objects as go

    # データ収集: {(set_name, model_key): score}
    all_combos = []
    all_model_keys = set()
    for sn, ar in success_results.items():
        scores = _safe_get(ar, "model_scores", {})
        for mk, ms in scores.items():
            all_combos.append({"セット": sn, "モデル": mk, "スコア": ms})
            all_model_keys.add(mk)

    if not all_combos:
        return

    set_names = list(success_results.keys())
    model_keys = sorted(all_model_keys)

    with ui.expansion("🏆 全セット×全推定器 クロス比較", icon="compare", value=True).classes(
        "full-width q-mb-md"
    ).style("border: 1px solid rgba(0,212,255,0.2); border-radius: 12px;"):

        # ── 1. 最優秀組み合わせカード ──
        best = max(all_combos, key=lambda x: x["スコア"])
        with ui.card().classes("q-pa-sm q-mb-md").style(
            "background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(123,47,247,0.12));"
            "border: 1px solid rgba(0,212,255,0.3); border-radius: 10px;"
        ):
            with ui.row().classes("items-center q-gutter-md"):
                ui.icon("emoji_events", color="amber", size="md")
                ui.label(f"最優秀: {best['セット']} × {best['モデル']}").classes("text-subtitle1 text-bold hero-gradient")
                ui.badge(f"{best['スコア']:.4f}", color="cyan").props("dense")

        # ── 2. ヒートマップ ──
        z_matrix = []
        for sn in set_names:
            ar = success_results[sn]
            scores = _safe_get(ar, "model_scores", {})
            row = [scores.get(mk, float("nan")) for mk in model_keys]
            z_matrix.append(row)

        # 外れ値を除外したカラーバー範囲を計算
        flat_z = np.array(z_matrix).flatten()
        valid_z = flat_z[~np.isnan(flat_z) & (flat_z > -100)]
        if len(valid_z) > 0:
            zmin_val = float(np.percentile(valid_z, 5))
            zmax_val = float(np.percentile(valid_z, 95))
            zmid_val = float(np.median(valid_z))
        else:
            zmin_val, zmax_val, zmid_val = None, None, None

        fig_hm = go.Figure(go.Heatmap(
            z=z_matrix,
            x=[mk[:20] for mk in model_keys],
            y=[sn[:25] for sn in set_names],
            colorscale="RdBu_r",  # coolwarm に近い青白赤の配色
            zmin=zmin_val,
            zmax=zmax_val,
            zmid=zmid_val,
            text=[[f"{v:.4f}" if not np.isnan(v) else "—" for v in row] for row in z_matrix],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hoverongaps=False,
            colorbar=dict(title="CVスコア"),
        ))
        fig_hm.update_layout(
            title="セット×推定器 スコアヒートマップ",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            autosize=True,
            margin=dict(l=100, r=50, t=50, b=100),
            xaxis={
                "tickangle": -45,
                "tickfont": {"size": 10},
                "automargin": True,
            },
            yaxis={
                "tickfont": {"size": 10},
                "automargin": True,
            },
            height=max(400, 50 * len(set_names) + 160),
        )
        
        # 凡例の調整
        fig_hm.update_layout(
            coloraxis_colorbar=dict(
                title="CV Score",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=250,
                yanchor="top", y=1,
                ticks="outside",
            )
        )

        render_plot_with_expand(fig_hm, title="セット×推定機 スコアヒートマップ", height=f"{max(400, 50 * len(set_names) + 160)}px")

        # ── 3. 特徴量重複分析 (新機能) ──
        ui.separator().classes("q-my-md")
        render_feature_comparison_dashboard(state)

        # ── 3. 統合ランキングテーブル ──
        ui.separator().classes("q-my-sm")
        ui.label("📋 統合ランキング（全組み合わせ）").classes("text-subtitle2 q-mb-xs")
        sorted_combos = sorted(all_combos, key=lambda x: x["スコア"], reverse=True)
        rows = []
        for rank, c in enumerate(sorted_combos, 1):
            is_best = (rank == 1)
            rows.append({
                "順位": rank,
                "セット": c["セット"],
                "推定器": c["モデル"],
                "CVスコア": f"{c['スコア']:.4f}",
                "最良": "🏆" if is_best else "",
            })
        cols = [
            {"name": k, "label": k, "field": k,
             "align": "left" if k in ("セット", "推定器") else "center", "sortable": True}
            for k in ["順位", "セット", "推定器", "CVスコア", "最良"]
        ]
        ui.table(columns=cols, rows=rows[:30]).classes("full-width").props("dense flat bordered")
        if len(rows) > 30:
            ui.label(f"... 上位30件を表示（全{len(rows)}件）").classes("text-caption text-grey-6")


# ================================================================
# ベスト推定器タブ（集約ビュー）
# ================================================================
def _render_best_estimator_tab(ar, state: dict) -> None:
    """ベストモデルのサマリー + OOF + Feature Importance + 残差 + 前処理後データを集約表示。"""
    import plotly.graph_objects as go

    model = _safe_get(ar, "best_pipeline")
    proc_X = _safe_get(ar, "processed_X")
    y_true = _safe_get(ar, "oof_true")
    y_pred = _safe_get(ar, "oof_predictions")

    # ── OOFメトリクス ──
    if y_true is not None and y_pred is not None:
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        residuals = y_t - y_p

        if _safe_get(ar, "task", "regression") == "regression":
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            try:
                r2   = r2_score(y_t, y_p)
                rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
                mae  = float(mean_absolute_error(y_t, y_p))
            except Exception:
                r2 = rmse = mae = float("nan")

            ui.label("📐 OOF回帰メトリクス").classes("text-subtitle1 text-bold q-mb-xs")
            with ui.row().classes("q-gutter-md q-mb-md"):
                for val, lbl, color in [
                    (f"{r2:.4f}", "R² (OOF)", "cyan"),
                    (f"{rmse:.4f}", "RMSE (OOF)", "amber"),
                    (f"{mae:.4f}", "MAE (OOF)", "green"),
                ]:
                    with ui.card().classes("q-pa-sm").style(
                        "min-width:90px; background:rgba(0,0,0,0.2); border-radius:8px;"
                        "border:1px solid rgba(0,212,255,0.15);"
                    ):
                        ui.label(val).classes(f"text-h6 text-bold text-{color}")
                        ui.label(lbl).classes("text-caption text-grey-5")

            # 予測実測プロット
            rng = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rng, y=rng, mode="lines",
                line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1.5),
                name="y = x",
            ))
            fig.add_trace(go.Scatter(
                x=y_t, y=y_p, mode="markers",
                marker=dict(size=6, color=residuals, colorscale="RdBu_r",
                            showscale=True, colorbar=dict(title="残差"), opacity=0.7),
                name="データ点",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.15)", height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="実測値", yaxis_title="予測値",
                title=f"予測 vs 実測 (OOF, n={len(y_t)})",
            )
            ui.plotly(fig).classes("full-width")

            # 残差分析（インライン・コンパクト版）
            with ui.expansion("📉 残差分析", icon="scatter_plot").classes("full-width q-mt-sm"):
                _render_residual_analysis(ar)

        else:
            # 分類タスク
            from sklearn.metrics import accuracy_score, f1_score
            try:
                acc = accuracy_score(y_t, y_p)
                f1 = f1_score(y_t, y_p, average="weighted", zero_division=0)
            except Exception:
                acc = f1 = float("nan")
            ui.label("📐 OOF分類メトリクス").classes("text-subtitle1 text-bold q-mb-xs")
            with ui.row().classes("q-gutter-md q-mb-md"):
                for val, lbl in [(f"{acc:.4f}", "Accuracy"), (f"{f1:.4f}", "F1-weighted")]:
                    with ui.card().classes("q-pa-sm glass-card"):
                        ui.label(val).classes("text-h6 text-bold hero-gradient")
                        ui.label(lbl).classes("text-caption text-grey-5")
            # 混同行列
            with ui.expansion("🔢 混同行列・ROC", icon="grid_on").classes("full-width q-mt-sm"):
                _render_classification_metrics(ar)

    # ── Feature Importance (自動表示) ──
    ui.separator().classes("q-my-md")
    ui.label("📊 Feature Importance").classes("text-subtitle1 text-bold q-mb-xs")
    if model is not None:
        try:
            estimator = model
            if hasattr(model, "steps"):
                estimator = model.steps[-1][1]
                if hasattr(estimator, "steps"):
                    estimator = estimator.steps[-1][1]

            feat_names = list(proc_X.columns) if proc_X is not None and hasattr(proc_X, "columns") else []

            if hasattr(estimator, "feature_importances_"):
                imp = estimator.feature_importances_
                names = feat_names[:len(imp)] if len(feat_names) >= len(imp) else [f"f{i}" for i in range(len(imp))]
                idx = np.argsort(imp)[::-1]
                top = min(20, len(idx))
                fig_fi = go.Figure(go.Bar(
                    x=imp[idx[:top]][::-1],
                    y=[names[i] for i in idx[:top]][::-1],
                    orientation="h",
                    marker=dict(color=imp[idx[:top]][::-1], colorscale="Viridis"),
                ))
                fig_fi.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=max(300, 22 * top), margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="重要度", title=f"Feature Importance ({_safe_get(ar, 'best_model_key', '不明')})",
                )
                ui.plotly(fig_fi).classes("full-width")
            elif hasattr(estimator, "coef_"):
                coefs = estimator.coef_.ravel()
                names = feat_names[:len(coefs)] if len(feat_names) >= len(coefs) else [f"f{i}" for i in range(len(coefs))]
                idx = np.argsort(np.abs(coefs))[::-1]
                top = min(20, len(idx))
                colors = ["rgba(74,222,128,0.7)" if coefs[i] > 0 else "rgba(248,113,113,0.7)" for i in idx[:top]]
                fig_coef = go.Figure(go.Bar(
                    x=coefs[idx[:top]][::-1],
                    y=[names[i] for i in idx[:top]][::-1],
                    orientation="h", marker_color=colors[::-1],
                ))
                fig_coef.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    height=max(300, 22 * top), margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="回帰係数", title=f"回帰係数 ({_safe_get(ar, 'best_model_key', '不明')})",
                )
                ui.plotly(fig_coef).classes("full-width")
            else:
                ui.label("ℹ️ SHAP解析で特徴量重要度を確認してください → 「解釈性・重要度」タブ").classes("text-grey-5")
        except Exception as ex:
            ui.label(f"Feature Importance取得エラー: {ex}").classes("text-red text-caption")
    else:
        ui.label("⚠️ モデルが取得できません").classes("text-amber")

    # ── 前処理後データ概要 ──
    if proc_X is not None and hasattr(proc_X, "shape"):
        ui.separator().classes("q-my-md")
        with ui.expansion("🔢 前処理後データ概要", icon="table_chart").classes("full-width"):
            _render_processed_data(ar)

    # ── 学習曲線 ──
    ui.separator().classes("q-my-md")
    with ui.expansion("📈 学習曲線 (Learning Curve)", icon="trending_up").classes("full-width"):
        _render_learning_curve(ar)


# ================================================================
# 推定器比較タブ（新設）
# ================================================================
def _render_model_comparison_tab(ar, state: dict) -> None:
    """全推定器のFoldスコアを多角的に比較する。"""
    import plotly.graph_objects as go

    model_details = _safe_get(ar, "model_details", {})
    scores = _safe_get(ar, "model_scores", {})

    if not scores:
        ui.label("モデルスコアがありません").classes("text-grey")
        return

    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    scoring = _safe_get(ar, "scoring", "score")

    ui.label("🔄 推定器横断比較").classes("text-h6 text-bold q-mb-md")
    ui.label(
        "全推定器のFoldスコアを箱ひげ図・レーダー・統計検定で比較し、最適なモデルを多角的に評価します。"
    ).classes("text-caption text-grey-5 q-mb-md")

    # ── 1. ランキングテーブル ──
    rows = []
    for rank, (mk, ms) in enumerate(sorted_models, 1):
        detail = model_details.get(mk, {})
        cv_s = detail.get("cv_scores", [])
        std = float(np.std(cv_s)) if cv_s else 0.0
        fit_time = detail.get("fit_time", 0) or 0
        rows.append({
            "順位": rank, "モデル": mk, "CVスコア": f"{ms:.4f}",
            "±std": f"{std:.4f}" if std else "—",
            "学習時間": f"{fit_time:.2f}s" if fit_time else "—",
            "Folds": len(cv_s) if cv_s else "—",
            "最良": "🏆" if rank == 1 else "",
        })
    cols = [
        {"name": k, "label": k, "field": k,
         "align": "left" if k == "モデル" else "center", "sortable": True}
        for k in ["順位", "モデル", "CVスコア", "±std", "学習時間", "Folds", "最良"]
    ]
    ui.table(columns=cols, rows=rows).classes("full-width").props("dense flat bordered")

    # ── 2. Fold別ボックスプロット ──
    fold_data = [
        (mk, det.get("cv_scores", []))
        for mk, det in model_details.items()
        if det.get("cv_scores")
    ]
    if fold_data:
        ui.separator().classes("q-my-md")
        ui.label("📦 Fold別スコア分布").classes("text-subtitle1 text-bold q-mb-xs")
        fig_box = go.Figure()
        for mk, cv_scores in sorted(fold_data, key=lambda x: np.mean(x[1]), reverse=True):
            fig_box.add_trace(go.Box(
                y=cv_scores, name=mk[:22], boxmean=True,
                marker_color="#00d4ff",
            ))
        fig_box.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)", height=380,
            margin=dict(l=10, r=10, t=30, b=60),
            yaxis_title=scoring, title="Fold別CVスコア分布",
            xaxis_tickangle=-30,
        )
        ui.plotly(fig_box).classes("full-width")

    # ── 3. レーダーチャート（上位5モデル）──
    if len(sorted_models) >= 3:
        ui.separator().classes("q-my-md")
        ui.label("🕸️ 多軸レーダー比較").classes("text-subtitle1 text-bold q-mb-xs")
        try:
            n_top = min(5, len(sorted_models))
            categories = [scoring, "安定性(1-std)", "速度スコア"]
            fig_rad = go.Figure()
            for mk, ms in sorted_models[:n_top]:
                detail = model_details.get(mk, {})
                cv_s = detail.get("cv_scores", [ms])
                std = float(np.std(cv_s)) if len(cv_s) > 1 else 0.0
                stability = max(0.0, 1.0 - std * 10.0)
                fit_time = detail.get("fit_time", 1.0) or 1.0
                speed = 1.0 - min(1.0, fit_time / 30.0)
                values = [ms, stability, speed, ms]
                cats = categories + [categories[0]]
                fig_rad.add_trace(go.Scatterpolar(
                    r=values, theta=cats, fill="toself", name=mk[:18], opacity=0.6,
                ))
            fig_rad.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=380,
                margin=dict(l=40, r=40, t=60, b=40),
                polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor="rgba(0,0,0,0.1)"),
                title=f"モデル比較レーダー (上位{n_top})", legend=dict(orientation="h", y=-0.1),
            )
            ui.plotly(fig_rad).classes("full-width")
        except Exception:
            pass

    # ── 4. 統計検定 ──
    ui.separator().classes("q-my-md")
    with ui.expansion("📐 モデル間統計検定（対応t検定）", icon="science").classes("full-width"):
        _render_model_significance(ar)

    # ── 5. チューニングへのリンク ──
    ui.separator().classes("q-my-md")
    with ui.expansion("🎯 チューニング", icon="tune").classes("full-width"):
        from frontend_nicegui.components.tuning_tab import render_tuning_tab
        render_tuning_tab(state)


# ================================================================
# モデル評価
# ================================================================
def _render_model_evaluation(ar) -> None:
    """モデルスコア比較テーブルとFold別スコア"""

    # ── パイプラインフロー図 ──
    proc_X = _safe_get(ar, "processed_X")
    n_feats = proc_X.shape[1] if proc_X is not None and hasattr(proc_X, "shape") else "?"
    n_models = len(_safe_get(ar, "model_scores", {}))
    best_model_key = _safe_get(ar, "best_model_key", "N/A")
    best_score = _safe_get(ar, "best_score", 0.0)

    flow_steps = [
        ("📂", "データ", f"{_safe_get(ar, 'n_samples', '?')}行"),
        ("⚙️", "前処理", f"{n_feats}特徴量"),
        ("🔄", f"CV({_safe_get(ar, 'cv_folds', '?')}fold)", f"{n_models}モデル"),
        ("🏆", best_model_key, f"{best_score:.4f}"),
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
    ui.label(f"スコアリング: {_safe_get(ar, 'scoring', 'score')}").classes("text-caption text-grey-5 q-mb-md")

    # ── スコア比較テーブル ──
    rows = []
    model_scores = _safe_get(ar, "model_scores", {})
    model_details = _safe_get(ar, "model_details", {})
    best_model_key = _safe_get(ar, "best_model_key")
    for key, score in sorted(model_scores.items(), key=lambda x: -x[1]):
        detail = model_details.get(key, {})
        is_best = key == best_model_key
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
        model_details = _safe_get(ar, "model_details", {})
        best_model_key = _safe_get(ar, "best_model_key")
        for key, detail in model_details.items():
            fold_scores = detail.get("fold_scores", [])
            if fold_scores:
                with ui.card().classes("glass-card q-pa-sm q-mb-sm hover-bounce"):
                    ui.label(f"{'🏆 ' if key == best_model_key else ''}{key}").classes(
                        "text-subtitle2 text-bold" if key == best_model_key else "text-subtitle2"
                    )
                    fold_text = " | ".join(
                        f"Fold{i+1}: {s:.4f}" for i, s in enumerate(fold_scores)
                    )
                    ui.label(fold_text).classes("text-caption text-grey-5")

    # ── モデル間統計検定 ──
    _render_model_significance(ar)

    # ── OOF予測サマリー ──
    oof_predictions = _safe_get(ar, "oof_predictions")
    oof_true = _safe_get(ar, "oof_true")
    if oof_predictions is not None and oof_true is not None:
        ui.separator()
        ui.label("📈 Out-of-Fold予測サマリー").classes("text-subtitle2 q-mt-md")
        try:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            if _safe_get(ar, "task", "regression") == "regression":
                r2 = r2_score(oof_true, oof_predictions)
                rmse = mean_squared_error(oof_true, oof_predictions, squared=False)
                mae = mean_absolute_error(oof_true, oof_predictions)
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
                acc = accuracy_score(oof_true, oof_predictions)
                f1 = f1_score(oof_true, oof_predictions, average="weighted", zero_division=0)
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
        if _safe_get(ar, "task", "regression") == "regression":
            ui.separator()
            with ui.expansion("📉 残差分析（OOF）", icon="scatter_plot").classes("full-width q-mt-sm"):
                _render_residual_analysis(ar)

    # ── 学習曲線 ──
    ui.separator()
    with ui.expansion("📈 学習曲線 (Learning Curve)", icon="trending_up").classes("full-width q-mt-sm"):
        _render_learning_curve(ar)

    # ── 分類タスク専用: 混同行列・ROC ──
    if _safe_get(ar, "task") in ("classification", "multiclass"):
        ui.separator()
        with ui.expansion("🔢 混同行列・ROC曲線", icon="grid_on").classes("full-width q-mt-sm"):
            _render_classification_metrics(ar)


# ================================================================
# 前処理後データ
# ================================================================
def _render_processed_data(ar) -> None:
    """前処理後のデータテーブルと統計量"""

    proc_X = _safe_get(ar, "processed_X")
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

    model = _safe_get(ar, "best_pipeline")
    X = _safe_get(ar, "X_train")
    y = _safe_get(ar, "y_train")

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
                proc_X = _safe_get(ar, "processed_X", X)
                scoring = "r2" if _safe_get(ar, "task", "regression") == "regression" else "accuracy"
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

                proc_X = _safe_get(ar, "processed_X", X)
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

                proc_X = _safe_get(ar, "processed_X", X)
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
    best_key = _safe_get(ar, "best_model_key")
    model_details = _safe_get(ar, "model_details", {})
    best_detail = model_details.get(best_key, {})
    best_folds = best_detail.get("fold_scores", [])

    if len(best_folds) < 3 or len(model_details) < 2:
        return

    ui.separator()
    with ui.expansion("📐 モデル間統計検定（対応t検定）", icon="science").classes("full-width q-mt-sm"):
        ui.label(f"基準モデル: 🏆 {best_key}").classes("text-caption text-grey q-mb-sm")

        rows = []
        for key, detail in model_details.items():
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
    model = _safe_get(ar, "best_pipeline")
    X = _safe_get(ar, "processed_X")
    y = _safe_get(ar, "y_train")

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

            cv_folds = _safe_get(ar, "cv_folds", 5)
            scoring = "r2" if _safe_get(ar, "task", "regression") == "regression" else "accuracy"

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
    y_true = _safe_get(ar, "oof_true")
    y_pred = _safe_get(ar, "oof_predictions")

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
    y_true = _safe_get(ar, "oof_true")
    y_pred = _safe_get(ar, "oof_predictions")

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

# ==============================================================================
# 指示に基づく追加ヘルパー関数（既存機能を破壊せず並列実装）
# ==============================================================================

def show_no_results_placeholder(container):
    """結果がない場合のプレースホルダー表示"""
    container.clear()
    with container:
        with ui.card().classes('w-full bg-gray-50 border border-dashed border-gray-300 p-8 items-center glass-card'):
            ui.icon('bar_chart', size='3rem', color='grey-7')
            ui.label('解析結果がまだありません').classes('text-lg text-grey-600 mt-2')
            ui.label('「🚀 解析開始」ボタンを押して解析を実行してください。').classes('text-sm text-grey-500')

def display_analysis_results_summary(results, container):
    """解析結果の簡易サマリーを描画（指示書に基づくマージ版）"""
    container.clear()
    with container:
        try:
            with ui.card().classes('w-full bg-green-50 border-l-4 border-green-500 p-4 mb-4 glass-card'):
                ui.label('✅ 解析完了').classes('font-bold text-green-800 text-lg')
                if isinstance(results, dict):
                    ui.markdown(f"**モデル**: {results.get('model_name', 'N/A')}")
                    ui.markdown(f"**スコア**: {results.get('score', 'N/A')}")
                    if 'best_model' in results:
                        ui.markdown(f"**最適モデル**: {results['best_model']}")
            
            # データテーブル (指示に基づく追加)
            if 'predictions_df' in results and results['predictions_df'] is not None:
                ui.label('📊 予測結果').classes('font-bold text-lg mt-4')
                df = results['predictions_df']
                if hasattr(df, 'head'):
                    ui.table.from_pandas(df.head(10)).classes('w-full')
                    
        except Exception as e:
            logger.error(f"簡易サマリー表示エラー: {e}")
            ui.notify(f'結果表示エラー: {str(e)}', type='negative')

def manual_refresh_from_storage(state, container=None):
    """手動で結果を更新（指示に基づく）"""
    results = (app.storage.user.get('analysis_results') or 
               app.storage.user.get('current_results'))
    if results:
        ui.notify('✅ 結果を更新します', type='positive')
        refresh_fn = state.get("_refresh_results")
        if refresh_fn:
            refresh_fn()
    else:
        ui.notify('⚠️ 保存された結果が見つかりません', type='warning')


# =====================================================================
# 完全再実装版 結果表示タブ (create_results_tab)
# =====================================================================

@ui.refreshable
def create_results_tab(state: dict) -> None:
    """結果・レポートタブのメイン構築関数（state 一貫性確保版）"""
    
    with ui.column().classes('w-full items-center gap-4 p-4'):
        
        # === ヘッダー ===
        ui.label('📊 解析結果・レポート').classes('text-2xl font-bold text-blue-800')
        
        # === 結果取得ヘルパー関数 ===
        def _get_results():
            """state から結果を取得（app.storage へのフォールバック付き）"""
            # 優先: state 内の結果
            automl = state.get("automl_result")
            pipeline = state.get("pipeline_result")
            
            # --- 他タブとの互換性レイヤー: AutoMLResult オブジェクトを dict に変換 ---
            if automl and not isinstance(automl, dict):
                try:
                    automl = {
                        'model_name': getattr(automl, 'best_model_key', 'N/A'),
                        'score': getattr(automl, 'best_score', 'N/A'),
                        'best_model': getattr(automl, 'best_model_key', 'N/A'),
                        'predictions_df': getattr(automl, 'test_predictions_df', None),
                        'cv_scores': getattr(automl, 'cv_scores', [])
                    }
                except Exception as e:
                    logger.error(f"AutoMLResultからdictへの変換エラー: {e}")
            # -------------------------------------------------------------------------

            if automl or pipeline:
                return {"automl": automl, "pipeline": pipeline, "source": "state"}
            
            # フォールバック: app.storage（互換性維持）
            try:
                from nicegui import app
                storage_results = app.storage.user.get("analysis_results")
                if storage_results:
                    return {"automl": storage_results, "pipeline": None, "source": "storage"}
            except Exception:
                pass
            
            return None
        
        # === 結果表示コンテナ ===
        result_container = ui.column().classes('w-full')
        
        # === 結果描画関数 ===
        def _display_results():
            """結果をコンテナに描画"""
            results = _get_results()
            
            if not results or (not results["automl"] and not results["pipeline"]):
                # 結果がない場合の表示
                with result_container:
                    with ui.card().classes('w-full bg-gray-50 border border-dashed border-gray-300 p-8 items-center'):
                        ui.icon('bar_chart', size='3rem').classes('text-gray-400')
                        ui.label('解析結果がまだありません').classes('text-lg text-gray-600 mt-2')
                        ui.label('「🚀 解析開始」ボタンを押して解析を実行してください。').classes('text-sm text-gray-500')
                        # 手動リロードボタン
                        ui.button('🔄 結果を再確認', on_click=lambda: _refresh_and_display()).props('outline mt-2')
                return
            
            # 結果がある場合の表示
            result_container.clear()
            with result_container:
                # AutoML 結果
                if results["automl"]:
                    _render_automl_results(results["automl"])
                
                # Pipeline 結果
                if results["pipeline"]:
                    _render_pipeline_results(results["pipeline"])
                
                # 出典表示
                ui.label(f'※ 結果取得元: {results["source"]}').classes('text-xs text-gray-400 mt-2')
        
        # === 手動リフレッシュ関数 ===
        async def _refresh_and_display():
            """結果を再取得して表示を更新"""
            ui.notify('結果を更新中...', type='info')
            await asyncio.sleep(0.1)  # UI 安定待ち
            create_results_tab.refresh()
            ui.notify('✅ 結果を表示しました', type='positive')
        
        # === 自動監視タイマー（3 秒ごと）===
        async def _auto_check():
            """state 内の結果変化を監視し、自動で表示更新"""
            results = _get_results()
            if results and (results["automl"] or results["pipeline"]):
                # 結果が存在し、かつ未表示なら描画
                if not hasattr(_auto_check, '_displayed_once'):
                    _auto_check._displayed_once = True
                    _display_results()
                    ui.notify('✅ 解析結果が準備できました', type='positive')
        
        # 3 秒ごとにチェック（一度だけ結果表示）
        ui.timer(3.0, _auto_check)
        
        # === 初期描画 ===
        _display_results()
        
        # === 手動更新ボタン（常時表示）===
        with ui.row().classes('w-full justify-end mt-4'):
            ui.button('🔄 結果を更新', 
                     on_click=lambda: asyncio.create_task(_refresh_and_display()),
                     icon='refresh').props('outline')

def _render_automl_results(results: dict) -> None:
    """AutoML 結果を描画"""
    with ui.card().classes('w-full bg-green-50 border-l-4 border-green-500 p-4 mb-4'):
        ui.label('✅ AutoML 解析完了').classes('font-bold text-green-800 text-lg')
        
        if isinstance(results, dict):
            # 基本情報
            ui.markdown(f"**モデル**: {results.get('model_name', 'N/A')}")
            ui.markdown(f"**スコア**: {results.get('score', 'N/A')}")
            if 'best_model' in results:
                ui.markdown(f"**最適モデル**: {results['best_model']}")
            if 'cv_scores' in results:
                scores = results['cv_scores']
                if isinstance(scores, list) and len(scores) > 0:
                    avg = np.mean(scores)
                    std = np.std(scores)
                    ui.markdown(f"**CV 平均**: {avg:.4f} ± {std:.4f}")
            
            # 予測データテーブル
            if 'predictions_df' in results and results['predictions_df'] is not None:
                df = results['predictions_df']
                if isinstance(df, pd.DataFrame) and not df.empty:
                    ui.label('📊 予測結果（先頭 10 行）').classes('font-bold text-lg mt-4')
                    ui.table.from_pandas(df.head(10)).classes('w-full')
            
            # 可視化
            if 'plot' in results and results['plot'] is not None:
                ui.label('📈 可視化').classes('font-bold text-lg mt-4')
                try:
                    ui.plotly(results['plot']).classes('w-full h-96')
                except Exception as e:
                    logger.error(f"プロット表示エラー: {e}")
                    ui.label(f'⚠️ プロット表示エラー: {str(e)}').classes('text-amber')

def _render_pipeline_results(results: dict) -> None:
    """Pipeline 結果を描画"""
    with ui.card().classes('w-full bg-blue-50 border-l-4 border-blue-500 p-4 mb-4'):
        ui.label('✅ Pipeline 解析完了').classes('font-bold text-blue-800 text-lg')
        
        if isinstance(results, dict):
            if 'steps' in results:
                ui.label('実行ステップ:').classes('font-bold mt-2')
                for i, step in enumerate(results['steps'], 1):
                    ui.label(f"{i}. {step}").classes('text-sm')
            
            if 'metrics' in results:
                ui.label('評価指標:').classes('font-bold mt-2')
                for key, val in results['metrics'].items():
                    ui.label(f"{key}: {val:.4f}" if isinstance(val, (int, float)) else f"{key}: {val}").classes('text-sm')
