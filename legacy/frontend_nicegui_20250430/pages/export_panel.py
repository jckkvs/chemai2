"""
frontend_nicegui/pages/export_panel.py

解析レポートのエクスポートUIパネル。
PDF / Word / Jupyter Notebook / ZIP の4形式に対応し、
バックエンドの backend.export モジュールを直接呼び出す。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import ui, run

logger = logging.getLogger(__name__)


def render_export_panel(state: dict[str, Any]) -> None:
    """エクスポートパネルを描画する。"""

    with ui.column().classes("full-width q-pa-md q-gutter-md"):

        # ── ヘッダー ──
        with ui.card().classes("full-width q-pa-md").style(
            "background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,47,247,0.08));"
            "border: 1px solid rgba(0,212,255,0.2); border-radius: 12px;"
        ):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.html('<span style="font-size:28px;">📤</span>')
                ui.label("解析レポート エクスポート").style(
                    "font-size: 20px; font-weight: 800; "
                    "background: linear-gradient(90deg, #00d4ff, #7b2ff7); "
                    "-webkit-background-clip: text; -webkit-text-fill-color: transparent;"
                )
            ui.label(
                "解析が完了した結果をPDF、Word、Jupyter Notebook、またはチャートZIPとしてダウンロードします。"
            ).classes("text-caption text-grey-5 q-mt-xs")

        # ── 解析結果なしのガード ──
        ar = state.get("automl_result")
        if ar is None:
            with ui.card().classes("full-width q-pa-lg text-center").style(
                "border: 1px dashed rgba(255,255,255,0.15); border-radius: 10px;"
            ):
                ui.html('<span style="font-size:48px; opacity:0.3;">📊</span>')
                ui.label("解析結果がまだありません").classes("text-h6 text-grey-5 q-mt-sm")
                ui.label(
                    "「📂 データ設定」タブでデータを読み込み、解析開始ボタンを押してください。"
                ).classes("text-caption text-grey-6")
            return

        # ── 解析サマリー表示 ──
        with ui.card().classes("full-width q-pa-md glass-card"):
            with ui.row().classes("q-gutter-md items-center"):
                ui.html('<span style="font-size:20px;">🏆</span>')
                ui.label(f"最良モデル: {ar.best_model_key}").classes("text-subtitle1 text-bold")
                ui.badge(f"スコア: {ar.best_score:.4f}", color="cyan").props("dense")

            metrics: dict = {}
            if ar.oof_true is not None and ar.oof_predictions is not None:
                import numpy as np
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                y_t = np.asarray(ar.oof_true).ravel()
                y_p = np.asarray(ar.oof_predictions).ravel()
                if ar.task == "regression":
                    metrics = {
                        "R²":   round(float(r2_score(y_t, y_p)), 4),
                        "RMSE": round(float(mean_squared_error(y_t, y_p, squared=False)), 4),
                        "MAE":  round(float(mean_absolute_error(y_t, y_p)), 4),
                    }

            if metrics:
                with ui.row().classes("q-gutter-sm q-mt-xs"):
                    for k, v in metrics.items():
                        with ui.card().classes("q-pa-xs").style(
                            "background: rgba(0,212,255,0.06); border-radius:6px; min-width:80px;"
                        ):
                            ui.label(str(v)).classes("text-body2 text-bold text-cyan")
                            ui.label(k).classes("text-caption text-grey-5")

        # ── フォーマット選択 ──
        with ui.card().classes("full-width q-pa-md glass-card"):
            ui.label("📋 出力形式を選択").classes("text-subtitle2 q-mb-sm")

            format_val = {"v": "pdf"}

            with ui.row().classes("q-gutter-sm"):
                for fmt, icon, label, desc in [
                    ("pdf",   "📄", "PDF",              "ReportLab製の高品質PDFレポート"),
                    ("docx",  "📝", "Word (.docx)",     "編集可能なWordドキュメント"),
                    ("ipynb", "📓", "Jupyter Notebook", "実行可能な解析ノートブック"),
                    ("zip",   "🗜️", "チャートZIP",       "全チャート画像を一括ダウンロード"),
                    ("py",    "✅", "再現スクリプト",       "モデルを利用したスタンドアロン予測スクリプト"),
                ]:
                    is_sel = format_val["v"] == fmt

                    def _select(f=fmt):
                        format_val["v"] = f
                        _rebuild()

                    ui.button(
                        f"{icon} {label}",
                        on_click=_select,
                    ).style(
                        f"border: {'2px' if is_sel else '1px'} solid "
                        f"{'#00d4ff' if is_sel else 'rgba(255,255,255,0.15)'};"
                        f"background: {'rgba(0,212,255,0.1)' if is_sel else 'transparent'};"
                        f"color: {'#00d4ff' if is_sel else '#9ca3af'};"
                        "border-radius: 8px; padding: 8px 16px; font-size:13px; cursor:pointer;"
                    ).props("flat no-caps").tooltip(desc)

            ui.label(f"選択中: {format_val['v'].upper()}").classes("text-caption text-cyan q-mt-xs")

        # ── 詳細設定 ──
        with ui.expansion("⚙️ 詳細設定", icon="settings").classes("full-width glass-card"):
            include_importance = ui.checkbox("特徴量重要度を含める", value=True)
            include_charts     = ui.checkbox("解析チャート画像を含める", value=True)
            include_data_head  = ui.checkbox("データサンプル（先頭5行）を含める", value=False)
            exp_name_input     = ui.input(
                label="ファイル名（拡張子不要）",
                value=f"chemai_report_{datetime.now().strftime('%Y%m%d_%H%M')}",
                placeholder="report_filename",
            ).props("outlined dense").classes("full-width q-mt-sm")
            output_dir_input   = ui.input(
                label="保存先フォルダ",
                value="exports",
                placeholder="exports",
            ).props("outlined dense").classes("full-width q-mt-sm")

        # ── エクスポートボタン ──
        status_container = ui.column().classes("full-width")

        async def _do_export():
            fmt = format_val["v"]
            filename = exp_name_input.value.strip() or "chemai_report"
            output_dir = output_dir_input.value.strip() or "exports"

            status_container.clear()
            with status_container:
                prog = ui.linear_progress(value=0, show_value=False).props("color=cyan rounded")
                lbl  = ui.label(f"⏳ {fmt.upper()} を生成中...").classes("text-grey-5 text-caption")

            # 結果辞書の組み立て
            importances: dict = {}
            try:
                if include_importance.value:
                    estimator = ar.best_pipeline
                    if hasattr(ar.best_pipeline, "steps"):
                        estimator = ar.best_pipeline.steps[-1][1]
                        if hasattr(estimator, "steps"):
                            estimator = estimator.steps[-1][1]
                    if hasattr(estimator, "feature_importances_"):
                        import numpy as np
                        proc_X = getattr(ar, "processed_X", None)
                        names = (
                            list(proc_X.columns) if proc_X is not None and hasattr(proc_X, "columns")
                            else [f"f{i}" for i in range(len(estimator.feature_importances_))]
                        )
                        importances = dict(zip(
                            names[:len(estimator.feature_importances_)],
                            estimator.feature_importances_.tolist(),
                        ))
            except Exception:
                pass

            result_dict: dict[str, Any] = {
                "best_model_name": ar.best_model_key,
                "metrics": metrics,
                "feature_importances": importances if include_importance.value else {},
                "chart_paths": state.get("_chart_paths", []) if include_charts.value else [],
                "ai_commentary": state.get("_ai_commentary", ""),
                # Notebook 用追加情報
                "target_col":   state.get("target_col", "target"),
                "feature_cols": (
                    list(getattr(ar, "processed_X", None).columns)
                    if getattr(ar, "processed_X", None) is not None
                    and hasattr(getattr(ar, "processed_X", None), "columns")
                    else []
                ),
                "best_params":  (
                    ar.model_details.get(ar.best_model_key, {}).get("params", {})
                    if hasattr(ar, "model_details") else {}
                ),
                "cv_folds": getattr(ar, "cv_folds", 5),
            }

            prog.value = 0.3
            lbl.text = f"⏳ {fmt.upper()} を書き込み中..."
            def _run_export():
                from backend.export import PDFExporter, WordExporter, NotebookExporter, ChartBundleExporter
                if fmt == "pdf":
                    return PDFExporter(output_dir).export(result_dict, filename)
                elif fmt == "docx":
                    return WordExporter(output_dir).export(result_dict, filename)
                elif fmt == "ipynb":
                    return NotebookExporter(output_dir).export(result_dict, filename)
                elif fmt == "zip":
                    return ChartBundleExporter(output_dir).export(result_dict, filename)
                elif fmt == "py":
                    import os
                    from backend.export.reproducibility import generate_reproduction_script
                    import joblib
                    
                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    base_name = filename
                    if base_name.endswith(".py"):
                        base_name = base_name[:-3]
                        
                    py_path = out_dir / f"{base_name}.py"
                    model_path = out_dir / f"{base_name}_model.pkl"
                    
                    # 1. Dump best model pipeline
                    joblib.dump(ar.best_pipeline, model_path)
                    
                    # 2. Get columns
                    t_col = state.get("target_col", "target")
                    s_col = state.get("smiles_cols", [])
                    s_col_str = s_col[0] if isinstance(s_col, list) and len(s_col) > 0 else (s_col if isinstance(s_col, str) else None)
                    if s_col_str is None:
                        s_col_str = state.get("smiles_col")
                        if isinstance(s_col_str, list) and len(s_col_str) > 0:
                            s_col_str = s_col_str[0].get("smiles_col")

                    # 3. Generate reproduction script
                    script_content = generate_reproduction_script(
                        model_path=model_path.name,
                        data_path="your_data.csv",
                        target_col=t_col,
                        task=ar.task,
                        smiles_col=s_col_str,
                    )
                    
                    with open(py_path, "w", encoding="utf-8") as f:
                        f.write(script_content)
                    
                    return py_path
                else:
                    raise ValueError(f"未対応のフォーマット: {fmt}")

            try:
                out_path: Path = await run.io_bound(_run_export)
                prog.value = 1.0
                lbl.text = f"✅ {out_path.name} を生成しました"
                ui.notify(f"✅ エクスポート完了: {out_path.name}", type="positive", timeout=5000)

                # ダウンロードボタン
                status_container.clear()
                with status_container:
                    with ui.row().classes("items-center q-gutter-sm"):
                        ui.icon("check_circle", color="green")
                        ui.label(f"✅ {out_path.name}").classes("text-green text-bold")
                    ui.label(f"保存先: {out_path}").classes("text-caption text-grey-5")
                    ui.button(
                        f"📥 {out_path.name} をダウンロード",
                        on_click=lambda: ui.download(str(out_path)),
                    ).props("outline color=cyan no-caps").classes("q-mt-sm")

            except Exception as ex:
                prog.value = 0
                lbl.text = f"❌ エクスポートエラー: {ex}"
                ui.notify(f"❌ エクスポートエラー: {str(ex)[:200]}", type="negative", timeout=8000)
                logger.exception("エクスポートエラー")

        ui.button(
            "🚀 エクスポート実行",
            on_click=_do_export,
        ).style(
            "background: linear-gradient(135deg, #00d4ff, #7b2ff7);"
            "color: white; border-radius: 10px; font-weight: 800;"
            "font-size: 15px; padding: 12px 32px; width: 100%;"
            "box-shadow: 0 4px 20px rgba(0,212,255,0.3);"
        ).props("no-caps")

        status_container


    def _rebuild():
        """フォーマット切替時に全体を再描画する（現状はnotify のみ）。"""
        pass
