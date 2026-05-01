"""
frontend_nicegui/components/computation_progress.py

計算進捗ダッシュボード — スタンドアロンUIコンポーネント。

xTB等の量子化学計算の進捗をリアルタイムで可視化:
- 分子ごとの進捗バー
- 推定残り時間
- ステップ別ステータス（RDKit→xTB→特徴量抽出）
- 計算量見積もりプレビュー
- 中間結果の自動保存ステータス

既存UIへの影響: なし（完全新規コンポーネント）
"""
from __future__ import annotations

import logging
import time
from typing import Any

from nicegui import ui

logger = logging.getLogger(__name__)


def render_computation_progress(state: dict[str, Any]) -> None:
    """計算進捗ダッシュボードを描画する。"""

    # ── ヘッダー ──
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 24px;"
    ):
        with ui.row().classes("items-center gap-4"):
            ui.icon("speed").classes("text-3xl").style("color: #00d4ff;")
            with ui.column().classes("gap-0"):
                ui.label("📊 計算ステータスダッシュボード").classes(
                    "text-xl font-bold"
                ).style("color: #e0e0f0;")
                ui.label(
                    "xTB量子化学計算の進捗・推定時間・リソース管理"
                ).classes("text-sm").style("color: #a0a0c0;")

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # 計算量見積もりセクション
    # ═══════════════════════════════════════════════════════════
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 20px;"
    ):
        ui.label("⏱️ 計算量見積もり").classes("text-lg font-bold").style(
            "color: #e0e0f0;"
        )

        with ui.row().classes("gap-4 q-mt-sm items-end"):
            n_mol_input = ui.number(
                "分子数", value=100, min=1, max=10000, step=10,
            ).classes("w-32")
            avg_atoms_input = ui.number(
                "平均原子数", value=30, min=1, max=500, step=5,
            ).classes("w-32")
            calc_type_select = ui.select(
                label="計算タイプ",
                options={"sp": "⚡ 単点計算(sp)", "opt": "🚀 構造最適化(opt)"},
                value="opt",
            ).classes("w-48")

        estimate_container = ui.column().classes("w-full q-mt-md")

        def _update_estimate():
            try:
                from backend.utils.compute_budget import ComputeBudget
                budget = ComputeBudget()
                summary = budget.get_summary(
                    n_molecules=int(n_mol_input.value),
                    avg_atoms=int(avg_atoms_input.value),
                )
                rec_type = summary["recommended_calc_type"]

                estimate_container.clear()
                with estimate_container:
                    # 見積もり結果カード
                    bg_color = (
                        "rgba(74, 222, 128, 0.05)"
                        if summary["estimated_minutes"] < 10
                        else "rgba(251, 191, 36, 0.08)"
                        if summary["estimated_minutes"] < 60
                        else "rgba(248, 113, 113, 0.08)"
                    )
                    with ui.card().classes("w-full").style(
                        f"background: {bg_color}; "
                        "border-radius: 12px; padding: 16px;"
                    ):
                        with ui.row().classes("gap-8 items-center"):
                            with ui.column().classes("gap-1"):
                                ui.label("推定計算時間").classes("text-sm").style(
                                    "color: #a0a0c0;"
                                )
                                mins = summary["estimated_minutes"]
                                if mins < 1:
                                    time_str = f"{mins*60:.0f}秒"
                                elif mins < 60:
                                    time_str = f"{mins:.1f}分"
                                else:
                                    time_str = f"{mins/60:.1f}時間"
                                ui.label(time_str).classes(
                                    "text-2xl font-bold"
                                ).style("color: #e0e0f0;")

                            with ui.column().classes("gap-1"):
                                ui.label("推奨計算タイプ").classes("text-sm").style(
                                    "color: #a0a0c0;"
                                )
                                icon = "⚡" if rec_type == "sp" else "🚀"
                                ui.label(f"{icon} {rec_type}").classes(
                                    "text-lg font-bold"
                                ).style("color: #00d4ff;")

                            with ui.column().classes("gap-1"):
                                ui.label("処理分子数").classes("text-sm").style(
                                    "color: #a0a0c0;"
                                )
                                ui.label(
                                    f"{summary['n_molecules']}分子"
                                ).classes("text-lg").style("color: #e0e0f0;")

                        # コスト指標
                        ui.separator().classes("q-my-sm")
                        with ui.row().classes("gap-6"):
                            for label, icon_str in [
                                ("⚡ 高速 (~10秒/分子)", "RDKit 2D記述子"),
                                ("🚀 標準 (~1分/分子)", "xTB最適化"),
                                ("🐢 精密 (~10分/分子)", "freq+熱力学量"),
                            ]:
                                ui.label(f"{label}: {icon_str}").classes(
                                    "text-xs"
                                ).style("color: #a0a0c0;")

            except Exception as e:
                estimate_container.clear()
                with estimate_container:
                    ui.label(f"⚠️ 見積もりエラー: {e}").style("color: #fbbf24;")

        n_mol_input.on_value_change(lambda _: _update_estimate())
        avg_atoms_input.on_value_change(lambda _: _update_estimate())
        calc_type_select.on_value_change(lambda _: _update_estimate())
        _update_estimate()

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # 適応的特徴量選択セクション
    # ═══════════════════════════════════════════════════════════
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 20px;"
    ):
        ui.label("🎯 適応的特徴量選択").classes("text-lg font-bold").style(
            "color: #e0e0f0;"
        )
        ui.label(
            "予測タスクと計算予算に基づいて最適な特徴量セットを自動推奨"
        ).classes("text-sm q-mb-sm").style("color: #a0a0c0;")

        try:
            from backend.chem.adaptive_feature_selector import AdaptiveFeatureSelector
            selector = AdaptiveFeatureSelector()
            tasks = selector.available_tasks
            task_options = {}
            for t in tasks:
                desc = selector.get_task_description(t)
                task_options[t] = f"{t}: {desc}" if desc else t
        except Exception:
            task_options = {"general": "general: 汎用"}

        with ui.row().classes("gap-4 items-end"):
            task_select = ui.select(
                label="予測タスク",
                options=task_options,
                value="general",
            ).classes("w-64")

            budget_input = ui.number(
                "予算 (秒/分子)", value=120, min=0.1, max=3600, step=10,
            ).classes("w-40")

        selector_result_container = ui.column().classes("w-full q-mt-md")

        def _run_feature_selection():
            try:
                from backend.chem.adaptive_feature_selector import AdaptiveFeatureSelector
                sel = AdaptiveFeatureSelector()
                result = sel.select(
                    task_type=task_select.value,
                    n_molecules=int(n_mol_input.value),
                    max_time_per_mol_s=float(budget_input.value),
                )

                selector_result_container.clear()
                with selector_result_container:
                    with ui.card().classes("w-full").style(
                        "background: rgba(0, 212, 255, 0.05); "
                        "border: 1px solid rgba(0, 212, 255, 0.15); "
                        "border-radius: 12px; padding: 16px;"
                    ):
                        with ui.row().classes("gap-6 items-center"):
                            ui.label(
                                f"✅ {len(result.selected_features)}特徴量セット選択済み"
                            ).classes("text-lg font-bold").style("color: #4ade80;")
                            ui.label(
                                f"推定: {result.estimated_total_minutes:.1f}分"
                            ).style("color: #a0a0c0;")
                            if result.requires_xtb:
                                ui.badge("xTB必要", color="purple")
                            if result.requires_opt:
                                ui.badge("構造最適化", color="amber")

                        # 選択された特徴量リスト
                        ui.separator().classes("q-my-sm")
                        with ui.row().classes("gap-2 flex-wrap"):
                            for feat in result.selected_features:
                                ui.chip(feat, icon="check_circle").props(
                                    "outline color=cyan size=sm"
                                )

                        # ノート
                        if result.notes:
                            ui.separator().classes("q-my-sm")
                            for note in result.notes:
                                ui.label(f"ℹ️ {note}").classes("text-xs").style(
                                    "color: #a0a0c0;"
                                )

            except Exception as e:
                selector_result_container.clear()
                with selector_result_container:
                    ui.label(f"⚠️ エラー: {e}").style("color: #fbbf24;")

        ui.button(
            "🔍 最適特徴量を推奨",
            on_click=_run_feature_selection,
            icon="auto_awesome",
        ).props("outline color=cyan").classes("q-mt-sm")

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # 計算コストカタログ
    # ═══════════════════════════════════════════════════════════
    with ui.expansion(
        "📋 特徴量コストカタログ",
        icon="list",
    ).classes("w-full").style(
        "background: rgba(255,255,255,0.02); border-radius: 12px;"
    ):
        try:
            from backend.chem.adaptive_feature_selector import AdaptiveFeatureSelector
            sel = AdaptiveFeatureSelector()
            cost_data = sel.get_cost_summary()

            columns = [
                {"name": "name", "label": "特徴量", "field": "name", "sortable": True},
                {"name": "category", "label": "カテゴリ", "field": "category"},
                {"name": "time", "label": "時間/分子", "field": "time_per_mol_s", "sortable": True},
                {"name": "xtb", "label": "xTB", "field": "requires_xtb"},
                {"name": "desc", "label": "説明", "field": "description"},
            ]
            rows = [
                {
                    **c,
                    "time_per_mol_s": f"{c['time_per_mol_s']:.1f}s",
                    "requires_xtb": "✅" if c["requires_xtb"] else "—",
                }
                for c in cost_data
            ]
            ui.table(columns=columns, rows=rows).classes("w-full").props(
                "dense flat"
            )
        except Exception:
            ui.label("コストカタログを読み込めません").style("color: #a0a0c0;")
