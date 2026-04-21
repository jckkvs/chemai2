"""
frontend_nicegui/components/constraint_panel.py

単調性制約、線形性、グループ化などの詳細な制約設定を管理する統合パネル。
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from nicegui import ui

from frontend_nicegui.components.column_meta_editor import render_column_meta_editor, extract_monotonic_from_column_meta

logger = logging.getLogger(__name__)

def render_constraint_panel(state: dict[str, Any]) -> None:
    """制約設定パネルを描画する。"""
    
    if state.get("df") is None:
        ui.label("⚠️ まずデータを読み込んでください").classes("text-amber q-pa-md")
        return

    df = state["df"]
    
    with ui.column().classes("full-width gap-4"):
        # ── 1. サマリーカード ──
        with ui.row().classes("full-width q-gutter-md"):
            mono_dict = extract_monotonic_from_column_meta(state)
            mono_count = len(mono_dict)
            
            with ui.card().classes("glass-card q-pa-md flex-1"):
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.icon("show_chart", color="indigo", size="sm")
                    ui.label("設定済みの単調性制約").classes("text-caption text-grey-5")
                ui.label(str(mono_count)).classes("text-h4 text-bold text-indigo")
                
            with ui.card().classes("glass-card q-pa-md flex-1"):
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.icon("auto_awesome", color="amber", size="sm")
                    ui.label("推奨される制約").classes("text-caption text-grey-5")
                ui.label("可").classes("text-h4 text-bold text-amber") # Auto-Suggest ready

        # ── 2. ヘルプと操作 ──
        with ui.card().classes("full-width bg-indigo-900/10 border-indigo-500/20 p-4"):
            with ui.row().classes("items-center q-gutter-md"):
                ui.icon("info", color="indigo-4")
                with ui.column():
                    ui.label("単調性制約とは？").classes("text-body2 font-bold text-indigo-2")
                    ui.label("「分子量が増えるほど溶解度は減る」といった事象の向きをモデルに強制します。不適切な傾向（オーバーフィッティング）を防ぎ、物理的に妥当な予測を可能にします。").classes("text-caption text-grey-4")
            
            with ui.row().classes("q-mt-md q-gutter-sm"):
                def _auto_suggest_all():
                    # smiles_feature_panel の _apply_suggestions と同等のロジック
                    from frontend_nicegui.components.column_meta_editor import _set_meta
                    suggestions = {
                        "MolWt": -1, "MW": -1, "MolecularWeight": -1,
                        "MolLogP": -1, "LogP": -1,
                        "TPSA": 1,
                        "NumHDonors": 1, "HBD": 1,
                        "NumHAcceptors": 1, "HBA": 1
                    }
                    applied_count = 0
                    all_cols = list(df.columns)
                    if state.get("precalc_df") is not None:
                        all_cols = list(state["precalc_df"].columns)
                    
                    for col in all_cols:
                        for s_key, s_val in suggestions.items():
                            if s_key.lower() in col.lower():
                                _set_meta(state, col, "monotonic", s_val)
                                applied_count += 1
                                break
                    ui.notify(f"✨ {applied_count} 個の制約を自動提案しました。下のリストで確認・調整してください。", type="positive")
                    if state.get("_refresh_tabs"): state["_refresh_tabs"]()

                ui.button("✨ 化学的知見から制約を自動提案", on_click=_auto_suggest_all).props("unelevated no-caps color=indigo")
                ui.button("🗑 制約をリセット", on_click=lambda: (state.update({"column_meta": {}}), ui.notify("リセットしました"), state["_refresh_tabs"]())).props("outline no-caps color=grey")

        # ── 3. 詳細エディタ ──
        ui.separator().classes("q-my-md")
        # 元のエディタを呼び出す
        # state["precalc_df"] がある場合はそちらを優先して表示対象にする
        display_df = state.get("precalc_df") if state.get("precalc_df") is not None else df
        render_column_meta_editor(state, display_df)
