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
        # ── 1. 化学的ドメイン知見の自動適用 (Premium Section) ──
        with ui.card().classes("full-width bg-purple-900/10 border-purple-500/30 p-4 q-mb-md"):
            with ui.row().classes("items-center q-gutter-sm mb-2"):
                ui.icon("psychology", color="purple-3", size="sm")
                ui.label("🧬 化学的ドメイン知見の自動適用").classes("text-lg font-bold text-white")
                ui.switch(value=state.get("auto_domain_knowledge", True), 
                          on_change=lambda e: state.update({"auto_domain_knowledge": e.value})).props("dense color=purple")
            
            ui.label("生成された記述子（MW, LogP等）に対し、物理化学的妥当性を考慮した単調性制約を自動的に提案・適用します。").classes("text-sm text-grey-4")
            
            if state.get("auto_domain_knowledge", True):
                with ui.row().classes("q-gutter-sm q-mt-sm flex-wrap"):
                    # 推奨される主要因を表示（チップ形式）
                    ui.chip("MW ↘", color="purple-9", text_color="white").props("outline").tooltip("分子量↑ → 水溶性↓")
                    ui.chip("LogP ↘", color="purple-9", text_color="white").props("outline").tooltip("脂溶性↑ → 水溶性↓")
                    ui.chip("TPSA ↗", color="purple-9", text_color="white").props("outline").tooltip("極性面積↑ → 水溶性↑")
                    ui.chip("HBD ↗", color="purple-9", text_color="white").props("outline").tooltip("水素結合供与体↑ → 水溶性↑")

                def _apply_suggestions():
                    from frontend_nicegui.components.column_meta_editor import _set_meta
                    # 基になる推奨ロジック
                    CHEMICAL_RATIONALES = {
                        "MolWt": -1, "MW": -1, "MolecularWeight": -1,
                        "MolLogP": -1, "LogP": -1,
                        "TPSA": 1,
                        "NumHDonors": 1, "HBD": 1,
                        "NumHAcceptors": 1, "HBA": 1,
                        "RingCount": -1, "NumRings": -1
                    }
                    applied_count = 0
                    all_cols = list(df.columns)
                    if state.get("precalc_df") is not None:
                        all_cols = list(state["precalc_df"].columns)
                    
                    for col in all_cols:
                        for s_key, s_val in CHEMICAL_RATIONALES.items():
                            if s_key.lower() in col.lower():
                                _set_meta(state, col, "monotonic", s_val)
                                applied_count += 1
                                break
                    ui.notify(f"✨ {applied_count} 個の記述子に推奨制約を適用しました。", type="positive", color="purple")
                    if state.get("_refresh_tabs"): state["_refresh_tabs"]()

                ui.button("✨ 推奨制約を今すぐ一括適用", on_click=_apply_suggestions).props("unelevated no-caps size=sm color=purple").classes("q-mt-sm")

        # ── 2. サマリーと操作 ──
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
                    ui.icon("info", color="grey", size="sm")
                    ui.label("全特徴量数").classes("text-caption text-grey-5")
                    n_feats = len(state["precalc_df"].columns) if state.get("precalc_df") is not None else len(df.columns)
                ui.label(str(n_feats)).classes("text-h4 text-bold text-grey-5")

        with ui.row().classes("q-mt-md q-gutter-sm"):
            ui.button("🗑 制約をリセット", on_click=lambda: (state.update({"column_meta": {}}), ui.notify("リセットしました"), state["_refresh_tabs"]())).props("outline no-caps color=grey")

        # ── 3. 詳細エディタ ──
        ui.separator().classes("q-my-md")
        # 元のエディタを呼び出す
        # state["precalc_df"] がある場合はそちらを優先して表示対象にする
        display_df = state.get("precalc_df") if state.get("precalc_df") is not None else df
        render_column_meta_editor(state, display_df)
