"""
frontend_nicegui/components/column_role_panel.py

列の役割（目的変数・説明変数・除外）と単調性制約をワンクリックで設定可能にする高速パネル。
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from nicegui import ui

from frontend_nicegui.components.column_meta_editor import _get_meta, _set_meta

logger = logging.getLogger(__name__)

# 単調性オプション定義
_MONO_OPTS = [
    (0, "➖ なし"),
    (1, "📈 増加"),
    (-1, "📉 減少"),
    (2, "🔄 自動")
]

# 役割オプション定義
_ROLE_OPTS = [
    ("feature", "説明変数", "primary"),
    ("target", "目的変数", "cyan"),
    ("exclude", "除外", "grey-7"),
    ("group", "グループID", "teal"),
    ("time", "時系列", "indigo"),
    ("weight", "重み", "blue-grey")
]

@ui.refreshable
def render_column_role_panel(state: dict[str, Any]) -> None:
    """列の役割と単調性制約の設定UI（ワンクリック・カード形式）。"""
    
    if state.get("df") is None:
        ui.label("⚠️ まずデータを読み込んでください").classes("text-amber q-pa-md")
        return

    df = state["df"]
    all_cols = list(df.columns)
    
    # --- 1. ヘッダー & 統計 ---
    with ui.row().classes("full-width items-center justify-between q-mb-md"):
        with ui.column():
            ui.label("🏷️ 列の役割と単調性設定").classes("text-h6 text-bold text-cyan")
            ui.label("役割や単調性をワンクリックで直接選択できます。").classes("text-caption text-grey")
        
        # ターゲット列の概要
        target_col = state.get("target_col", "未設定")
        with ui.card().classes("bg-cyan-900/20 q-pa-xs border-cyan/20"):
            ui.label(f"🎯 目的変数: {target_col}").classes("text-subtitle2 text-cyan q-px-sm")

    # --- 2. 検索 & フィルタ ---
    # 検索ワードを保持するためのstate
    if "_col_search" not in state:
        state["_col_search"] = ""
    
    with ui.row().classes("full-width q-mb-md items-center q-gutter-md"):
        search = ui.input(
            label="列名で検索",
            placeholder="Search columns...",
            value=state["_col_search"],
            on_change=lambda e: _update_search(state, e.value)
        ).props("outlined dense clearable icon=search").classes("w-64")
        
        # クイックセット
        with ui.row().classes("q-gutter-sm"):
            ui.button("すべて説明変数に", on_click=lambda: _set_all_roles(state, "feature")).props("outline dense no-caps size=sm color=primary")
            ui.button("すべて除外", on_click=lambda: _set_all_roles(state, "exclude")).props("outline dense no-caps size=sm color=grey")

    # --- 3. カラムリスト (カード) ---
    search_term = state["_col_search"].lower()
    filtered_cols = [c for c in all_cols if not search_term or search_term in c.lower()]

    with ui.column().classes("full-width q-gutter-y-sm"):
        if not filtered_cols:
            ui.label("一致する列が見つかりません").classes("text-grey q-pa-lg text-center full-width")
        
        for col in filtered_cols:
            _render_column_card(state, df, col)

def _update_search(state: dict, val: str):
    state["_col_search"] = val or ""
    render_column_role_panel.refresh()

def _render_column_card(state: dict, df: pd.DataFrame, col: str):
    """個別のカラム設定カード"""
    # 現在の役割を判定
    current_role = "feature"
    if col == state.get("target_col"): current_role = "target"
    elif col in state.get("exclude_cols", []): current_role = "exclude"
    elif col == state.get("group_col"): current_role = "group"
    elif col == state.get("time_col"): current_role = "time"
    elif col == state.get("weight_col"): current_role = "weight"

    # メタ情報
    meta = _get_meta(state, col)
    current_mono = meta.get("monotonic", 0)

    # カードのデザイン
    card_classes = "full-width q-pa-sm transition-all duration-300 "
    if current_role == "target":
        card_class = card_classes + "bg-cyan-900/10 border-cyan/40"
    elif current_role == "exclude":
        card_class = card_classes + "bg-grey-900/20 opacity-60 grayscale"
    else:
        card_class = card_classes + "glass-card hover:border-cyan/30"

    with ui.card().classes(card_class).style("border: 1px solid rgba(0,188,212,0.1)"):
        with ui.row().classes("w-full items-center q-gutter-sm no-wrap"):
            # 1. 列情報
            with ui.column().classes("q-gutter-none").style("min-width: 140px; max-width: 180px;"):
                ui.label(col).classes("text-body2 text-bold truncate").tooltip(col)
                with ui.row().classes("items-center q-gutter-xs text-caption text-grey-5"):
                    ui.label(str(df[col].dtype))
                    ui.label("•")
                    ui.label(f"欠損:{df[col].isna().sum()/len(df)*100:.1f}%")

            ui.separator().props("vertical")

            # 2. 役割選択 (ワンクリック)
            with ui.row().classes("items-center q-gutter-xs flex-grow justify-center"):
                for role_key, role_label, role_color in _ROLE_OPTS:
                    is_active = (current_role == role_key)
                    btn_props = "unelevated" if is_active else "flat"
                    btn_color = role_color if is_active else "grey-6"
                    
                    def _on_role_click(_role=role_key, _col=col):
                        _update_role(state, _col, _role)

                    ui.button(role_label, on_click=_on_role_click).props(
                        f"dense no-caps {btn_props} color={btn_color} size=sm"
                    ).classes("q-px-sm").style("min-width: 70px; border-radius: 4px;")

            ui.separator().props("vertical")

            # 3. 単調性選択
            with ui.row().classes("items-center q-gutter-xs").style("min-width: 200px;"):
                ui.label("📈 単調性:").classes("text-caption text-grey q-mr-xs")
                for mono_val, mono_label in _MONO_OPTS:
                    is_active = (current_mono == mono_val)
                    btn_style = (
                        "background:rgba(123, 47, 247, 0.3); border:1px solid rgba(123, 47, 247, 0.6);"
                        if is_active else
                        "background:rgba(255,255,255,0.05);"
                    )
                    
                    def _on_mono_click(_val=mono_val, _col=col):
                        _set_meta(state, _col, "monotonic", _val)
                        render_column_role_panel.refresh()

                    ui.button(mono_label.split(" ")[0], on_click=_on_mono_click).props(
                        "flat dense no-caps size=sm"
                    ).style(btn_style + "min-width: 28px; border-radius: 4px;").tooltip(mono_label)

def _update_role(state: dict, col: str, role: str):
    """状態の役割を更新（整合性を保つ）"""
    df = state.get("df")
    
    # 以前の役割をクリーニング
    if state.get("target_col") == col: state["target_col"] = ""
    if col in state.get("exclude_cols", []):
        try: state["exclude_cols"].remove(col)
        except: pass
    if state.get("group_col") == col: state["group_col"] = ""
    if state.get("time_col") == col: state["time_col"] = ""
    if state.get("weight_col") == col: state["weight_col"] = ""

    # 新しい役割を適用
    if role == "target":
        state["target_col"] = col
        # タスクタイプの自動更新
        if df is not None:
            if pd.api.types.is_float_dtype(df[col]):
                state["task_type"] = "regression"
            else:
                state["task_type"] = "classification"
    elif role == "exclude":
        if "exclude_cols" not in state: state["exclude_cols"] = []
        if col not in state["exclude_cols"]: state["exclude_cols"].append(col)
    elif role == "group": state["group_col"] = col
    elif role == "time": state["time_col"] = col
    elif role == "weight": state["weight_col"] = col
    
    ui.notify(f"'{col}' の役割を更新しました", type="positive", timeout=800)
    state["precalc_done"] = False
    
    # 全体リフレッシュ
    render_column_role_panel.refresh()
    
    # 他のタブもリフレッシュが必要な場合
    refresh_all = state.get("_refresh_tabs")
    if refresh_all: refresh_all()

def _set_all_roles(state: dict, role: str):
    """全カラムの役割を一括設定"""
    df = state.get("df")
    if df is None: return
    
    if role == "feature":
        state["target_col"] = ""
        state["exclude_cols"] = []
        state["group_col"] = ""
        state["time_col"] = ""
        state["weight_col"] = ""
    elif role == "exclude":
        state["exclude_cols"] = list(df.columns)
        state["target_col"] = ""
    
    ui.notify("すべての列の役割を一括更新しました", type="info")
    render_column_role_panel.refresh()
