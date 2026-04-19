"""
列の役割と単調性設定パネル

特徴量グルーピング機能を統合
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

import pandas as pd

from nicegui import ui

logger = logging.getLogger(__name__)


@ui.refreshable
def render_column_role_panel(state: dict[str, Any]):
    """列の役割と単調性設定パネル"""
    ui.label("📋 列の役割・単調性・グルーピング").classes("text-2xl font-bold q-mb-md")
    
    ui.markdown("""
1. **列の役割**: 説明変数/目的変数/除外を設定
2. **単調性制約**: 化学적知見に基づく単調性関係を定義
3. **特徴量グルーピング**: 同じ物理量の特徴量をグループ化し、整合性のある標準化を実現
    """).classes("text-body2 text-grey-7 q-mb-md")
    
    if "df" not in state or state["df"] is None:
        ui.notify("先にデータを読み込んでください", type="warning")
        return
    
    df = state["df"]
    
    # --- 状態の初期化と整合性チェック ---
    if "column_roles" not in state:
        state["column_roles"] = {}
    
    # 既存の state から復元
    target = state.get("target_col")
    excludes = state.get("exclude_cols", [])
    for col in df.columns:
        if col == target:
            state["column_roles"][col] = "target"
        elif col in excludes:
            state["column_roles"][col] = "exclude"
        else:
            state["column_roles"][col] = "feature"
    
    # ─────────────────────────────────────────────
    # 1. 列の役割設定（テーブル編集）
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("1️⃣ 列の役割設定").classes("text-lg font-bold q-mb-sm")
        ui.markdown("列名をクリックして役割を設定（説明変数/目的変数/除外）").classes("text-sm text-grey-7 q-mb-sm")
        _render_column_role_table(state)
    
    # ─────────────────────────────────────────────
    # 2. 単調性制約設定
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("2️⃣ 単調性制約設定").classes("text-lg font-bold q-mb-sm")
        ui.markdown("特徴量と目的変数の間の単調性関係を定義（化学的知見の反映）").classes("text-sm text-grey-7 q-mb-sm")
        _render_monotonicity_section(state)
    
    # ─────────────────────────────────────────────
    # 3. 特徴量グルーピング設定
    # ─────────────────────────────────────────────
    with ui.card().classes("w-full q-mb-md"):
        ui.label("3️⃣ 特徴量グルーピング").classes("text-lg font-bold q-mb-sm")
        ui.markdown("""
関連する特徴量をグループ化し、**グループ単位で標準化**します。
同じ温度センサー群などが同じスケールで扱われ、LASSO/RIDGE/RBFでの計算が物理的に妥当になります。
        """).classes("text-sm text-grey-7 q-mb-sm")
        
        # 特徴量グルーピングパネルを呼び出し
        from frontend_nicegui.components.feature_grouping_panel import render_feature_grouping_panel
        render_feature_grouping_panel(state)


def _render_column_role_table(state: dict):
    """列の役割設定テーブル"""
    df = state["df"]
    roles = state.get("column_roles", {})
    
    # 各行のデータ作成
    rows = []
    for col in df.columns:
        role = roles.get(col, "feature")
        rows.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "missing": f"{df[col].isna().sum() / len(df) * 100:.1f}%",
            "role": role
        })
    
    # テーブル定義
    columns = [
        {"name": "name", "label": "列名", "field": "name", "align": "left", "sortable": True},
        {"name": "dtype", "label": "データ型", "field": "dtype", "align": "center"},
        {"name": "missing", "label": "欠損(%)", "field": "missing", "align": "right"},
        {
            "name": "role",
            "label": "役割",
            "field": "role",
            "align": "left",
            "editable": True,
            "type": "select",
            "options": ["feature", "target", "exclude"]
        }
    ]
    
    # 編集イベント
    def on_cell_edit(e):
        col_name = e.row["name"]
        if e.column == "role":
            new_role = e.value
            state["column_roles"][col_name] = new_role
            
            # --- 既存 state との同期 ---
            if new_role == "target":
                state["target_col"] = col_name
                # 他の列が target だった場合は feature に戻す（単一 target 制約）
                for c, r in state["column_roles"].items():
                    if c != col_name and r == "target":
                        state["column_roles"][c] = "feature"
            elif new_role == "exclude":
                if "exclude_cols" not in state:
                    state["exclude_cols"] = []
                if col_name not in state["exclude_cols"]:
                    state["exclude_cols"].append(col_name)
                if state.get("target_col") == col_name:
                    state["target_col"] = ""
            else:  # feature
                if col_name == state.get("target_col"):
                    state["target_col"] = ""
                if col_name in state.get("exclude_cols", []):
                    state["exclude_cols"].remove(col_name)
            
            role_labels = {"feature": "説明変数", "target": "目的変数", "exclude": "除外"}
            ui.notify(f"'{col_name}' → {role_labels[new_role]}", color="positive", timeout=1000)
            state["precalc_done"] = False
            render_column_role_panel.refresh()
    
    table = ui.table(columns=columns, rows=rows, row_key="name").classes("w-full").props("dense")
    table.on("cell-edit", on_cell_edit)


def _render_monotonicity_section(state: dict):
    """単調性制約設定セクション"""
    column_roles = state.get("column_roles", {})
    feature_cols = [col for col, role in column_roles.items() if role == "feature"]
    
    if not feature_cols:
        ui.notify("説明変数が設定されていません", type="info")
        return
    
    # 既存の column_meta か monotonicity_constraints から取得
    from frontend_nicegui.components.column_meta_editor import _get_meta, _set_meta
    
    with ui.grid(columns=3).classes("w-full q-gutter-md"):
        for col in feature_cols:
            with ui.card().classes("w-full"):
                ui.label(col).classes("font-bold text-sm")
                meta = _get_meta(state, col)
                current_mono_val = meta.get("monotonic", 0)
                
                # 表示用ラベルへの変換
                val_to_key = {0: "none", 1: "increasing", -1: "decreasing"}
                key_to_val = {"none": 0, "increasing": 1, "decreasing": -1}
                current = val_to_key.get(current_mono_val, "none")
                
                select = ui.select(
                    options=[
                        ("none", "－ 制約なし"),
                        ("increasing", "↗ 単調増加"),
                        ("decreasing", "↘ 単調減少"),
                    ],
                    value=current,
                    label="単調性"
                ).props("dense outlined").classes("w-full")
                
                def on_change(e, col_name=col):
                    new_val = key_to_val.get(e.value, 0)
                    _set_meta(state, col_name, "monotonic", new_val)
                    labels = {"none": "制約なし", "increasing": "増加 ↗", "decreasing": "減少 ↘"}
                    ui.notify(f"'{col_name}' → {labels[e.value]}", color="positive", timeout=1000)
                
                select.on("change", on_change)
