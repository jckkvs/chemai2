"""
frontend_nicegui/components/column_role_panel.py

列の役割（目的変数・説明変数・除外）と、単調性制約を統合して設定するパネル。
AG Grid を使用してプレミアムな操作感を提供します。
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from nicegui import ui

from frontend_nicegui.components.column_meta_editor import _get_meta, _set_meta

# 単調性表示マップ
_MONO_DISPLAY = {
    0: "－ なし",
    1: "↗ 増加",
    -1: "↘ 減少",
    2: "〰️ 自動"
}

def render_column_role_panel(state: dict[str, Any]) -> None:
    """列の役割と単調性制約の設定UIを描画する。"""
    
    if state.get("df") is None:
        ui.label("⚠️ まずデータを読み込んでください").classes("text-amber q-pa-md")
        return

    df = state["df"]
    all_cols = list(df.columns)

    with ui.row().classes("full-width items-center justify-between q-mb-sm"):
        with ui.row().classes("items-center q-gutter-sm"):
            ui.icon("label_important", color="cyan").classes("text-h6")
            ui.label("列の役割と単調性設定").classes("text-subtitle1 text-bold")
        
        # ターゲット列の表示
        ui.label(f"🎯 目的変数: {state.get('target_col', '未設定')}").classes("text-subtitle2 text-cyan bg-cyan-900/20 q-px-sm rounded")

    ui.label('「役割」や「単調性」セルを'
             '<b class="text-cyan-400">クリック</b>して変更してください。'
             '化学的知見（MW増加→溶解度減少など）を反映できます。').classes('text-caption text-grey-5 q-mb-md')

    # AG Gridのデータ準備
    def _get_row_data():
        rows = []
        for col in all_cols:
            role = "説明変数"
            if col == state.get("target_col"): role = "目的変数"
            elif col in state.get("exclude_cols", []): role = "除外"
            elif col == state.get("group_col"): role = "グループID"
            elif col == state.get("time_col"): role = "時系列"
            elif col == state.get("weight_col"): role = "Sample Weight"
            
            meta = _get_meta(state, col)
            mono_val = meta.get("monotonic", 0)
            
            row_data_item = {
                "col_name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique(dropna=True)),
                "na_pct": round(int(df[col].isna().sum()) / len(df) * 100, 1) if len(df) > 0 else 0,
                "role": role,
                "monotonic": _MONO_DISPLAY.get(mono_val, "－ なし"),
                "_mono_key": mono_val,
                "_role_key": role
            }
            rows.append(row_data_item)
        return rows

    grid_options = {
        "columnDefs": [
            {"headerName": "列名", "field": "col_name", "sortable": True, "filter": True, "width": 200},
            {"headerName": "データ型", "field": "dtype", "width": 110},
            {"headerName": "欠損(%)", "field": "na_pct", "width": 90},
            {
                "headerName": "役割", 
                "field": "role", 
                "width": 160,
                "cellStyle": {"backgroundColor": "rgba(0, 188, 212, 0.08)", "cursor": "pointer", "fontWeight": "bold"}
            },
            {
                "headerName": "単調性制約", 
                "field": "monotonic", 
                "width": 160,
                "cellStyle": {"backgroundColor": "rgba(123, 47, 247, 0.08)", "cursor": "pointer", "fontWeight": "bold"}
            }
        ],
        "rowData": _get_row_data(),
        "rowSelection": "single",
        "suppressRowClickSelection": True,
    }

    grid = ui.aggrid(grid_options).classes("full-width").style("height: 480px;")

    def _open_edit_dialog(col_name: str, field: str, current_value: Any):
        """編集ダイアログを表示"""
        title = "単調性制約" if field == "monotonic" else "列の役割"
        
        with ui.dialog() as dlg, ui.card().classes('q-pa-md glass-card').style('min-width: 320px;'):
            ui.label(f"'{col_name}' の {title}").classes("text-lg font-bold q-mb-md")
            
            if field == "monotonic":
                options = {k: v for k, v in _MONO_DISPLAY.items()}
                # 現在の数値をキーにして初期化
                selected = ui.select(options=options, value=current_value).props("dense outlined").classes("w-full")
            else:
                options = ["説明変数", "目的変数", "除外", "グループID", "時系列", "Sample Weight"]
                selected = ui.select(options=options, value=current_value).props("dense outlined").classes("w-full")
            
            with ui.row().classes("w-full justify-end q-mt-lg gap-2"):
                ui.button("キャンセル", on_click=dlg.close).props("flat")
                ui.button("保存", on_click=lambda: _save_and_refresh(col_name, field, selected.value, dlg), 
                          color="cyan").props("unelevated")
        dlg.open()

    def _save_and_refresh(col_name, field, new_val, dlg):
        if field == "role":
            # 前の役割をクリア
            if col_name == state.get("target_col"): state["target_col"] = ""
            if col_name in state.get("exclude_cols", []): 
                try: state["exclude_cols"].remove(col_name)
                except: pass
            if col_name == state.get("group_col"): state["group_col"] = ""
            if col_name == state.get("time_col"): state["time_col"] = ""
            if col_name == state.get("weight_col"): state["weight_col"] = ""
            
            # 新しい役割を設定
            if new_val == "目的変数":
                state["target_col"] = col_name
                state["task_type"] = "regression" if pd.api.types.is_float_dtype(df[col_name]) else "classification"
            elif new_val == "除外":
                if "exclude_cols" not in state: state["exclude_cols"] = []
                if col_name not in state["exclude_cols"]: state["exclude_cols"].append(col_name)
            elif new_val == "グループID": state["group_col"] = col_name
            elif new_val == "時系列": state["time_col"] = col_name
            elif new_val == "Sample Weight": state["weight_col"] = col_name
            ui.notify(f"'{col_name}' の役割を {new_val} に変更しました", type="positive")
        
        elif field == "monotonic":
            _set_meta(state, col_name, "monotonic", int(new_val))
            ui.notify(f"'{col_name}' の単調性制約を更新しました", type="positive")

        state["precalc_done"] = False
        dlg.close()
        
        # グリッドの更新
        grid.options['rowData'] = _get_row_data()
        grid.update()
        
        refresh = state.get("_refresh_tabs")
        if refresh: refresh()

    async def handle_cell_click(e):
        field = e.args.get("colId")
        if field not in ["role", "monotonic"]:
            return
        
        data = e.args.get("data", {})
        col_name = data.get("col_name")
        curr_val = data.get("_mono_key") if field == "monotonic" else data.get("role")
        
        _open_edit_dialog(col_name, field, curr_val)

    grid.on('cellClicked', handle_cell_click)
