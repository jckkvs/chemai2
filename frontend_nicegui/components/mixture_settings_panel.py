"""
frontend_nicegui/components/mixture_settings_panel.py

混合物特徴量抽出における記述子ごとの加重方法（Weight/Mole）を
ユーザーが手動で上書き設定するためのコンポーネント。
"""
from __future__ import annotations

import json
import os
import logging
from typing import Dict, List, Any

from nicegui import ui

from backend.chem.descriptor_weighting_classifier import (
    EXPLICIT_DESCRIPTOR_MAP,
    classify_descriptor,
    save_manual_mappings,
    _get_json_path
)

logger = logging.getLogger(__name__)

def render_mixture_settings_content() -> None:
    """混合物記述子の加重方法設定テーブルの内容を描画する"""
    
    # 状態保持用のバッファ
    json_path = _get_json_path()
    manual_data: Dict[str, Dict[str, str]] = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                manual_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load manual mappings: {e}")

    def get_all_rows() -> List[Dict[str, Any]]:
        # 明示的なマップと手動設定の和を表示
        all_keys = set(EXPLICIT_DESCRIPTOR_MAP.keys()) | set(manual_data.keys())
        rows = []
        for k in sorted(all_keys):
            wtype, rationale = classify_descriptor(k)
            rows.append({
                "name": k,
                "weighting": wtype,
                "rationale": rationale,
                "manual": k in manual_data
            })
        return rows

    def refresh_table():
        table.rows = get_all_rows()

    with ui.column().classes("full-width q-gutter-sm q-pa-md"):
        ui.markdown("""
        **混合物の記述子計算時の加重方式を上書きします。**
        - **weight**: 質量に比例する性質（密度、表面積など）
        - **mole**: 分子数に比例する性質（電荷、軌道エネルギーなど）
        - **context**: 文脈依存（デフォルト）
        """).classes("text-caption text-grey-6")

        with ui.row().classes("full-width items-center q-gutter-md"):
            search_input = ui.input(placeholder="記述子名で検索...").classes("grow").props("dense filled dark clearable icon=search")
            
        columns = [
            {"name": "name", "label": "記述子名", "field": "name", "sortable": True, "align": "left"},
            {"name": "weighting", "label": "加重方法", "field": "weighting", "sortable": True, "align": "center"},
            {"name": "manual", "label": "手動設定", "field": "manual", "sortable": True, "align": "center"},
            {"name": "rationale", "label": "根拠 / 備考", "field": "rationale", "align": "left"},
            {"name": "actions", "label": "操作", "field": "id", "align": "center"},
        ]

        table = ui.table(
            columns=columns, 
            rows=get_all_rows(), 
            pagination=10
        ).classes("full-width").props("dense flat bordered dark virtual-scroll")
        
        # 検索フィルタのバインド
        search_input.bind_value_to(table, "filter")

        # セル描画のカスタマイズ
        table.add_slot("body-cell-weighting", """
            <q-td :props="props">
                <q-chip :color="props.value === 'weight' ? 'green' : (props.value === 'mole' ? 'blue' : 'grey')" 
                        text-color="white" dense size="sm">
                    {{ props.value }}
                </q-chip>
            </q-td>
        """)

        table.add_slot("body-cell-manual", """
            <q-td :props="props">
                <q-icon v-if="props.value" name="edit" color="orange" size="xs">
                    <q-tooltip>ユーザー設定あり</q-tooltip>
                </q-icon>
                <q-icon v-else name="auto_awesome" color="grey-6" size="xs">
                    <q-tooltip>システムデフォルト</q-tooltip>
                </q-icon>
            </q-td>
        """)

        def update_weighting(row_name: str, new_type: str):
            current_w, current_r = classify_descriptor(row_name)
            manual_data[row_name] = {
                "weighting": new_type,
                "rationale": current_r if "Loaded from JSON" not in current_r else "User override"
            }
            ui.notify(f"【{row_name}】を {new_type} に設定しました")
            refresh_table()

        def reset_to_default(row_name: str):
            if row_name in manual_data:
                del manual_data[row_name]
                ui.notify(f"【{row_name}】をデフォルトに戻しました")
                refresh_table()

        # 操作ボタン
        table.add_slot("body-cell-actions", """
            <q-td :props="props">
                <q-btn-group flat>
                    <q-btn flat round size="xs" icon="scale" color="green" @click="$parent.$emit('set_weight', props.row.name)">
                        <q-tooltip>重量比 (weight) に設定</q-tooltip>
                    </q-btn>
                    <q-btn flat round size="xs" icon="bubble_chart" color="blue" @click="$parent.$emit('set_mole', props.row.name)">
                        <q-tooltip>モル比 (mole) に設定</q-tooltip>
                    </q-btn>
                    <q-btn flat round size="xs" icon="history" color="orange" @click="$parent.$emit('reset', props.row.name)">
                        <q-tooltip>デフォルトに戻す</q-tooltip>
                    </q-btn>
                </q-btn-group>
            </q-td>
        """)

        # イベントハンドリング
        table.on("set_weight", lambda e: update_weighting(e.args, "weight"))
        table.on("set_mole", lambda e: update_weighting(e.args, "mole"))
        table.on("reset", lambda e: reset_to_default(e.args))

        async def save_all():
            ok, msg = save_manual_mappings(manual_data)
            ui.notify(msg, type="positive" if ok else "negative")

        with ui.row().classes("full-width justify-end q-mt-sm"):
            ui.button("💾 加重設定を保存", on_click=save_all).props("unelevated color=purple-6 no-caps icon=save")
