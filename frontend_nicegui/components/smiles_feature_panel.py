"""
frontend_nicegui/components/smiles_feature_panel.py

SMILES記述子生成、混合物設定、および化学的ドメイン知識（単調性制約の自動提案）を管理するパネル。
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from nicegui import ui

from frontend_nicegui.components.column_meta_editor import _set_meta, _get_meta

logger = logging.getLogger(__name__)

# ── 定数: 化学的根拠と推奨制約 ──
CHEMICAL_RATIONALES = {
    "MolWt": {"label": "分子量 (MW)", "rationale": "分子量↑ → 分子間相互作用↑ → 水溶性↓", "suggest": -1},
    "MolLogP": {"label": "脂溶性 (LogP)", "rationale": "疎水性↑ → 水への溶解性↓", "suggest": -1},
    "TPSA": {"label": "極性面積 (TPSA)", "rationale": "極性↑ → 水との相互作用↑ → 水溶性↑", "suggest": 1},
    "NumHDonors": {"label": "HBD 数", "rationale": "水素結合供与体↑ → 親水性↑ → 水溶性↑", "suggest": 1},
    "NumHAcceptors": {"label": "HBA 数", "rationale": "水素結合受容体↑ → 親水性↑ → 水溶性↑", "suggest": 1},
    "NumRotatableBonds": {"label": "回転可能結合数", "rationale": "柔軟性↑ → 溶解度に複雑な影響", "suggest": 0},
    "RingCount": {"label": "環数", "rationale": "環数↑ → 剛性↑ → 溶解性↓", "suggest": -1},
}

def render_smiles_feature_panel(state: dict[str, Any]) -> None:
    """SMILES特徴量生成とドメイン知識設定UIを描画する。"""
    
    if state.get("df") is None:
        ui.label("⚠️ まずデータを読み込んでください").classes("text-amber q-pa-md")
        return

    all_cols = list(state["df"].columns)

    # ── 1. SMILES / 混合成分設定 ──
    with ui.expansion("🧬 SMILES / 混合成分設定", icon="science").classes("full-width glass-card q-mb-md").props("default-opened"):
        if "smiles_components" not in state:
            scol = state.get("smiles_col", "")
            if scol and scol in all_cols:
                state["smiles_components"] = [{"smiles_col": scol, "fraction_col": "（なし）"}]
            else:
                state["smiles_components"] = []

        comps_container = ui.column().classes("full-width q-gutter-xs p-2")
        
        def _render_comps():
            comps_container.clear()
            with comps_container:
                smiles_opts = ["（なし）"] + all_cols
                frac_opts = ["（なし）"] + all_cols
                
                for i, comp in enumerate(state["smiles_components"]):
                    with ui.row().classes("items-center full-width justify-between no-wrap"):
                        def _on_s(e, idx=i):
                            state["smiles_components"][idx]["smiles_col"] = e.value
                            state["precalc_done"] = False
                            if idx == 0:
                                state["smiles_col"] = e.value if e.value != "（なし）" else ""
                        def _on_f(e, idx=i):
                            state["smiles_components"][idx]["fraction_col"] = e.value
                            state["precalc_done"] = False
                            
                        s_val = comp.get("smiles_col", "（なし）")
                        f_val = comp.get("fraction_col", "（なし）")
                        
                        ui.select(smiles_opts, value=s_val if s_val in smiles_opts else "（なし）", 
                                  label=f"SMILES {i+1}", on_change=_on_s).classes("col-5").props("dense outlined")
                        ui.select(frac_opts, value=f_val if f_val in frac_opts else "（なし）",
                                  label=f"割合(%) {i+1}", on_change=_on_f).classes("col-5").props("dense outlined")
                                  
                        def _del(idx=i):
                            state["smiles_components"].pop(idx)
                            if len(state["smiles_components"]) == 0:
                                state["smiles_col"] = ""
                            _render_comps()
                        ui.button(icon="close", on_click=_del).props("flat dense color=red").classes("col-1")
                        
                with ui.row().classes("items-center full-width justify-between q-mt-xs"):
                    ui.button("＋ 成分追加", on_click=lambda: (state["smiles_components"].append({"smiles_col": "（なし）", "fraction_col": "（なし）"}), _render_comps())).props("outline dense color=cyan size=sm")
                    
                    ui.radio({"wt": "wt%", "mol": "mol%"}, value=state.get("fraction_type", "wt"),
                             on_change=lambda e: state.update({"fraction_type": e.value})).props("dense inline").tooltip("割合の単位 (wt% / mol%)")
        
        _render_comps()
        ui.label("構成成分を追加し、加重平均による混合系の特徴量を自動計算します。").classes("text-caption text-grey-5 q-mb-sm p-2")

    if not state.get("smiles_col"):
        ui.label("ℹ️ SMILES列が設定されていません。数値データのみの場合はそのまま EDA タブへ進んでください。").classes("text-grey-5 q-pa-md")
        return

    # ── 2. 記述子選択ワークフロー ──
    from frontend_nicegui.components.descriptor_selector import render_descriptor_selector
    render_descriptor_selector(state)

    # ── 3. (REMOVED) 化学的ドメイン知見設定 ──
    # ここに配置されていたドメイン知見設定は、論理的整合性のために「制約設定」タブへ移動されました。

    # ── 5. 計算結果サマリー ──
    if state.get("precalc_done") and state.get("precalc_df") is not None:
        from frontend_nicegui.components.data_tab import _show_descriptor_summary
    # stateにリフレッシュ関数を登録 (統合パネル側で管理するため不要な場合は削除可能だが、互換性のためにnoopを置くか削除)
    pass
