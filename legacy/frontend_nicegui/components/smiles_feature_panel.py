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

    # ── 2. 記述子プラグイン設定 ──
    from frontend_nicegui.components.descriptor_plugins_ui import render_descriptor_plugins
    render_descriptor_plugins(state)

    # ── 3. 進化した化学的ドメイン知見設定 (User Request) ──
    with ui.card().classes("full-width glass-card q-mt-md p-4"):
        with ui.row().classes("items-center q-gutter-sm mb-2"):
            ui.icon("psychology", color="pink", size="sm")
            ui.label("🧪 化学的ドメイン知見の自動適用").classes("text-md font-bold text-white")
            ui.switch(value=state.get("auto_domain_knowledge", True), 
                      on_change=lambda e: state.update({"auto_domain_knowledge": e.value})).props("dense")
        
        ui.label("生成された記述子（MW, LogP等）に対し、物理化学的妥当性を考慮した単調性制約を自動的に提案・適用します。").classes("text-caption text-grey-5")
        
        if state.get("auto_domain_knowledge", True):
            with ui.row().classes("q-gutter-sm q-mt-sm"):
                ui.chip("MW ↘", color="pink-9", text_color="white").tooltip("分子量が増えると水溶性は下がる傾向 (Solubility予測時)")
                ui.chip("LogP ↘", color="pink-9", text_color="white").tooltip("脂溶性が高いと水溶性は下がる")
                ui.chip("TPSA ↗", color="pink-9", text_color="white").tooltip("極性面積が大きいと水溶性は上がる")
            
            def _apply_suggestions():
                applied_count = 0
                all_feature_cols = all_cols
                if state.get("precalc_df") is not None:
                    all_feature_cols = list(state["precalc_df"].columns)
                
                for col in all_feature_cols:
                    for s_key, info in CHEMICAL_RATIONALES.items():
                        if s_key.lower() in col.lower() and info["suggest"] != 0:
                            _set_meta(state, col, "monotonic", info["suggest"])
                            applied_count += 1
                            break
                ui.notify(f"✅ {applied_count} 個の記述子にドメイン知見に基づく制約を提案しました。", type="positive")
                if state.get("_refresh_tabs"): state["_refresh_tabs"]()
                _render_constraints_grid.refresh()

            ui.button("✨ 推奨制約を今すぐ適用", on_click=_apply_suggestions).props("outline no-caps size=sm color=pink").classes("q-mt-sm")

    # ── 4. 🧪 化学的な制約設定 (Dynamic Configuration) ──
    @ui.refreshable
    def _render_constraints_grid():
        active_descs = state.get("selected_descriptors", [])
        # キー記述子のみを抽出
        key_descs_in_use = []
        for d in active_descs:
            for k in CHEMICAL_RATIONALES:
                if k.lower() in d.lower():
                    key_descs_in_use.append((d, k))
                    break
        
        if not key_descs_in_use:
            return

        with ui.card().classes("full-width glass-card q-mt-md p-4"):
            with ui.row().classes("items-center q-gutter-sm mb-3"):
                ui.icon("straighten", color="cyan", size="sm")
                ui.label("🧪 生成済み特徴量の詳細制約設定").classes("text-md font-bold text-white")
            
            ui.label("選択された記述子のうち、化学적意味付けが明確なものに対して個別に制約を設定できます。").classes("text-caption text-grey-5 q-mb-md")

            with ui.grid(columns=2).classes("w-full q-gutter-md"):
                for col_name, key in key_descs_in_use:
                    info = CHEMICAL_RATIONALES[key]
                    meta = _get_meta(state, col_name)
                    mono_val = str(meta.get("monotonic", 0))

                    with ui.card().classes("q-pa-sm bg-black-20"):
                        with ui.row().classes("items-center justify-between full-width no-wrap"):
                            with ui.column().classes("col-grow"):
                                ui.label(col_name).classes("text-sm font-bold text-white")
                                ui.label(info["rationale"]).classes("text-xs text-grey-5")
                            
                            # 単調性トグル
                            with ui.row().classes("items-center q-gutter-xs"):
                                for val, icon in [("0", "➖"), ("1", "↗"), ("-1", "↘")]:
                                    active = (val == mono_val)
                                    btn_color = "cyan" if active else "grey-8"
                                    
                                    def _change(v=val, c=col_name):
                                        _set_meta(state, c, "monotonic", int(v))
                                        _render_constraints_grid.refresh()
                                        ui.notify(f"{c}: 制約を更新しました", color="positive", position="bottom-right", timeout=1000)
                                        if state.get("_refresh_tabs"): state["_refresh_tabs"]()

                                    ui.button(icon, on_click=_change).props(f"flat dense unelevated size=sm color={btn_color}").classes("q-px-xs")

    _render_constraints_grid()

    # ── 5. 計算結果サマリー ──
    if state.get("precalc_done") and state.get("precalc_df") is not None:
        from frontend_nicegui.components.data_tab import _show_descriptor_summary
        _show_descriptor_summary(state, ui.column().classes("full-width q-mt-md"))

    # stateにリフレッシュ関数を登録
    state["_refresh_smiles_constraints"] = _render_constraints_grid.refresh
