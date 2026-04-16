"""
記述子選択コンポーネント (Descriptor Selector)
プリセット、目的変数、相関係数、エンジン、検索、分散、数え上げ系 のタブ構成
"""
import logging
from typing import Dict, List, Set, Any, Optional

import numpy as np
import pandas as pd
from nicegui import ui, run

# カタログとアダプタのインポート
from frontend_nicegui.components.descriptor_catalog import get_catalog, SUPPORTED_ENGINES

logger = logging.getLogger(__name__)

# ============================================================
# 数え上げ系記述子の定義 (RDKit互換命名規則)
# ============================================================
COUNTING_DESCRIPTORS = [
    ("NumHAcceptors", "水素結合受容体数 (HBA)"),
    ("NumHDonors", "水素結合供与体数 (HBD)"),
    ("NumRotatableBonds", "回転可能結合数"),
    ("RingCount", "総環数"),
    ("NumAromaticRings", "芳香族環数"),
    ("NumAliphaticRings", "脂肪族環数"),
    ("NumSaturatedRings", "飽和環数"),
    ("NumHeterocycles", "複素環数"),
    ("HeavyAtomCount", "重原子数"),
    ("NumAliphaticCarbocycles", "脂肪族炭素環数"),
    ("NumAromaticHeterocycles", "芳香族複素環数"),
    ("NumAliphaticHeterocycles", "脂肪族複素環数"),
    ("NumSaturatedHeterocycles", "飽和複素環数"),
    ("NumAmideBonds", "アミド結合数"),
    ("NumSpiroAtoms", "スピロ原子数"),
    ("NumBridgeheadAtoms", "橋頭原子数"),
    ("NumAtomStereoCenters", "立体中心原子数"),
    ("NumUnspecifiedAtomStereoCenters", "未指定立体中心数"),
    ("FractionCSP3", "sp3炭素割合 (0.0-1.0)"),
    ("TPSA", "極性表面積 (Å²)"),
]

# ============================================================
# メインレンダリング関数
# ============================================================
def render_descriptor_selector(state: dict, on_selection_change=None):
    """
    記述子選択UIをレンダリングする
    """
    # 状態初期化
    if "descriptor_tab" not in state:
        state["descriptor_tab"] = "preset"
    if "selected_descriptors" not in state:
        state["selected_descriptors"] = set()
    elif isinstance(state["selected_descriptors"], (list, dict)):
        # 互換性維持: listならsetに、dictならフラットなsetに変換
        if isinstance(state["selected_descriptors"], dict):
            flat = set()
            for v in state["selected_descriptors"].values():
                if isinstance(v, list): flat.update(v)
            state["selected_descriptors"] = flat
        else:
            state["selected_descriptors"] = set(state["selected_descriptors"])

    if "available_descriptors" not in state:
        state["available_descriptors"] = set()
    
    with ui.column().classes("w-full") as container:
        container.set_id("descriptor-selector-root")
        
        # ─────────────────────────────────────────────
        # 上部情報バー
        # ─────────────────────────────────────────────
        with ui.row().classes("w-full items-center justify-between q-mb-md bg-grey-900 q-pa-sm rounded-lg"):
            ui.label("📦 特徴量セット:").classes("text-grey-400 text-xs")
            ui.chip("MolAI+PCA (5)", color="pink", size="sm").props("outline")
            ui.chip("汎用QSPR (5)", color="purple", size="sm").props("outline")
            ui.chip("相関Top-N (2)", color="blue", size="sm").props("outline")
            ui.space()
            
            n_sel = len(state['selected_descriptors'])
            n_avail = len(state.get("precalc_df").columns) if state.get("precalc_df") is not None else 0
            ui.chip(f"{n_sel}/{n_avail} 選択中", color="cyan", size="sm")
            
            with ui.row().classes("items-center q-gutter-xs"):
                ui.button(icon="save").props("flat dense color=primary").tooltip("セット保存")
                ui.button(icon="add").props("flat dense").tooltip("新規作成")
            
        # ─────────────────────────────────────────────
        # タブヘッダー
        # ─────────────────────────────────────────────
        with ui.row().classes("w-full items-center gap-1 q-mb-md border-b border-grey-800 pb-2"):
            _render_tab_button("✨ プリセット", "preset", state)
            _render_tab_button("🎯 目的変数", "target", state)
            _render_tab_button("📈 相関係数", "correlation", state)
            _render_tab_button("⚙️ エンジン", "engine", state)
            _render_tab_button("🔍 検索", "search", state)
            _render_tab_button("📊 分散", "variance", state)
            _render_tab_button("🔢 数え上げ系", "counting", state)

        # ─────────────────────────────────────────────
        # タブコンテンツ
        # ─────────────────────────────────────────────
        with ui.column().classes("w-full"):
            tab = state["descriptor_tab"]
            if tab == "preset":
                _render_preset_panel(state, on_selection_change)
            elif tab == "target":
                _render_target_panel(state)
            elif tab == "correlation":
                _render_correlation_panel(state)
            elif tab == "engine":
                _render_engine_panel(state)
            elif tab == "search":
                _render_search_panel(state, on_selection_change)
            elif tab == "variance":
                _render_variance_panel(state)
            elif tab == "counting":
                _render_counting_panel(state, on_selection_change)

# ============================================================
# ヘルパー関数群
# ============================================================
def _render_tab_button(label: str, tab_id: str, state: dict):
    is_active = state["descriptor_tab"] == tab_id
    active_classes = "bg-cyan-900/50 text-cyan-300 border-b-2 border-cyan-400"
    inactive_classes = "text-grey-400 hover:text-grey-200 hover:bg-grey-800"
    
    ui.button(
        label,
        on_click=lambda: _switch_tab(tab_id, state)
    ).classes(f"px-3 py-2 rounded-t-sm transition-all text-xs font-medium {active_classes if is_active else inactive_classes}").props("flat no-caps")

def _switch_tab(tab_id: str, state: dict):
    state["descriptor_tab"] = tab_id
    if state.get("_refresh_smiles_integrated"): state["_refresh_smiles_integrated"]()

def _render_descriptor_card(feat_id: str, feat_name: str, state: dict, on_selection_change=None):
    is_selected = feat_id in state["selected_descriptors"]
    card_classes = f"w-full q-pa-sm cursor-pointer transition-all {'bg-cyan-900/20 border-cyan-600' if is_selected else 'bg-grey-800/30 border-grey-700 hover:border-grey-500'} border rounded"
    
    with ui.card().classes(card_classes) as card:
        with ui.row().classes("w-full items-center no-wrap"):
            sw = ui.switch(
                value=is_selected,
                on_change=lambda e, fid=feat_id: _toggle_descriptor(fid, e.value, state, on_selection_change)
            ).props("dense size=sm color=cyan")
            
            with ui.column().classes("flex-grow overflow-hidden"):
                ui.label(feat_name).classes("text-xs font-bold text-grey-2 truncate")
                ui.label(feat_id).classes("text-[10px] text-grey-5 font-mono truncate")
            
            if is_selected:
                ui.icon("check", color="cyan", size="xs")

        card.on('click', lambda: sw.set_value(not sw.value))

def _toggle_descriptor(feat_id: str, is_selected: bool, state: dict, on_selection_change=None):
    if is_selected: state["selected_descriptors"].add(feat_id)
    else: state["selected_descriptors"].discard(feat_id)
    if on_selection_change: on_selection_change(state["selected_descriptors"])

# ============================================================
# 🔢 数え上げ系 タブ パネル
# ============================================================
def _render_counting_panel(state: dict, on_selection_change=None):
    with ui.card().classes("w-full bg-grey-900/50 q-pa-md border border-grey-800"):
        with ui.row().classes("w-full items-center justify-between q-mb-sm"):
            ui.label("🔢 数え上げ系記述子 (RDKit)").classes("text-lg font-bold text-cyan-400")
            n_sel = len([fid for fid, _ in COUNTING_DESCRIPTORS if fid in state['selected_descriptors']])
            ui.badge(f"{n_sel}個選択中", color="cyan")
        
        ui.markdown("""
        原子数、環数、官能基数などの**整数カウント**記述子です。
        解釈性が高く、モデルの判断根拠を明確にするのに非常に有効です。
        """).classes("text-xs text-grey-5 q-mb-md")
        
        with ui.grid(columns=3).classes("w-full q-gutter-sm"):
            for fid, fname in COUNTING_DESCRIPTORS:
                _render_descriptor_card(fid, fname, state, on_selection_change)
        
        with ui.row().classes("w-full q-mt-md justify-end q-gutter-sm"):
            ui.button("すべて選択", on_click=lambda: state["selected_descriptors"].update([f[0] for f in COUNTING_DESCRIPTORS]) or _switch_tab("counting", state)).props("outline size=sm")
            ui.button("選択解除", on_click=lambda: [state["selected_descriptors"].discard(f[0]) for f in COUNTING_DESCRIPTORS] or _switch_tab("counting", state)).props("outline size=sm")
            ui.button("よく使う5件", on_click=lambda: state["selected_descriptors"].update(["NumHAcceptors", "NumHDonors", "NumRotatableBonds", "RingCount", "TPSA"]) or _switch_tab("counting", state)).props("unelevated color=cyan size=sm")

# ============================================================
# ⚙️ エンジン タブ パネル（計算ロジック統合）
# ============================================================
def _render_engine_panel(state: dict):
    with ui.card().classes("w-full bg-grey-900/50 q-pa-md border border-grey-800"):
        ui.label("⚙️ 計算エンジン制御").classes("text-lg font-bold text-orange-400 q-mb-sm")
        
        ui.markdown("""
        使用する計算エンジンを選択して「特徴量を計算」ボタンを押してください。
        数千個の記述子を一度に計算可能です。
        """).classes("text-xs text-grey-5 q-mb-md")
        
        # エンジンリスト（一部抜粋・簡略化）
        engines = [
             ("RDKit", "物理化学・官能基", "⚡高速"),
             ("Mordred", "包括的2D/3D (1800+)", "🟡中速"),
             ("scikit-FP", "ECFP/MACCS等", "⚡高速"),
             ("xTB", "量子化学 (HOMO/LUMO)", "🔴低速"),
        ]
        
        with ui.grid(columns=2).classes("w-full q-gutter-md q-mb-md"):
            for name, dsc, speed in engines:
                with ui.row().classes("items-center justify-between bg-grey-800/40 q-pa-sm rounded"):
                    ui.label(f"📦 {name}").classes("text-sm font-bold")
                    ui.label(speed).classes("text-[10px] text-grey-6")
                    ui.switch(value=True).props("dense size=sm color=orange")
        
        # 計算実行ボタン
        with ui.column().classes("w-full items-center border-t border-grey-800 q-pt-md"):
            is_done = state.get("precalc_done", False)
            btn = ui.button(
                "📈 特徴量を計算する" if not is_done else "🔄 再計算",
                on_click=lambda: _run_manual_compute(state)
            ).props("unelevated color=orange size=lg").classes("px-8 py-2 text-bold")
            
            if is_done:
                ui.label(f"✅ {len(state.get('precalc_df').columns)} 個の記述子が計算済みです").classes("text-xs text-green q-mt-sm")

async def _run_manual_compute(state: dict):
    ui.notify("記述子の計算を開始します...", color="warning")
    # 既存のロジックを呼び出すか、ここに移植する（ここでは移植を簡略化）
    # 実際には descriptor_plugins_ui.py の _manual_compute を実行
    from frontend_nicegui.components.descriptor_plugins_ui import _manual_compute_logic
    await _manual_compute_logic(state)
    if state.get("_refresh_smiles_integrated"): state["_refresh_smiles_integrated"]()

# ============================================================
# ✨ プリセット タブ パネル
# ============================================================
def _render_preset_panel(state: dict, on_selection_change=None):
    with ui.card().classes("w-full bg-grey-900/50 q-pa-md border border-grey-800"):
        ui.label("✨ 主要記述子プリセット").classes("text-lg font-bold text-cyan-400 q-mb-sm")
        
        presets = [
            ("基本セット", ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors"]),
            ("溶解性分析", ["MolLogP", "TPSA", "LabuteASA", "FractionCSP3", "RingCount"]),
            ("反応性/量子", ["HomoEnergy", "LumoEnergy", "HomoLumoGap", "DipoleMoment"]),
        ]
        
        with ui.row().classes("w-full q-gutter-md"):
            for name, descs in presets:
                def _apply(d=descs):
                    state["selected_descriptors"].update(d)
                    ui.notify(f"「{name}」プリセットを適用しました")
                    _switch_tab("preset", state)
                
                with ui.card().classes("q-pa-sm bg-grey-800/30 border border-grey-700 hover:border-cyan-600 cursor-pointer").on('click', _apply):
                    ui.label(name).classes("text-sm font-bold")
                    ui.label(f"{len(descs)} 記述子").classes("text-[10px] text-grey-6")

# ============================================================
# 🎯 目的変数 タブ パネル
# ============================================================
def _render_target_panel(state: dict):
    df = state.get("df")
    cols = list(df.columns) if df is not None else []
    
    with ui.card().classes("w-full bg-grey-900/50 q-pa-md border border-grey-800"):
        ui.label("🎯 目的変数（ターゲット）設定").classes("text-lg font-bold text-purple-400 q-mb-sm")
        
        with ui.row().classes("w-full items-center q-gutter-md"):
            ui.select(
                options=cols,
                value=state.get("target_col"),
                label="解析対象の目的変数を選択",
                on_change=lambda e: (state.update({"target_col": e.value}), ui.notify(f"ターゲットを {e.value} に設定しました"))
            ).classes("flex-grow").props("outlined dense color=purple")
            
            ui.button("相関をチェック", on_click=lambda: _switch_tab("correlation", state)).props("outline color=purple size=sm")

        ui.markdown("""
        目的変数を選択すると、各記述子との相関係数が計算され、
        「📈 相関係数」タブで有効な特徴量を自動抽出できるようになります。
        """).classes("text-xs text-grey-5 q-mt-sm")

# ============================================================
# 📈 相関係数 タブ パネル
# ============================================================
def _render_correlation_panel(state: dict):
    with ui.card().classes("w-full bg-grey-900/50 q-pa-md border border-grey-800"):
        ui.label("📈 相関係数フィルタリング").classes("text-lg font-bold text-blue-400 q-mb-sm")
        
        target = state.get("target_col")
        if not target:
            ui.label("⚠️ まず「🎯 目的変数」を選択してください").classes("text-amber text-xs")
            return

        ui.markdown(f"**ターゲット: {target}** との相関に基づき記述子を抽出します。").classes("text-xs text-grey-4 q-mb-md")
        
        with ui.row().classes("w-full items-center q-gutter-md"):
            slider = ui.slider(min=0.1, max=0.9, step=0.05, value=0.3).props("label-always color=blue").classes("flex-grow")
            ui.button("上位記述子を選択", on_click=lambda: ui.notify(f"|r| > {slider.value} の記述子を選択しました (Mock)")).props("unelevated color=blue size=sm")

# ============================================================
# 🔍 検索 タブ パネル
# ============================================================
def _render_search_panel(state: dict, on_selection_change=None):
    precalc_df = state.get("precalc_df")
    all_available = list(precalc_df.columns) if precalc_df is not None else []
    
    with ui.card().classes("w-full bg-grey-900/50 q-pa-md border border-grey-800"):
        ui.label("🔍 記述子を直接検索").classes("text-lg font-bold text-green-400 q-mb-sm")
        
        search_container = ui.column().classes("w-full")
        
        def _update_results(e):
            term = e.value.lower()
            results_area.clear()
            with results_area:
                filtered = [c for c in all_available if term in c.lower()][:20]
                if not filtered:
                    ui.label("一致する記述子が見つかりません").classes("text-grey-6 text-xs")
                else:
                    with ui.grid(columns=2).classes("w-full q-gutter-sm"):
                        for fid in filtered:
                            _render_descriptor_card(fid, fid, state, on_selection_change)
        
        with ui.row().classes("w-full items-center q-gutter-md q-mb-md"):
            search_input = ui.input(
                placeholder="名称で検索 (例: Ring, MolLogP...)",
                on_change=_update_results
            ).classes("flex-grow").props("outlined dense color=green")
            ui.icon("search", color="green")
            
        results_area = ui.column().classes("w-full")
        ui.label("計算済みの全記述子から個別に選択できます。").classes("text-[10px] text-grey-6")

# ============================================================
# 📊 分散 タブ パネル
# ============================================================
def _render_variance_panel(state: dict):
    with ui.card().classes("w-full bg-grey-900/50 q-pa-md border border-grey-800"):
        ui.label("📊 低分散フィルタ").classes("text-lg font-bold text-yellow-400 q-mb-sm")
        
        ui.markdown("""
        すべてのサンプルでほぼ同じ値を持つ（分散が極めて低い）記述子を一括除外します。
        これらの記述子はモデル学習に寄与せず、ノイズとなる可能性があります。
        """).classes("text-xs text-grey-5 q-mb-md")
        
        with ui.row().classes("w-full items-center justify-between"):
            ui.button("低分散記述子を自動除外 (Mock)", on_click=lambda: ui.notify("一定値の記述子 12個 を除外しました")).props("unelevated color=yellow-9 text-black size=sm")
            ui.label("閾値: 95% 同一値").classes("text-[10px] text-grey-6")
