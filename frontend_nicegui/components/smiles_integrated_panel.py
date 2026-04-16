# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/smiles_integrated_panel.py

SMILES特徴量に関連する全フロー（生成・セット管理・制約設定）を統合したコンポーネント。
"""
from __future__ import annotations
import logging
from typing import Any
from nicegui import ui

logger = logging.getLogger(__name__)

def render_smiles_integrated_panel(state: dict[str, Any]) -> None:
    """SMILES特徴量・制約統合パネルを描画する。"""
    
    ui.label("⚗️ SMILES特徴量・制約設定").classes("text-2xl font-bold q-mb-md")
    
    ui.markdown("""
    SMILESから記述子を生成し、解析用の特徴量セットを作成、さらに化学的制約を設定します。
    複数の特徴量セットを比較することで、より精度の高いモデル構築が可能です。
    """).classes("text-body2 text-grey-7 q-mb-md")

    with ui.tabs().classes("full-width").props("dense active-color=cyan indicator-color=cyan text-color=grey") as inner_tabs:
        tab_gen = ui.tab("gen", label="🔬 特徴量生成", icon="auto_fix_normal")
        tab_sets = ui.tab("sets", label="📦 特徴量セット管理", icon="folder_open")
        tab_cons = ui.tab("cons", label="📐 制約設定", icon="science")

    with ui.tab_panels(inner_tabs, value="gen").classes("full-width"):
        
        # ── 1. 特徴量生成 ──
        with ui.tab_panel("gen"):
            _gen_container = ui.column().classes("full-width")
            def _rebuild_gen():
                _gen_container.clear()
                with _gen_container:
                    try:
                        from frontend_nicegui.components.smiles_feature_panel import render_smiles_feature_panel
                        render_smiles_feature_panel(state)
                    except Exception as e:
                        logger.error(f"Error rendering smiles_feature_panel: {e}", exc_info=True)
                        ui.label(f"⚠️ 生成パネルの読込エラー: {e}").classes("text-red")
            _rebuild_gen()

        # ── 2. 特徴量セット管理 ──
        with ui.tab_panel("sets"):
            _sets_container = ui.column().classes("full-width")
            def _rebuild_sets():
                _sets_container.clear()
                with _sets_container:
                    try:
                        from frontend_nicegui.components.feature_set_manager import render_feature_set_manager
                        render_feature_set_manager(state)
                    except Exception as e:
                        logger.error(f"Error rendering feature_set_manager: {e}", exc_info=True)
                        ui.label(f"⚠️ セット管理パネルの読込エラー: {e}").classes("text-red")
            _rebuild_sets()

        # ── 3. 制約設定 ──
        with ui.tab_panel("cons"):
            _cons_container = ui.column().classes("full-width")
            def _rebuild_cons():
                _cons_container.clear()
                with _cons_container:
                    try:
                        from frontend_nicegui.components.constraint_panel import render_constraint_panel
                        render_constraint_panel(state)
                    except Exception as e:
                        logger.error(f"Error rendering constraint_panel: {e}", exc_info=True)
                        ui.label(f"⚠️ 制約パネルの読込エラー: {e}").classes("text-red")
            _rebuild_cons()

    # リフレッシュ関数を登録
    def _refresh_all():
        _rebuild_gen()
        _rebuild_sets()
        _rebuild_cons()
    
    state["_refresh_smiles_integrated"] = _refresh_all
