"""
SMILES特徴量・制約設定 統合パネル
グルーピング機能をサブタブとして追加
"""
from __future__ import annotations
import logging
from typing import Dict, List, Any
from nicegui import ui

logger = logging.getLogger(__name__)

def render_smiles_integrated_panel(state: dict[str, Any]):
    """
    SMILES特徴量・制約設定パネル（統合版）
    
    4つのタブに論理的に整理：
    1. 特徴量生成
    2. 特徴量セット管理
    3. 特徴量グルーピング
    4. 制約設定
    """
    
    ui.label("⚗️ SMILES特徴量・制約設定").classes("text-2xl font-bold q-mb-md")
    
    ui.markdown("""
    SMILESから分子記述子を生成し、グルーピング・制約を設定します。
    複数の特徴量セットを作成・選択できます。
    """).classes("text-body2 text-grey-7 q-mb-md")
    
    # ──────────────────────────────────────────
    # 4つのタブに整理
    # ──────────────────────────────────────────
    with ui.tabs().classes('w-full').props('align=left dense active-color=cyan indicator-color=cyan text-color=grey') as smiles_tabs:
        tab_generate = ui.tab('🔬 特徴量生成', icon='auto_fix_normal')
        tab_manage = ui.tab('📦 特徴量セット管理', icon='folder_open')
        tab_grouping = ui.tab('📦 特徴量グルーピング', icon='category')
        tab_constraints = ui.tab('📐 単調性制約', icon='science')
    
    with ui.tab_panels(smiles_tabs, value=tab_generate).classes('w-full'):
        
        # 1. 特徴量生成タブ
        with ui.tab_panel(tab_generate):
            _render_feature_generation_section(state)
        
        # 2. 特徴量セット管理タブ
        with ui.tab_panel(tab_manage):
            _render_feature_set_management_section(state)
        
        # 3. 特徴量グルーピングタブ
        with ui.tab_panel(tab_grouping):
            from frontend_nicegui.components.smiles_grouping_panel import render_smiles_grouping_panel
            render_smiles_grouping_panel(state)
        
        # 4. 単調性制約タブ
        with ui.tab_panel(tab_constraints):
            _render_constraint_settings_section(state)

    # リフレッシュ関数を登録しておく（他から呼ばれる可能性があるため）
    def _refresh_all():
        smiles_tabs.set_value(tab_generate)
    state["_refresh_smiles_integrated"] = _refresh_all


# ============================================================
# ヘルパーセクション
# ============================================================

def _render_feature_generation_section(state: dict):
    """特徴量生成セクション"""
    try:
        from frontend_nicegui.components.smiles_feature_panel import render_smiles_feature_panel
        render_smiles_feature_panel(state)
    except Exception as e:
        logger.error(f"Error in feature generation section: {e}")
        ui.label(f"⚠️ エラー: {e}")

def _render_feature_set_management_section(state: dict):
    """特徴量セット管理セクション"""
    try:
        from frontend_nicegui.components.feature_set_manager import render_feature_set_manager
        render_feature_set_manager(state)
    except Exception as e:
        logger.error(f"Error in feature set management section: {e}")
        ui.label(f"⚠️ エラー: {e}")

def _render_constraint_settings_section(state: dict):
    """制約設定セクション"""
    try:
        from frontend_nicegui.components.constraint_panel import render_constraint_panel
        render_constraint_panel(state)
    except Exception as e:
        logger.error(f"Error in constraint settings section: {e}")
        ui.label(f"⚠️ エラー: {e}")
