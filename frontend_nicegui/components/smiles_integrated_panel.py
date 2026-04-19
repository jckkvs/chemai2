"""
SMILES特徴量・制約設定 統合パネル (Blueprint準拠版)
"""
from __future__ import annotations
import logging
from typing import Dict, List, Any
from nicegui import ui

from backend.chem.smiles_feature_calculator import SMILESFeatureCalculator
from backend.data.feature_merger import FeatureMerger, MergedDataResult
from backend.models.monotonic_constraints import UnifiedConstraintManager
from frontend_nicegui.components.feature_set_selector import FeatureSetSelector
from frontend_nicegui.components.unified_constraints_panel import UnifiedConstraintsPanel

logger = logging.getLogger(__name__)

def render_smiles_integrated_panel(state: dict[str, Any]):
    """
    SMILES特徴量・制約設定パネル (Blueprint準拠版)
    """
    
    ui.label("🚀 SMILES特徴量・制約統合設定").classes("text-2xl font-bold q-mb-md")
    
    # ──────────────────────────────────────────
    # 計算 & マージ処理
    # ──────────────────────────────────────────
    async def calculate_features(descriptor_set):
        if not descriptor_set:
            return
        
        smiles_col = state.get("smiles_col")
        if not smiles_col:
            ui.notify("SMILES列が選択されていません。データタブで設定してください。", type="warning")
            return
            
        df = state.get("df")
        if df is None:
            ui.notify("データがロードされていません。", type="negative")
            return

        with ui.dialog().props('persistent') as dialog, ui.card().classes('q-pa-lg items-center'):
            ui.spinner(size='lg')
            ui.label(f'"{descriptor_set.name}" の特徴量を計算中...')
            dialog.open()

            try:
                # 計算 & マージ
                calculator = SMILESFeatureCalculator()
                merger = FeatureMerger(calculator)
                
                # 計算実行 (非同期で回したいが calculator.calculate は同期なので run_in_executor )
                import asyncio
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: merger.merge(
                        df=df,
                        smiles_column=smiles_col,
                        descriptor_set=descriptor_set,
                        target_column=state.get("target_col"),
                        group_columns=[state.get("group_col")] if state.get("group_col") else None
                    )
                )
                
                # 状態に保存
                state['merged_result'] = result
                # 制約マネージャーがなければ作成
                if 'constraint_manager' not in state or state['constraint_manager'] is None:
                    state['constraint_manager'] = UnifiedConstraintManager()
                
                # precalc_df (既存EDA互換) を更新
                state["precalc_df"] = result.features  # TODO: 既存との列名衝突回避確認

                ui.notify(f"計算完了: 計 {result.n_features} 個の特徴量を統合しました", type="positive")
                
                # タブを自動で「制約設定」へ
                smiles_tabs.set_value(tab_constraints)
                
            except Exception as e:
                logger.error(f"SMILES calculation error: {e}")
                ui.notify(f"エラーが発生しました: {str(e)}", type="negative")
            finally:
                dialog.close()

    # ──────────────────────────────────────────
    # タブ構成
    # ──────────────────────────────────────────
    with ui.tabs().classes('w-full').props('align=left dense active-color=primary indicator-color=primary text-color=grey') as smiles_tabs:
        tab_manage = ui.tab('📦 特徴量セット計算', icon='science')
        tab_constraints = ui.tab('📐 単調性制約', icon='rule')
        tab_eda = ui.tab('📊 セット比較EDA', icon='query_stats')
    
    with ui.tab_panels(smiles_tabs, value=tab_manage).classes('w-full bg-transparent'):
        
        # 1. 特徴量セット計算/選択
        with ui.tab_panel(tab_manage):
            selector = FeatureSetSelector(
                smiles_column=state.get("smiles_col"),
                on_calculate=lambda: calculate_features(selector.get_selected_set())
            )
            selector.render(ui.column().classes('w-full'))
        
        # 2. 統合制約設定
        with ui.tab_panel(tab_constraints):
            merged_res: MergedDataResult = state.get('merged_result')
            if not merged_res:
                ui.label("特徴量を計算すると、ここに制約設定パネルが表示されます。").classes("text-grey italic q-pa-lg")
            else:
                panel = UnifiedConstraintsPanel(
                    feature_columns=merged_res.all_numeric_columns,
                    feature_set_info=merged_res.feature_set_info,
                    on_constraints_updated=lambda c: logger.info(f"Constraints updated: {len(c)}")
                )
                # 既存のmanagerを同期
                panel.constraint_manager = state['constraint_manager']
                panel.render(ui.column().classes('w-full'))
            
        # 3. 特徴量セット比較EDA
        with ui.tab_panel(tab_eda):
            merged_res: MergedDataResult = state.get('merged_result')
            if not merged_res:
                ui.label("特徴量を計算すると、セット比較分析が表示されます。").classes("text-grey italic q-pa-lg")
            else:
                from backend.data.eda import plot_feature_set_comparison
                with ui.row().classes('w-full'):
                    fig_pca = plot_feature_set_comparison(merged_res, plot_type='pca')
                    if fig_pca:
                        ui.plotly(fig_pca).classes('col')
                    
                    fig_corr = plot_feature_set_comparison(merged_res, plot_type='correlation')
                    if fig_corr:
                        ui.plotly(fig_corr).classes('col')

    # リフレッシュ関数
    def _refresh_all():
        smiles_tabs.set_value(tab_manage)
    state["_refresh_smiles_integrated"] = _refresh_all
