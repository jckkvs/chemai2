"""
frontend_nicegui/main.py
タブ切り替えロジック修正版 - 完全版
"""
import sys
from pathlib import Path

# ============================================================================
#  インポートパスの自動設定 (重要)
# ============================================================================
# プロジェクトルート (chemai2/) を sys.path に追加
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nicegui import ui, app
import pandas as pd
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# ============================================================================
#  UIレイアウト
# ============================================================================
@ui.page('/')
def main_page():
    """メインページ"""
    
    # 設定の読み込み
    from backend.config.llm_settings import LLMConfig
    from frontend_nicegui.components.llm_config_dialog import LLMConfigDialog
    from frontend_nicegui.pages.data_upload_tab import DataUploadPage
    from frontend_nicegui.pages.llm_assistant_tab import LLMAssistantPage
    from frontend_nicegui.pages.automl_page import AutoMLPage
    
    llm_config = LLMConfig.load().get_effective_config()
    llm_dialog = LLMConfigDialog(config=llm_config, on_config_change=lambda c: c.save())

    # ページインスタンスの作成と連携
    automl_page = AutoMLPage()
    data_page = DataUploadPage(llm_config=llm_config, automl_page=automl_page)
    assistant_page = LLMAssistantPage(llm_config=llm_config, llm_dialog=llm_dialog)
    
    with ui.header().classes('bg-white shadow-sm'):
        with ui.row().classes('w-full justify-between items-center px-4'):
            ui.label('ChemAI MI Studio').classes('text-xl font-bold text-primary')
            with ui.row().classes('gap-2'):
                ui.label('v1.0.0').classes('text-xs text-gray-500')
                llm_dialog.create_trigger_button(ui.row(), label='⚙️', icon='settings')
    
    # タブの作成
    with ui.tabs().classes('w-full bg-white shadow-sm') as t:
        data_tab = ui.tab('Data Upload & Cleaning', icon='cloud_upload')
        llm_tab = ui.tab('LLM Assistant', icon='auto_awesome')
        auto_ml_tab = ui.tab('AutoML', icon='model_training')
        viz_tab = ui.tab('Visualization (Future)', icon='insights')
    
    with ui.tab_panels(t, value=data_tab).classes('w-full max-w-7xl mx-auto mt-4'):
        with ui.tab_panel(data_tab):
            data_page.render()
        with ui.tab_panel(llm_tab):
            assistant_page.render()
        with ui.tab_panel(auto_ml_tab):
            automl_page.render()
        with ui.tab_panel(viz_tab):
            with ui.card().classes('w-full'):
                ui.label('📊 可視化機能').classes('text-xl font-bold')
                ui.label('可視化機能は開発中です').classes('text-center text-gray-500 p-8')
    
    with ui.footer().classes('bg-gray-100'):
        with ui.row().classes('w-full justify-center py-2'):
            ui.label('ChemAI MI Studio - Materials Informatics Platform').classes('text-xs text-gray-600')


# ============================================================================
#  アプリケーション起動
# ============================================================================
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='ChemAI MI Studio',
        host='0.0.0.0',
        port=8080,
        reload=False,
        storage_secret='chemai2_secret_key_2026'
    )
