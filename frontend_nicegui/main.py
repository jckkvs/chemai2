"""
frontend_nicegui/main.py
タブ切り替えロジック修正版
"""
from nicegui import ui, app
import pandas as pd
from typing import Optional, Dict
import logging

# 設定
from backend.config.llm_settings import LLMConfig

# UIコンポーネント
from frontend_nicegui.components.llm_config_dialog import LLMConfigDialog

# ページ
from frontend_nicegui.pages.data_upload_tab import DataUploadPage
from frontend_nicegui.pages.llm_assistant_tab import LLMAssistantPage
from frontend_nicegui.pages.automl_page import AutoMLPage

logger = logging.getLogger(__name__)

# ============================================================================
#  グローバル状態
# ============================================================================
llm_config = LLMConfig.load().get_effective_config()
llm_dialog = LLMConfigDialog(config=llm_config, on_config_change=lambda c: c.save())

# ページインスタンス
data_page = DataUploadPage(llm_config=llm_config)
assistant_page = LLMAssistantPage(llm_config=llm_config, llm_dialog=llm_dialog)
automl_page = AutoMLPage()

# タブ制御用変数
active_tab_container = {'value': None}
tabs_container = {'value': None}

# アプリケーション状態
app.storage.general['uploaded_data'] = None


def navigate_to_automl(data: pd.DataFrame):
    """AutoMLページへ遷移する関数"""
    if data is not None:
        automl_page.load_data(data)
    
    tabs = tabs_container['value']
    if tabs is not None:
        # AutoMLタブを検索して切り替え
        for tab in tabs._tabs.values():
            if 'AutoML' in tab.text:
                tabs.value = tab
                break


# ============================================================================
#  UIレイアウト
# ============================================================================
@ui.page('/')
def main_page():
    """メインページ"""
    
    with ui.header().classes('bg-white shadow-sm'):
        with ui.row().classes('w-full justify-between items-center px-4'):
            ui.label('ChemAI MI Studio').classes('text-xl font-bold text-primary')
            with ui.row().classes('gap-2'):
                ui.label('v1.0.0').classes('text-xs text-gray-500')
                llm_dialog.create_trigger_button(ui.row(), label='⚙️', icon='settings')
    
    # タブの作成
    with ui.tabs().classes('w-full bg-white shadow-sm') as t:
        tabs_container['value'] = t
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


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='ChemAI MI Studio',
        host='0.0.0.0',
        port=8080,
        reload=False,
        storage_secret='chemai2_secret_key_2026'
    )
