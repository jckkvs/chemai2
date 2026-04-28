"""
frontend_nicegui/main.py
完全統合版 - すべての機能を連携
"""
from nicegui import ui, app
import pandas as pd
from typing import Optional, Dict
import logging

# 設定
from backend.config.llm_settings import LLMConfig

# データ処理
from backend.data.file_uploader import read_csv_smart, read_excel_smart, assess_data_quality
from backend.data.data_cleaner import DataCleanerLLM

# LLM
from backend.llm.prompt_templates import create_external_prompt

# UIコンポーネント
from frontend_nicegui.components.file_upload_zone import FileUploadZone
from frontend_nicegui.components.llm_config_dialog import LLMConfigDialog

# ページ
from frontend_nicegui.pages.data_upload_tab import DataUploadPage
from frontend_nicegui.pages.llm_assistant_tab import LLMAssistantPage
from frontend_nicegui.pages.automl_page import AutoMLPage

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
#  グローバル状態 (遅延初期化用)
# ============================================================================
_llm_config = None
_llm_dialog = None
_data_page = None
_assistant_page = None
_automl_page = None

def get_ui_components():
    global _llm_config, _llm_dialog, _data_page, _assistant_page, _automl_page
    if _llm_config is None:
        _llm_config = LLMConfig.load().get_effective_config()
        _llm_dialog = LLMConfigDialog(config=_llm_config, on_config_change=lambda c: c.save())
        _data_page = DataUploadPage(llm_config=_llm_config)
        _assistant_page = LLMAssistantPage(llm_config=_llm_config, llm_dialog=_llm_dialog)
        _automl_page = AutoMLPage()
    return _llm_config, _llm_dialog, _data_page, _assistant_page, _automl_page

# アプリケーション状態
app.storage.general['uploaded_data'] = None
app.storage.general['quality_report'] = None
app.storage.general['current_page'] = 'data'


# ============================================================================
#  UIレイアウト
# ============================================================================
@ui.page('/')
def main_page():
    """メインページ"""
    llm_config, llm_dialog, data_page, assistant_page, automl_page = get_ui_components()
    
    # ヘッダー
    with ui.header().classes('bg-white shadow-sm'):
        with ui.row().classes('w-full justify-between items-center px-4 py-2'):
            with ui.row().classes('items-center'):
                ui.icon('psychology', size='2em', color='primary')
                ui.label('ChemAI MI Studio').classes('text-xl font-bold text-primary ml-2')
            
            with ui.row().classes('gap-4 items-center'):
                ui.label('v1.0.0').classes('text-xs text-gray-500')
                ui.button(icon='settings', on_click=llm_dialog.open).props('flat round color=primary')
    
    # ナビゲーションタブ
    with ui.tabs().classes('w-full bg-white shadow-sm') as tabs:
        data_tab = ui.tab('Data Upload & Cleaning', icon='cloud_upload')
        llm_tab = ui.tab('LLM Assistant', icon='auto_awesome')
        auto_ml_tab = ui.tab('AutoML', icon='model_training')
        viz_tab = ui.tab('Visualization (Future)', icon='insights')
    
    # タブパネル
    with ui.tab_panels(tabs, value=data_tab).classes('w-full max-w-7xl mx-auto mt-4'):
        with ui.tab_panel(data_tab):
            data_page.render()
        
        with ui.tab_panel(llm_tab):
            assistant_page.render()
        
        with ui.tab_panel(auto_ml_tab):
            automl_page.render()
        
        with ui.tab_panel(viz_tab):
            with ui.card().classes('w-full p-8 text-center'):
                ui.icon('bar_chart', size='4em', color='gray')
                ui.label('可視化機能は開発中です').classes('text-xl text-gray-500 mt-4')
    
    # フッター
    with ui.footer().classes('bg-gray-100 text-gray-600 border-t'):
        with ui.row().classes('w-full justify-center py-2'):
            ui.label('© 2026 ChemAI MI Studio - Materials Informatics Platform').classes('text-xs')


# ============================================================================
#  データ連携関数
# ============================================================================
def navigate_to_automl(data: pd.DataFrame):
    """AutoMLページへ遷移"""
    _, _, _, _, automl_page = get_ui_components()
    automl_page.load_data(data)
    ui.notify('AutoMLページへ移動しました', type='info')


# ============================================================================
#  エラーハンドリング
# ============================================================================
@ui.page('/error')
def error_page():
    """エラーページ"""
    with ui.column().classes('w-full items-center p-8'):
        ui.icon('error', size='4em', color='red')
        ui.label('エラーが発生しました').classes('text-xl font-bold text-red-600 mt-4')
        ui.button('ホームに戻る', on_click=lambda: ui.navigate.to('/')).classes('mt-4')


# ============================================================================
#  アプリケーション起動
# ============================================================================
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='ChemAI MI Studio',
        host='0.0.0.0',
        port=8085,
        reload=True,
        storage_secret='chemai2_secret_key_2026_final'
    )
