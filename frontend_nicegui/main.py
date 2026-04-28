"""
frontend_nicegui/main.py
ChemAI ML Studio - NiceGUI Entry Point
"""
from nicegui import ui
import logging

# LLM & Data Upload Extensions
from backend.config.llm_settings import LLMConfig
from frontend_nicegui.components.llm_config_dialog import LLMConfigDialog
from frontend_nicegui.pages.data_upload_tab import DataUploadPage
from frontend_nicegui.pages.llm_assistant_tab import LLMAssistantPage

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configurations
llm_config = LLMConfig.load().get_effective_config()

# Global state / components
llm_dialog = LLMConfigDialog(config=llm_config, on_config_change=lambda c: c.save())
data_page = DataUploadPage(llm_config=llm_config)
assistant_page = LLMAssistantPage(llm_config=llm_config, llm_dialog=llm_dialog)

# UI Layout
ui.query('body').style('background-color: #f5f5f5;')

with ui.header().classes('items-center bg-primary text-white p-4'):
    ui.icon('psychology', size='2em')
    ui.label('ChemAI ML Studio').classes('text-2xl font-bold ml-2')
    ui.space()
    with ui.row().classes('items-center gap-4'):

        ui.label('Materials Informatics Autonomous Platform').classes('text-sm opacity-80')
        # Settings button in header for global access
        ui.button(icon='settings', on_click=llm_dialog.open).props('flat round color=white')

with ui.tabs().classes('w-full bg-white shadow-sm') as tabs:
    data_tab = ui.tab('Data Upload & Cleaning', icon='cloud_upload')
    llm_tab = ui.tab('LLM Assistant', icon='auto_awesome')
    # Placeholders for existing features
    auto_ml_tab = ui.tab('AutoML (Future)', icon='model_training')
    viz_tab = ui.tab('Visualization (Future)', icon='insights')

with ui.tab_panels(tabs, value=data_tab).classes('w-full max-w-5xl mx-auto mt-4'):
    with ui.tab_panel(data_tab):
        data_page.render()
        
    with ui.tab_panel(llm_tab):
        assistant_page.render()
        
    with ui.tab_panel(auto_ml_tab):
        with ui.card().classes('p-8 text-center'):
            ui.icon('construction', size='4em', color='gray')
            ui.label('既存のAutoML機能との統合を準備中です。').classes('text-xl text-gray-500 mt-4')
            ui.button('従来のUIを開く（仮）', on_click=lambda: ui.notify('実装予定')).classes('mt-4')

    with ui.tab_panel(viz_tab):
        with ui.card().classes('p-8 text-center'):
            ui.icon('bar_chart', size='4em', color='gray')
            ui.label('可視化機能は開発中です。').classes('text-xl text-gray-500 mt-4')

# Footer
with ui.footer().classes('bg-gray-100 text-gray-500 text-xs p-2 justify-center'):
    ui.label('© 2026 ChemAI Project - Materials Informatics Suite')

# Run the app
ui.run(title='ChemAI ML Studio', port=8085, dark=False, reload=True)
