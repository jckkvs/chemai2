from nicegui import ui
from frontend_nicegui.utils import app_state
from frontend_nicegui.components.domain_knowledge_input import domain_knowledge_button

def page_llm_interview() -> None:
    """LLM Interview page - just the chat"""

    # Data not ready
    if not app_state.data_loaded or app_state.target_column is None:
        with ui.card().classes('w-full max-w-3xl mx-auto p-8 relative').style('background-color: #111827; color: #F9FAFB; border: none;'):
            domain_knowledge_button('llm_interview')
            ui.label('準備ができていません').classes('text-2xl font-bold text-white mb-4')
            ui.label('先にデータを入れて、予測したい物性を選んでください').classes('text-gray-300 mb-6')
            ui.button(
                'データを入れる →',
                icon='upload',
                on_click=lambda: app_state.navigate_to('data_upload')
            ).props('color=primary size=lg')
        return

    # Ready - just show chat
    with ui.card().classes('w-full max-w-3xl mx-auto p-4').style('background-color: #111827; color: #F9FAFB; border: none;'):

        ui.label('AIと相談する').classes('text-2xl font-bold text-white mb-2')
        ui.label(f'目標: {app_state.target_column} を調整したい').classes('text-gray-300 mb-6')

        # Simple chat area
        with ui.card().classes('w-full bg-gray-800 p-0').style('border: 1px solid #374151;'):
            # Chat history (placeholder)
            with ui.scroll_area().classes('h-80 w-full p-4'):
                with ui.row().classes('w-full justify-start mb-4'):
                    with ui.card().classes('bg-blue-900 p-3 max-w-lg').style('border: 1px solid #1E40AF;'):
                        ui.label('AI').classes('text-xs text-blue-300 font-bold')
                        ui.label('どうしたいですか？').classes('text-white')

            ui.separator().classes('bg-gray-700')

            # Input
            with ui.row().classes('w-full p-4 items-center gap-2'):
                ui.input(placeholder='メッセージを入力...').classes('flex-1')
                ui.button('送信', icon='send').props('color=primary')

        # After chat, show next step
        ui.separator().classes('bg-gray-700 mt-6')
        ui.label('相談が終わったら').classes('text-lg text-gray-300 mb-4')
        ui.button(
            '結果を見る →',
            icon='arrow_forward',
            on_click=lambda: app_state.navigate_to('decision')
        ).props('color=primary size=lg').classes('w-full')
