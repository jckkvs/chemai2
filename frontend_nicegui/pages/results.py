from nicegui import ui
from frontend_nicegui.utils import app_state

def page_results() -> None:
    """Results page - simple"""

    if not app_state.data_loaded:
        with ui.card().classes('w-full max-w-3xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):
            ui.label('結果がありません').classes('text-2xl font-bold text-white mb-4')
            ui.button(
                'データを入れる →',
                icon='upload',
                on_click=lambda: ui.navigate.to('/#data_upload')
            ).props('color=primary size=lg')
        return

    with ui.card().classes('w-full max-w-3xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):

        ui.label('結果').classes('text-2xl font-bold text-white mb-6')

        # Simple result display
        with ui.card().classes('w-full bg-gray-800 p-6').style('border: 1px solid #374151;'):
            ui.label('提案された条件').classes('text-lg font-bold text-white mb-4')
            ui.label('AIが提案する条件がここに表示されます...').classes('text-gray-500 italic')

        # Action
        ui.separator().classes('bg-gray-700 my-6')
        ui.label('条件を確認したら').classes('text-lg text-gray-300 mb-4')
        ui.button(
            '新しいデータとして追加する',
            icon='add',
            on_click=lambda: ui.notify('機能は開発中です', type='info')
        ).props('color=primary size=lg').classes('w-full')
