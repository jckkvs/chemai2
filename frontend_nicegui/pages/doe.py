from nicegui import ui
from frontend_nicegui.utils import app_state

def page_doe() -> None:
    """DOE page - simple experiment planning"""

    if not app_state.data_loaded:
        with ui.card().classes('w-full max-w-3xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):
            ui.label('データがありません').classes('text-2xl font-bold text-white mb-4')
            ui.button(
                'データを入れる →',
                icon='upload',
                on_click=lambda: ui.navigate.to('/#data_upload')
            ).props('color=primary size=lg')
        return

    with ui.card().classes('w-full max-w-3xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):

        ui.label('実験を計画する').classes('text-2xl font-bold text-white mb-2')
        ui.label(f'現在 {len(app_state.data_df)}サンプル。新しい条件を提案します。').classes('text-gray-300 mb-6')

        # Simple input
        n_experiments = ui.number(
            '追加する実験の数',
            value=10,
            min=5,
            max=50
        ).classes('w-full mb-4')

        ui.button(
            '条件を提案させる',
            icon='science',
            on_click=lambda: ui.notify('提案を生成中...', type='info')
        ).props('color=primary size=lg').classes('w-full')

        # Back to decision
        ui.separator().classes('bg-gray-700 my-6')
        ui.button(
            '戻る：次はどう動く？',
            icon='arrow_back',
            on_click=lambda: ui.navigate.to('/#decision')
        ).props('flat color=gray').classes('w-full')
