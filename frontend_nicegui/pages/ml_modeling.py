from nicegui import ui
from frontend_nicegui.utils import app_state


def page_ml_modeling() -> None:
    """ML Modeling page - simplified as a means to decision support"""
    if not app_state.data_loaded:
        with ui.card().classes('w-full max-w-7xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):
            ui.label('🤖 MLモデリング').classes('text-3xl font-bold text-white mb-4')
            ui.separator().classes('bg-gray-700 mb-4')
            ui.label('先にデータをアップロードしてください。').classes('text-gray-400')
            ui.button('データアップロードへ →', icon='upload',
                      on_click=lambda: ui.navigate.to('/#data_upload')).props('color=primary')
        return

    with ui.card().classes('w-full max-w-7xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):
        ui.label('🤖 MLモデリング').classes('text-3xl font-bold text-white mb-2')
        ui.label('意思決定のための予測モデルを構築します。').classes('text-gray-300 mb-6')
        ui.separator().classes('bg-gray-700 mb-6')

        if app_state.target_column is None:
            ui.label('先にEDAで目的変数を選択してください。').classes('text-yellow-400')
            ui.button('EDAへ →', icon='arrow_forward',
                      on_click=lambda: ui.navigate.to('/#eda')).props('color=primary')
            return

        # Simple model training
        with ui.card().classes('w-full bg-gray-800 p-6').style('border: 1px solid #374151;'):
            ui.label('モデル学習').classes('text-xl font-bold text-white mb-4')

            ui.label(f'目的変数: {app_state.target_column}').classes('text-blue-300 mb-2')
            ui.label(f'サンプル数: {len(app_state.data_df)}').classes('text-gray-400 mb-4')

            ui.button(
                'モデルを学習する',
                icon='school',
                on_click=lambda: ui.notify('モデル学習機能は開発中です', type='info')
            ).props('color=primary size=lg').classes('w-full')

        # Results (placeholder)
        ui.separator().classes('bg-gray-700 my-6')
        with ui.card().classes('w-full bg-gray-800 p-6').style('border: 1px solid #374151;'):
            ui.label('学習結果').classes('text-xl font-bold text-white mb-4')
            ui.label('モデル学習後に結果が表示されます...').classes('text-gray-500 italic')

        # Navigation
        ui.separator().classes('bg-gray-700 mt-8')
        with ui.row().classes('w-full justify-between'):
            ui.button('← 前へ: 意思決定支援', icon='arrow_back',
                      on_click=lambda: ui.navigate.to('/#decision')).props('flat color=gray')
            ui.button('次へ: 実験計画法 →', icon='arrow_forward',
                      on_click=lambda: ui.navigate.to('/#doe')).props('flat color=gray')
