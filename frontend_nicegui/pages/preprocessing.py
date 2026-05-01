from nicegui import ui
from frontend_nicegui.utils import app_state
from frontend_nicegui.components.domain_knowledge_input import domain_knowledge_button


def page_preprocessing() -> None:
    """Preprocessing and feature selection page - simplified"""
    if not app_state.data_loaded:
        with ui.card().classes('w-full max-w-7xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):
            ui.label('🔧 前処理・特徴量').classes('text-3xl font-bold text-white mb-4')
            ui.separator().classes('bg-gray-700 mb-4')
            ui.label('先にデータをアップロードしてください。').classes('text-gray-400')
            ui.button('データアップロードへ →', icon='upload',
                      on_click=lambda: ui.navigate.to('/#data_upload')).props('color=primary')
        return

    with ui.card().classes('w-full max-w-7xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):
        ui.label('🔧 前処理・特徴量').classes('text-3xl font-bold text-white mb-2')
        ui.label('データの前処理と特徴量計算を行います。').classes('text-gray-300 mb-6')
        ui.separator().classes('bg-gray-700 mb-6')

        # Simple preprocessing options
        with ui.card().classes('w-full bg-gray-800 p-6').style('border: 1px solid #374151;'):
            ui.label('前処理オプション').classes('text-xl font-bold text-white mb-4')

            ui.checkbox('欠損値を補完する', value=True).classes('text-white')
            ui.checkbox('外れ値を除去する', value=False).classes('text-white')
            ui.checkbox('カテゴリ変数をエンコードする', value=True).classes('text-white')

        # Feature calculation
        ui.separator().classes('bg-gray-700 my-6')
        with ui.card().classes('w-full bg-gray-800 p-6').style('border: 1px solid #374151;'):
            ui.label('特徴量計算（SMILESがある場合）').classes('text-xl font-bold text-white mb-4')

            df = app_state.data_df
            smiles_cols = [col for col in df.columns if 'smiles' in col.lower() or 'SMILES' in col]

            if smiles_cols:
                ui.label(f'SMILES列を検出: {", ".join(smiles_cols)}').classes('text-green-400 mb-4')
                ui.button(
                    '特徴量を計算',
                    icon='calculate',
                    on_click=lambda: ui.notify('特徴量計算機能は開発中です', type='info')
                ).props('color=primary').classes('w-full')
            else:
                ui.label('SMILES列がありません。特徴量計算はスキップされます。').classes('text-gray-500')

        # Domain knowledge
        ui.separator().classes('bg-gray-700 my-6')
        with ui.card().classes('w-full bg-gray-800 p-6').style('border: 1px solid #374151;'):
            ui.label('ドメイン知識').classes('text-xl font-bold text-white mb-4')
            ui.label('材料・物質系に関する知識があれば入力してください（任意）。').classes('text-gray-400 mb-4')

            ui.textarea(label='例: 温度が上がると屈折率は下がる', placeholder='知識を入力...').classes('w-full')

        # Navigation
        ui.separator().classes('bg-gray-700 mt-8')
        with ui.row().classes('w-full justify-between'):
            ui.button('← 前へ: LLMヒアリング', icon='arrow_back',
                      on_click=lambda: ui.navigate.to('/#llm_interview')).props('flat color=gray')
            ui.button('次へ: EDA →', icon='arrow_forward',
                      on_click=lambda: ui.navigate.to('/#eda')).props('color=primary')
