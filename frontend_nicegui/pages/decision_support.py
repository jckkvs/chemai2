from nicegui import ui
from frontend_nicegui.utils import app_state
from frontend_nicegui.components.domain_knowledge_input import domain_knowledge_button

def page_decision_support() -> None:
    """Decision Support page - just what user needs to know"""

    # Data not ready
    if not app_state.data_loaded or app_state.target_column is None:
        with ui.card().classes('w-full max-w-3xl mx-auto p-8 relative').style('background-color: #111827; color: #F9FAFB; border: none;'):
            domain_knowledge_button('decision')
            ui.label('準備ができていません').classes('text-2xl font-bold text-white mb-4')
            ui.label('先にデータを入れて、予測したい物性を選んでください').classes('text-gray-300 mb-6')
            ui.button(
                'データを入れる →',
                icon='upload',
                on_click=lambda: ui.navigate.to('/#data_upload')
            ).props('color=primary size=lg')
        return

    # Data ready - show simple status and next action
    with ui.card().classes('w-full max-w-3xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):

        # Progress indicator (user feedback: show overall progress)
        current_step = 4  # Decision Support is step 4
        total_steps = 5
        progress = int((current_step - 1) / total_steps * 100)
        ui.label('【現在のステップ 4/5】').classes('text-sm text-gray-400 mb-2')
        with ui.row().classes('w-full items-center gap-2 mb-4'):
            ui.label('❶ データを入れる').classes('text-xs text-green-400')
            ui.label('→').classes('text-gray-500')
            ui.label('❷ LLM相談').classes('text-xs text-green-400')
            ui.label('→').classes('text-gray-500')
            ui.label('❸ データを眺める').classes('text-xs text-green-400')
            ui.label('→').classes('text-gray-500')
            ui.label('❹ 次のアクション提案').classes('text-xs text-blue-300 font-bold')
            ui.label('→').classes('text-gray-500')
            ui.label('❺ 結果を確認').classes('text-xs text-gray-500')

        df = app_state.data_df
        n_samples = len(df)

        # Goal display (from section 2.3 of improvement spec)
        if app_state.target_column:
            with ui.card().classes('w-full bg-blue-900 p-4 mb-6').style('border: 1px solid #1E40AF;'):
                ui.label(f'目標：{app_state.target_column} を予測').classes('text-xl font-bold text-white')
                # Mock achievement probability (will be calculated by ML later)
                if n_samples >= 50:
                    ui.label('現在の達成確率：約70%（デモデータ）').classes('text-sm text-blue-300')
                else:
                    ui.label('達成確率を計算するには、もっとデータが必要です').classes('text-sm text-yellow-300')

        # Simple status
        ui.label('現在の状況').classes('text-2xl font-bold text-white mb-4')

        with ui.card().classes('w-full bg-gray-800 p-4').style('border: 1px solid #374151;'):
            ui.label(f'データ: {n_samples}サンプル').classes('text-lg text-gray-300')
            ui.label(f'予測対象: {app_state.target_column}').classes('text-lg text-blue-300')

        ui.separator().classes('bg-gray-700 my-6')

        # What to do next - the only thing user needs
        # User feedback: show "まずはデータを増やしましょう" first when achievement probability is low
        if n_samples < 50:
            ui.label('まずは：データを増やしましょう').classes('text-xl font-bold text-yellow-300 mb-4').tooltip('達成確率が低い場合、まずはデータを増やすことが最優先です')
            ui.label(f'現在 {n_samples}サンプル。もう少し増やすと良い結果が出やすくなります').classes('text-gray-300 mb-4')
            ui.button(
                'データを追加する →',
                icon='upload',
                on_click=lambda: ui.navigate.to('/#data_upload')
            ).props('color=amber size=lg').classes('w-full')
        else:
            ui.label('次は：AIに提案させよう').classes('text-xl font-bold text-green-300 mb-4')
            ui.label('これだけのデータがあれば、目標を達成する条件を提案できます').classes('text-gray-300 mb-4')
            ui.button(
                'AIに提案させる →',
                icon='psychology',
                on_click=lambda: ui.notify('提案を生成中...', type='info')
            ).props('color=green size=lg').classes('w-full')

        # Test result trend display (user feedback: show progress trend)
        ui.separator().classes('bg-gray-700 my-6')
        ui.label('テスト結果の推移').classes('text-lg font-bold text-white mb-4').tooltip('周期ごとのテスト結果の推移（進捗状況）')
        with ui.card().classes('w-full bg-gray-800 p-4').style('border: 1px solid #374151;'):
            ui.label('周期ごとのテスト結果（成功数）').classes('text-sm text-gray-300 mb-2')
            # Simple text-based trend
            trend_text = """
            | 周期 | 成功 | 失敗 | スキップ | エラー |
            |---|---|---|---|---|
            | 第4周期・1回 | 62 | 54 | 90 | 116 |
            | 第6周期・2回 | 62 | 54 | 90 | 116 |
            | 第7周期・3回 | 62 | 54 | 90 | 116 |
            """
            ui.markdown(trend_text).classes('text-xs text-gray-400')
            ui.label('⚠️ テスト結果に変化なし。エラー116件の修正が必要です。').classes('text-xs text-yellow-400 mt-2')
