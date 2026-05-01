from nicegui import ui
from frontend_nicegui.utils import app_state
from frontend_nicegui.components.domain_knowledge_input import domain_knowledge_button, show_domain_knowledge_dialog
from frontend_nicegui.components.test_trend import render_test_trend



# Provider mapping
PROVIDER_MAP = {
    'ローカルLLM (BONSAI)': 'local',
    'OpenAI API': 'openai',
    'Anthropic API': 'anthropic',
    'Google API': 'google',
    'OpenRouter API': 'openrouter',
}
PROVIDER_DISPLAY = list(PROVIDER_MAP.keys())

def _get_current_provider_display():
    current = app_state.llm_config.get('provider', 'local')
    for k, v in PROVIDER_MAP.items():
        if v == current:
            return k
    return 'ローカルLLM (BONSAI)'

def show_llm_settings() -> None:
    """Show LLM configuration"""
    with ui.card().classes('w-full bg-gray-800 p-6').style('border: 1px solid #374151;'):
        ui.label('💬 LLM設定').classes('text-xl font-bold text-white mb-4')

        # Provider selection
        provider_select = ui.select(
            PROVIDER_DISPLAY,
            label='LLMプロバイダー',
            value=_get_current_provider_display()
        ).classes('w-full')

        # Model input
        model_input = ui.input(
            label='モデル名',
            value=app_state.llm_config.get('model', '')
        ).classes('w-full')

        ui.separator().classes('bg-gray-700 my-4')

        # API Key
        api_key_input = ui.input(
            label='API Key',
            password=True,
            value=app_state.llm_config.get('api_key', '')
        ).classes('w-full')

        # API Base (visible for OpenRouter etc.)
        api_base_input = ui.input(
            label='API Base URL（OpenRouter等の場合）',
            value=app_state.llm_config.get('api_base', '')
        ).classes('w-full')

        ui.label('ローカルLLM使用時はAPI Key不要').classes('text-xs text-gray-500')

        # Save button
        def save_settings():
            # Save provider
            selected = provider_select.value
            app_state.llm_config['provider'] = PROVIDER_MAP[selected]
            # Auto-set OpenRouter API base
            if PROVIDER_MAP[selected] == 'openrouter':
                app_state.llm_config['api_base'] = 'https://openrouter.ai/api/v1'
                api_base_input.value = 'https://openrouter.ai/api/v1'
            # Save model
            app_state.llm_config['model'] = model_input.value
            # Save API key
            app_state.llm_config['api_key'] = api_key_input.value
            # Save API base if manually set
            if api_base_input.value:
                app_state.llm_config['api_base'] = api_base_input.value
            ui.notify('設定を保存しました', type='positive')

        ui.button('保存', icon='save', on_click=save_settings).props('color=primary')


def show_display_settings() -> None:
    """Show display configuration"""
    from frontend_nicegui.utils import app_state

    with ui.card().classes('w-full bg-gray-800 p-6').style('border: 1px solid #374151;'):
        ui.label('⚙️ 表示・計算設定').classes('text-xl font-bold text-white mb-4')

        ui.checkbox('ダークモード（推奨）', value=True).classes('text-white')
        ui.checkbox('LLMストリーミング表示', value=True).classes('text-white')

        ui.separator().classes('bg-gray-700 my-4')

        # Beginner mode toggle
        ui.label('🎓 初心者モード').classes('text-lg font-bold text-white mb-2')
        ui.label('LLMが各ステップを自動で誘導します。データアップロード後は自動的に「データを眺める」ページに移動します。').classes('text-xs text-gray-400 mb-3')

        def on_beginner_mode_change(e):
            app_state.beginner_mode = e.value
            if e.value:
                ui.notify('初心者モードを有効にしました。LLMがステップを誘導します。', type='positive')
            else:
                ui.notify('初心者モードを無効にしました。', type='info')

        ui.checkbox(
            '初心者モードを有効にする',
            value=app_state.beginner_mode
        ).classes('text-white').on_value_change(on_beginner_mode_change)

        ui.separator().classes('bg-gray-700 my-4')

        ui.select(
            ['自動検出', 'CPUのみ', 'GPU (CUDA)'],
            label='計算デバイス',
            value='自動検出'
        ).classes('w-full')


def show_domain_knowledge() -> None:
    """Show domain knowledge management"""
    from frontend_nicegui.utils.domain_knowledge import domain_knowledge

    with ui.card().classes('w-full bg-gray-800 p-6').style('border: 1px solid #374151;'):
        ui.label('🧠 ドメイン知識管理').classes('text-xl font-bold text-white mb-4')

        ui.label('保存されたドメイン知識').classes('text-lg font-bold text-blue-300 mb-2')

        # 実際のドメイン知識を表示
        items = domain_knowledge.get_all()
        if items:
            for item in items:
                with ui.row().classes('w-full items-start gap-2 p-2'):
                    ui.icon('psychology', color='blue')
                    with ui.column().classes('gap-0 flex-1'):
                        ui.label(f"[{item['type']}]").classes('text-xs text-gray-400')
                        ui.label(item['content']).classes('text-sm text-white')
                        if item['context']:
                            ui.label(f"Context: {item['context']}").classes('text-xs text-gray-500')
        else:
            ui.label('まだドメイン知識が保存されていません').classes('text-gray-500 italic')

        # Add new knowledge button
        ui.button('新しい知識を追加', icon='add',
                on_click=lambda: show_domain_knowledge_dialog('settings')).props('color=primary')

        # Clear all button
        if items:
            def clear_knowledge():
                domain_knowledge.clear()
                ui.notify('全ての知識を削除しました', type='positive')
                app_state.navigate_to('settings')

            ui.button('全て削除', icon='delete',
                    on_click=clear_knowledge).props('flat color=negative')


def page_settings() -> None:
    """Settings page"""
    with ui.card().classes('w-full max-w-7xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):
        # Header
        ui.label('⚙️ 設定').classes('text-3xl font-bold text-white mb-2')
        ui.label('アプリケーションの設定とドメイン知識の管理').classes('text-gray-300 mb-6')
        ui.separator().classes('bg-gray-700 mb-6')

        # Settings tabs
        with ui.tabs().classes('w-full') as tabs:
            ui.tab('llm', '💬 LLM設定')
            ui.tab('display', '⚙️ 表示・計算')
            ui.tab('knowledge', '🧠 ドメイン知識')
            ui.tab('test_trend', '📊 テスト結果')

        with ui.tab_panels(tabs, value='llm').classes('w-full'):
            with ui.tab_panel('llm'):
                show_llm_settings()

            with ui.tab_panel('display'):
                show_display_settings()

            with ui.tab_panel('knowledge'):
                show_domain_knowledge()

            with ui.tab_panel('test_trend'):
                render_test_trend()

        # Navigation
        ui.separator().classes('bg-gray-700 mt-8')
        with ui.row().classes('w-full justify-center'):
            ui.button('🏠 ウェルカムへ戻る', icon='home',
                      on_click=lambda: app_state.navigate_to('welcome')).props('flat color=gray')
