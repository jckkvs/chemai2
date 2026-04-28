"""
frontend_nicegui/pages/llm_assistant_tab.py
LLMアシスタント・プロンプト生成画面
"""
from nicegui import ui
from backend.llm.prompt_templates import create_external_prompt
from backend.config.llm_settings import LLMConfig
from frontend_nicegui.components.llm_config_dialog import LLMConfigDialog

class LLMAssistantPage:
    def __init__(self, llm_config: LLMConfig, llm_dialog: LLMConfigDialog):
        self.llm_config = llm_config
        self.llm_dialog = llm_dialog
        self.prompt_area = None

    def render(self):
        with ui.column().classes('w-full p-4'):
            with ui.row().classes('w-full justify-between items-center mb-4'):
                ui.label('LLMアシスタント').classes('text-2xl font-bold')
                # 設定へのリンク（目立たない配置）
                self.llm_dialog.create_trigger_button(ui.row(), label='⚙️ 設定')

            # 外部チャット用プロンプト生成
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('items-center mb-2'):
                    ui.icon('security', color='primary')
                    ui.label('🔐 セキュア環境向け: 外部高精度LLM用プロンプト生成').classes('font-bold text-lg')
                
                ui.label('ローカルで実行できない場合や、より強力なモデル（ChatGPT, Claude等）を使いたい場合、このプロンプトをコピーして外部チャットで使用してください。').classes('text-sm text-gray-600 mb-4')
                
                self.prompt_area = ui.textarea(label='生成されたプロンプト').props('readonly outlined autogrow').classes('w-full font-mono text-xs mb-4')
                
                with ui.row().classes('gap-2'):
                    ui.button('分析方針のプロンプトを生成', on_click=lambda: self._generate_prompt("analysis_planning")).props('elevated')
                    ui.button('データクリーニングのプロンプトを生成', on_click=lambda: self._generate_prompt("data_cleaning")).props('outline')
                    ui.button('📋 コピー', on_click=self._copy_prompt).props('flat').bind_enabled_from(self.prompt_area, 'value')

            # LLMチャット（将来的な実装用プレースホルダ）
            with ui.card().classes('w-full opacity-50'):
                ui.label('🤖 インライン・チャット（開発中）').classes('font-bold mb-2')
                ui.label('ローカルLLMまたはAPI経由で直接対話が可能になります。').classes('text-xs')

    def _generate_prompt(self, step: str):
        """外部チャット用プロンプトを生成"""
        # 本来は現在のセッション情報やデータ要約を渡す
        data_summary = "現在読み込まれているデータの統計情報..." 
        
        user_goal = "化学データの物性予測と自動解析"
        user_question = "このデータから最適な分析方針を提案してください" if step == "analysis_planning" else "データのクリーニング方法を提案してください"
        
        prompt = create_external_prompt(
            user_goal=user_goal,
            data_summary=data_summary,
            user_question=user_question,
            current_step=step
        )
        self.prompt_area.value = prompt
        ui.notify('プロンプトを生成しました。コピーして外部チャットで使用してください', type='positive')

    def _copy_prompt(self):
        if self.prompt_area.value:
            ui.clipboard.write(self.prompt_area.value)
            ui.notify('プロンプトをクリップボードにコピーしました', icon='content_copy')
