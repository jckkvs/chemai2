"""
frontend_nicegui/pages/llm_assistant_tab.py
LLMアシスタント - プロンプトのみ生成 / ローカルLLM / 外部API の3モード
"""
from nicegui import ui
from typing import Optional, Dict, List
import logging
import asyncio
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class LLMAssistantPage:
    """LLMアシスタントページ - チャット + プロンプト生成 + レポート生成

    動作モード：
    - prompt_only: プロンプトのみ生成（セキュア推奨、外部LLMにコピペ用）
    - local: ローカルLLMを使用（bonsai-8b等）
    - api: 外部APIを使用（OpenAI等）
    """

    def __init__(self, llm_config, llm_dialog, state=None):
        self.llm_config = llm_config
        self.llm_dialog = llm_dialog
        self.chat_history: List[Dict[str, str]] = []
        self.state = state  # Optional state for report generation

        # ウィジェット参照
        self._mode_select: Optional[ui.select] = None
        self._prompt_only_panel: Optional[ui.card] = None
        self._local_panel: Optional[ui.card] = None
        self._api_panel: Optional[ui.card] = None

    def render(self):
        """ページを描画"""
        with ui.column().classes('w-full max-w-5xl mx-auto p-4'):
            ui.label('✨ LLM アシスタント').classes('text-2xl font-bold mb-4')

            # モード選択
            with ui.card().classes('w-full mb-4'):
                ui.label('🔄 LLM実行モード').classes('font-bold mb-2')
                with ui.row().classes('w-full gap-4'):
                    self._mode_select = ui.select(
                        options={
                            'prompt_only': '🔒 プロンプトのみ生成（セキュア推奨・外部LLM用）',
                            'local': '💬 ローカルLLMを使用（bonsai-8b等）',
                            'api': '🌐 外部APIを使用（OpenAI等）',
                        },
                        value=self.llm_config.mode if self.llm_config else 'local'
                    ).classes('flex-1')

                    with ui.column().classes('ml-auto text-right'):
                        ui.label('LLM設定').classes('text-xs text-gray-500')
                        self.llm_dialog.create_trigger_button(ui.row(), label='⚙️', icon='settings')

            # --- プロンプトのみ生成モード ---
            self._prompt_only_panel = ui.card().classes('w-full mb-4')
            with self._prompt_only_panel:
                ui.label('🔒 プロンプトのみ生成モード（セキュア）').classes('font-bold')
                ui.label(
                    'LLMを実際には呼び出さず、外部LLM（ChatGPT, Claude等）に投げる用のプロンプトを生成します。'
                    'セキュリティを重視する場合に推奨します。'
                ).classes('text-sm text-gray-600 mb-2')

                # プロンプトテンプレート選択
                self._prompt_template = ui.select(
                    options={
                        'standard': '標準',
                        'chemistry': '化学特化',
                        'code': 'コード生成',
                    },
                    label='プロンプトテンプレート',
                    value='standard'
                ).classes('w-full').props('dense filled dark')

                # ユーザー入力
                self._prompt_user_input = ui.textarea(
                    label='プロンプトの指示を入力',
                    placeholder='例：溶媒の誘電率を予測する機械学習モデルを設計したい...'
                ).classes('w-full').props('outlined autogrow')

                self._prompt_result_area = ui.textarea(
                    label='生成されたプロンプト'
                ).props('readonly outlined autogrow').classes('w-full h-48 font-mono text-sm')

                with ui.row():
                    ui.button('📋 プロンプト生成', on_click=self._generate_prompt_only, color='primary')
                    ui.button('📋 コピー', on_click=self._copy_prompt).props('outline')

            # --- ローカルLLMチャット用パネル ---
            self._local_panel = ui.card().classes('w-full mb-4')
            self._local_panel.visible = False

            with self._local_panel:
                ui.label('💬 ローカルLLMチャット').classes('font-bold')
                ui.label('ローカルLLM（bonsai-8b等）と対話').classes('text-sm text-gray-600 mb-2')

                # チャット履歴表示
                self._chat_display = ui.column().classes('w-full p-4 bg-gray-50 rounded h-80 overflow-y-auto border')
                self._update_chat_display()

                # 入力エリア
                with ui.row().classes('w-full gap-2 mt-2'):
                    self._chat_input = ui.input(placeholder='質問を入力...').classes('flex-1')
                    self._send_btn = ui.button('送信', on_click=self._send_message_local, color='primary')
                    self._clear_btn = ui.button('クリア', on_click=self._clear_chat).props('outline')

                # ステータス
                self._status_label = ui.label('').classes('text-xs text-gray-500 mt-2')

            # --- 外部APIチャット用パネル ---
            self._api_panel = ui.card().classes('w-full mb-4')
            self._api_panel.visible = False

            with self._api_panel:
                ui.label('🌐 外部APIチャット').classes('font-bold')
                ui.label('OpenAI API等の外部LLMと対話').classes('text-sm text-gray-600 mb-2')

                # API設定表示
                with ui.row().classes('w-full gap-2'):
                    self._api_model_label = ui.label('').classes('text-xs text-gray-500')
                    ui.button('設定変更', on_click=self.llm_dialog.open).props('flat dense')

                self._api_chat_display = ui.column().classes('w-full p-4 bg-gray-50 rounded h-80 overflow-y-auto border')

                with ui.row().classes('w-full gap-2 mt-2'):
                    self._api_chat_input = ui.input(placeholder='質問を入力...').classes('flex-1')
                    ui.button('送信', on_click=self._send_message_api, color='primary')

            # モード変更イベント
            self._mode_select.on_value_change(self._on_mode_changed)

            # 初期表示設定
            self._update_panel_visibility()

    def _update_panel_visibility(self):
        """現在のモードに応じてパネル表示を切り替え"""
        mode = self._mode_select.value if self._mode_select else 'local'
        if self._prompt_only_panel:
            self._prompt_only_panel.visible = (mode == 'prompt_only')
        if self._local_panel:
            self._local_panel.visible = (mode == 'local')
        if self._api_panel:
            self._api_panel.visible = (mode == 'api')

    def _on_mode_changed(self, event):
        """LLM実行モード変更"""
        self._update_panel_visibility()

        mode = self._mode_select.value
        if mode == 'prompt_only':
            ui.notify('🔒 プロンプトのみ生成モード（セキュア）', type='info')
        elif mode == 'local':
            ui.notify('💬 ローカルLLMモード', type='info')
        else:
            ui.notify('🌐 外部APIモード', type='info')

    async def _generate_prompt_only(self):
        """プロンプトのみ生成モード：バックエンドの generate_prompt_for_external_llm() を呼び出し"""
        user_instruction = self._prompt_user_input.value or ''
        if not user_instruction.strip():
            ui.notify('⚠️ プロンプトの指示を入力してください', type='warning')
            return

        try:
            # バックエンドのLLMManagerを呼び出し
            from backend.llm.manager import LLMManager
            manager = LLMManager()
            template = self._prompt_template.value if self._prompt_template else 'standard'

            # プロンプト生成（prompt_onlyモードではLLMを呼ばない）
            prompt = manager.generate_prompt_for_external_llm(
                user_prompt=user_instruction,
                template=template,
            )

            self._prompt_result_area.value = prompt
            ui.notify('✨ プロンプトを生成しました。コピーして外部LLMで使用してください', type='positive')

        except Exception as e:
            logger.error(f"プロンプト生成エラー: {e}", exc_info=True)
            ui.notify(f'❌ プロンプト生成エラー: {str(e)}', type='negative')

    def _copy_prompt(self):
        """プロンプトをコピー"""
        if self._prompt_result_area and self._prompt_result_area.value:
            ui.clipboard.write(self._prompt_result_area.value)
            ui.notify('✨ プロンプトをコピーしました', type='positive')
        else:
            ui.notify('⚠️ プロンプトを生成してください', type='warning')

    async def _send_message_local(self):
        """ローカルLLMでメッセージ送信"""
        message = self._chat_input.value.strip()
        if not message:
            return

        self.chat_history.append({'role': 'user', 'content': message})
        self._update_chat_display()
        self._chat_input.value = ''
        self._status_label.text = '⏳️ ローカルLLM処理中...'
        self._send_btn.enabled = False

        try:
            # バックエンドのLLMManagerを呼び出し
            from backend.llm.manager import LLMManager
            manager = LLMManager()

            # ストリーミング応答を取得
            response_text = ""
            async for token in manager.stream_chat(message, temperature=0.7, max_tokens=1024):
                response_text += token
                # 最後のメッセージを更新
                if self.chat_history and self.chat_history[-1]['role'] == 'assistant':
                    self.chat_history[-1]['content'] = response_text
                else:
                    self.chat_history.append({'role': 'assistant', 'content': response_text})
                self._update_chat_display()

            self._status_label.text = '✅ 完了'

        except Exception as e:
            logger.error(f"ローカルLLMエラー: {e}", exc_info=True)
            error_msg = f"エラー: {str(e)}"
            self.chat_history.append({'role': 'error', 'content': error_msg})
            self._update_chat_display()
            self._status_label.text = '❌ エラーが発生しました'
        finally:
            self._send_btn.enabled = True

    async def _send_message_api(self):
        """外部APIでメッセージ送信"""
        message = self._api_chat_input.value.strip()
        if not message:
            return

        # TODO: API呼び出し実装（OpenAI等）
        ui.notify('🌐 外部APIモードは開発中です', type='warning')

    def _update_chat_display(self):
        """チャット履歴を画面に表示"""
        if not hasattr(self, '_chat_display'):
            return

        self._chat_display.clear()

        if not self.chat_history:
            with self._chat_display:
                ui.label('💬 会話を始めてください').classes('text-center text-gray-400 mt-8')
            return

        for msg in self.chat_history:
            role = msg['role']
            content = msg['content']

            with self._chat_display:
                if role == 'user':
                    with ui.row().classes('w-full justify-end'):
                        with ui.card().classes('bg-blue-100 max-w-sm'):
                            ui.label(content).classes('text-sm')

                elif role == 'assistant':
                    with ui.row().classes('w-full justify-start'):
                        with ui.card().classes('bg-gray-100 max-w-sm'):
                            ui.markdown(content).classes('text-sm')

                elif role == 'error':
                    with ui.row().classes('w-full justify-start'):
                        with ui.card().classes('bg-red-100'):
                            ui.label(f'❌ {content}').classes('text-sm text-red-600')

        self._chat_display.scroll_to('bottom')

    def _clear_chat(self):
        """チャット履歴をクリア"""
        self.chat_history.clear()
        self._update_chat_display()
        if hasattr(self, '_status_label'):
            self._status_label.text = ''
        ui.notify('✅ チャット履歴をクリアしました', type='positive')
