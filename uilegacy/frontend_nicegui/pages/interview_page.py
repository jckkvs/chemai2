"""
frontend_nicegui/pages/interview_page.py
LLM対話式ヒアリングページ - 仕様書9.7節に基づく実装
6フェーズの対話セッション（データ要約→解析目的→予測対象→実験誤差→変数性質→特徴量選択→CV手法→単調性制約→確認）
"""
from nicegui import ui
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class InterviewPage:
    """LLM対話式ヒアリングページ"""

    def __init__(self):
        self.current_phase: int = 0
        self.phase_data: Dict[str, Any] = {}
        self.conversation: List[Dict] = []
        self._phase_labels = [
            '① データ要約',
            '② 解析目的（意思決定ゴール）',
            '③ 予測対象・目標値',
            '④ 実験誤差・変数性質',
            '⑤ 特徴量選択・ドメイン知識',
            '⑥ 単調性制約・確認',
        ]

    def render(self):
        """Interviewページを描画"""
        with ui.column().classes('w-full max-w-7xl mx-auto p-4'):

            # ヘッダー
            with ui.card().classes('w-full mb-4'):
                with ui.row().classes('w-full items-center'):
                    ui.label('💬 LLM対話式ヒアリング').classes('text-2xl font-bold text-primary')
                    ui.space()
                    ui.label('6フェーズの対話セッション').classes('text-gray-600')

            # フロー進捗表示
            with ui.row().classes('w-full mb-4'):
                for i, label in enumerate(self._phase_labels):
                    color = 'text-primary' if i == self.current_phase else ('text-green-600' if i < self.current_phase else 'text-gray-400')
                    icon = '✅' if i < self.current_phase else ('🔄' if i == self.current_phase else '⚪')
                    with ui.column().classes('items-center'):
                        ui.label(icon).classes(f'text-lg {color}')
                        ui.label(label).classes(f'text-xs {color}')

            # 現在のフェーズ内容
            self._phase_card = ui.card().classes('w-full mb-4')
            with self._phase_card:
                self._phase_title = ui.label('').classes('text-xl font-bold mb-2')
                self._phase_description = ui.markdown('')
                self._phase_content = ui.column().classes('w-full mt-4')

            # 対話履歴
            self._conversation_card = ui.card().classes('w-full mb-4')
            with self._conversation_card:
                ui.label('💬 対話履歴').classes('font-bold text-lg mb-2')
                self._chat_container = ui.column().classes('w-full max-h-96 overflow-y-auto')

            # 入力エリア
            self._input_card = ui.card().classes('w-full mb-4')
            with self._input_card:
                with ui.row().classes('w-full gap-2'):
                    self._user_input = ui.textarea(
                        placeholder='ここに回答を入力...',
                        label='あなたの回答',
                    ).classes('flex-1').props('rows=3')
                    with ui.column():
                        ui.button('📤 送信', on_click=self._send_message, color='primary').props('dense')
                        ui.button('⏭ スキップ', on_click=self._skip_phase, color='warning').props('dense outline')

            # ナビゲーションボタン
            with ui.row().classes('w-full justify-center gap-4'):
                self._prev_btn = ui.button('← 前へ', on_click=self._prev_phase).props('outline dense')
                self._next_btn = ui.button('次へ →', on_click=self._next_phase, color='primary').props('dense')
                ui.space()
                self._finish_btn = ui.button('✅ 完了', on_click=self._finish_interview, color='positive').props('dense')

            # 初期化
            self._load_phase(self.current_phase)

    def _load_phase(self, phase: int):
        """指定フェーズを読み込み"""
        self.current_phase = phase
        self._phase_content.clear()

        phase_info = {
            0: {
                'title': '① データ要約',
                'description': 'LLMがデータを分析中...',
                'content': 'データの概要を自動生成し、欠損・異常値を指摘します。',
            },
            1: {
                'title': '② 解析目的（意思決定ゴール）',
                'description': '**あなたは何を決めたいですか？**\n\n例：屈折率を1.6以上にしたい、コストを最小化したい、など',
                'content': '- 予測のためのモデル構築\n- 実験計画のためのデータ補完\n- 目標達成のための条件探索',
            },
            2: {
                'title': '③ 予測対象・目標値',
                'description': '**どの変数を予測しますか？目標値は？**\n\n例：屈折率を1.6以上、Tgを150℃以下、など',
                'content': '- 目的変数を選択\n- 目標値を設定\n- 達成可否の判定基準',
            },
            3: {
                'title': '④ 実験誤差・変数性質',
                'description': '**実験誤差はどのくらい？変数の性質は？**\n\n例：屈折率の測定誤差は±0.01、温度は制御可能、など',
                'content': '- 実験誤差の大きさ\n- 制御可能変数 vs 成り行き変数\n- 変数間の物理化学的関連性',
            },
            4: {
                'title': '⑤ 特徴量選択・ドメイン知識',
                'description': '**どの特徴量が重要ですか？ドメイン知識を教えてください**\n\n例：分極率は絶対に重要、密度はあまり効かない、など',
                'content': '- 特徴量の優先度設定\n- ドメイン知識の入力\n- LLMによる特徴量選択支援',
            },
            5: {
                'title': '⑥ 単調性制約・確認',
                'description': '**変数と目的変数の間に単調性はありますか？**\n\n例：温度が上がると屈折率は下がる（単調減少）、など',
                'content': '- 単調性制約の設定\n- 線形性制約の設定\n- 全設定の確認',
            },
        }

        info = phase_info.get(phase, phase_info[0])
        self._phase_title.text = info['title']
        self._phase_description.content = info['description']

        with self._phase_content:
            ui.markdown(info['content']).classes('text-sm text-gray-600')

        # ボタン状態
        self._prev_btn.enabled = (phase > 0)
        self._next_btn.visible = (phase < 5)
        self._finish_btn.visible = (phase == 5)

    def _send_message(self):
        """メッセージを送信"""
        message = self._user_input.value
        if not message:
            return

        # ユーザーメッセージを追加
        self.conversation.append({
            'role': 'user',
            'phase': self.current_phase,
            'content': message,
        })

        self._user_input.value = ''
        self._update_conversation()

        # 簡易的なLLM応答（実際はLLMを呼び出す）
        response = self._generate_response(message)
        self.conversation.append({
            'role': 'assistant',
            'phase': self.current_phase,
            'content': response,
        })
        self._update_conversation()

    def _generate_response(self, message: str) -> str:
        """簡易的な応答生成（実際はLLMを呼び出す）"""
        phase_responses = {
            0: f'データを分析中...\n\n{message}に基づき、データの特性を評価します。',
            1: f'なるほど、「{message}」ですね。\n\nそれでは、その目標を達成するために必要な分析プランを立てましょう。',
            2: f'目標値「{message}」ですね。\n\nその達成に向けたモデリング戦略を考えます。',
            3: f'実験誤差「{message}」ですね。\n\nその誤差を考慮した解析手法を選択します。',
            4: f'ドメイン知識「{message}」を承知しました。\n\nその知識を特徴量選択と制約設定に反映します。',
            5: f'単調性制約「{message}」ですね。\n\nそれでは、全設定を確認しましょう。',
        }
        return phase_responses.get(self.current_phase, 'ありがとうございます。')

    def _update_conversation(self):
        """対話履歴を更新"""
        self._chat_container.clear()
        with self._chat_container:
            for msg in self.conversation:
                if msg['role'] == 'user':
                    with ui.row().classes('w-full justify-end'):
                        with ui.card().classes('bg-blue-100 max-w-lg'):
                            ui.label(f"[{self._phase_labels[msg['phase']]}]").classes('text-xs text-blue-600')
                            ui.label(msg['content']).classes('text-sm')
                else:
                    with ui.row().classes('w-full justify-start'):
                        with ui.card().classes('bg-gray-100 max-w-lg'):
                            ui.label(f"[{self._phase_labels[msg['phase']]}]").classes('text-xs text-gray-600')
                            ui.markdown(msg['content']).classes('text-sm')

    def _skip_phase(self):
        """フェーズをスキップ"""
        if self.current_phase < 5:
            self._next_phase()

    def _next_phase(self):
        """次のフェーズへ"""
        if self.current_phase < 5:
            self.current_phase += 1
            self._load_phase(self.current_phase)
            ui.notify(f'フェーズ {self.current_phase + 1} へ進みます', type='info')

    def _prev_phase(self):
        """前のフェーズへ"""
        if self.current_phase > 0:
            self.current_phase -= 1
            self._load_phase(self.current_phase)
            ui.notify(f'フェーズ {self.current_phase + 1} に戻ります', type='info')

    def _finish_interview(self):
        """ヒアリング完了"""
        with ui.dialog() as dialog:
            with ui.card().classes('w-full max-w-2xl'):
                ui.label('✅ ヒアリング完了').classes('text-xl font-bold mb-4')

                # サマリー
                ui.label('📊 ヒアリング結果サマリー').classes('font-bold text-lg mb-2')

                summary = f"""
                **目標**: {self.conversation[2]['content'] if len(self.conversation) > 2 else '未設定'}

                **ドメイン知識**:
                {self.conversation[4]['content'] if len(self.conversation) > 4 else '未入力'}

                **単調性制約**:
                {self.conversation[5]['content'] if len(self.conversation) > 5 else '未設定'}
                """

                ui.markdown(summary).classes('text-sm')

                with ui.row().classes('w-full justify-end gap-2'):
                    ui.button('閉じる', on_click=dialog.close).props('flat')
                    ui.button('✅ この内容で確定', on_click=lambda: [dialog.close(), ui.notify('✅ ヒアリング結果を保存しました', type='positive')], color='primary')

        dialog.open()
