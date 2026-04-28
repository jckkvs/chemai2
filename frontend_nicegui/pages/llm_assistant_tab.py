"""
frontend_nicegui/pages/llm_assistant_tab.py
LLMアシスタント - ローカルLLM + 外部チャットプロンプト生成
"""
from nicegui import ui
from typing import Optional, Dict, List
import logging
import asyncio

logger = logging.getLogger(__name__)


class LLMAssistantPage:
    """LLMアシスタントページ - チャット + プロンプト生成"""

    def __init__(self, llm_config, llm_dialog):
        self.llm_config = llm_config
        self.llm_dialog = llm_dialog
        self.chat_history: List[Dict[str, str]] = []

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
                            'external': '🔐 外部高精度LLM向けプロンプト生成（セキュア）',
                            'local': '💬 ローカルLLMで実行（Ollama）',
                        },
                        value='external'
                    ).classes('flex-1')

                    with ui.column().classes('ml-auto text-right'):
                        ui.label('LLM設定').classes('text-xs text-gray-500')
                        self.llm_dialog.create_trigger_button(ui.row(), label='⚙️', icon='settings')

            # 外部チャット用パネル
            self._external_panel = ui.card().classes('w-full mb-4')
            with self._external_panel:
                ui.label('🔐 外部高精度LLM用プロンプト生成').classes('font-bold')
                ui.label('ローカル実行できない場合、このプロンプトをコピーして外部チャット（ChatGPT, Claude等）で使用').classes('text-sm text-gray-600 mb-2')

                self._prompt_area = ui.textarea().props('readonly outlined autogrow').classes('w-full h-48 font-mono text-sm')

                with ui.row():
                    ui.button('📋 プロンプト生成', on_click=self._generate_external_prompt, color='primary')
                    ui.button('📋 コピー', on_click=self._copy_prompt).props('outline')

            # ローカルチャット用パネル
            self._local_panel = ui.card().classes('w-full mb-4')
            self._local_panel.visible = False

            with self._local_panel:
                ui.label('💬 ローカルLLMチャット').classes('font-bold')
                ui.label('Ollama等のローカルLLMと対話（デモ: 静的応答）').classes('text-sm text-gray-600 mb-2')

                # チャット履歴表示
                self._chat_display = ui.column().classes('w-full p-4 bg-gray-50 rounded h-80 overflow-y-auto border')
                self._update_chat_display()

                # 入力エリア
                with ui.row().classes('w-full gap-2 mt-2'):
                    self._chat_input = ui.input(placeholder='質問を入力...').classes('flex-1')
                    self._send_btn = ui.button('送信', on_click=self._send_message_sync, color='primary')
                    self._clear_btn = ui.button('クリア', on_click=self._clear_chat).props('outline')

                # ステータス
                self._status_label = ui.label('').classes('text-xs text-gray-500 mt-2')

            # モード変更イベント（selectのvalue_changeイベント）
            self._mode_select.on_value_change(self._on_mode_changed)

    def _on_mode_changed(self, event):
        """LLM実行モード変更"""
        mode = self._mode_select.value
        self._external_panel.visible = (mode == 'external')
        self._local_panel.visible = (mode == 'local')

        if mode == 'local':
            ui.notify('💬 ローカルLLMモード', type='info')
        else:
            ui.notify('🔐 外部チャット向けプロンプト生成', type='info')

    def _generate_external_prompt(self):
        """外部チャット用プロンプトを生成"""
        if not self.chat_history:
            system_context = """# ChemAI Data Analysis Assistant

あなたは化学データ分析の専門家アシスタントです。
以下のコンテキストとデータに基づいて、分析支援を行ってください。

## セッション情報
- アプリ: ChemAI ML Studio
- ユーザーの分析目的: 化学データの自動解析
- 利用可能な機能:
  * AutoML: 自動機械学習（RF, XGBoost, LightGBM等）
  * 化学記述子: RDKit, Mordred, COSMO-RS等
  * 解釈性: SHAP, SAGE値
  * 可視化: PCA, t-SNE, UMAP

## 期待する回答形式
1. 分析戦略の提案（化学的背景を踏まえて）
2. 推奨するパラメータ設定
3. 注意点と制約事項
4. 検証方法

## 制約事項
- 化学的妥当性を最優先
- 数値計算は単位・有効数字に注意
- コードを提示する場合は、必ず説明と化学的背景を付与"""
        else:
            system_context = """# ChemAI Data Analysis Assistant (Conversation)

前の会話:
"""
            for msg in self.chat_history[-5:]:
                role = msg['role'].capitalize()
                content_preview = msg['content'][:80]
                system_context += f"\n{role}: {content_preview}"

        prompt = system_context + """

## 現在の質問
[ここに新しい質問を入力]

## 回答形式
Markdown形式で構造化して回答してください"""

        self._prompt_area.value = prompt
        ui.notify('✓ プロンプトを生成しました。コピーして外部チャットで使用してください', type='positive')

    def _copy_prompt(self):
        """プロンプトをコピー"""
        if self._prompt_area.value:
            ui.clipboard.write(self._prompt_area.value)
            ui.notify('✓ プロンプトをコピーしました', type='positive')
        else:
            ui.notify('⚠️ プロンプトを生成してください', type='warning')

    def _send_message_sync(self):
        """メッセージ送信（同期版）"""
        message = self._chat_input.value.strip()
        if not message:
            return

        # ユーザーメッセージを表示
        self.chat_history.append({'role': 'user', 'content': message})
        self._update_chat_display()
        self._chat_input.value = ''

        self._status_label.text = '⏳ 処理中...'
        self._send_btn.enabled = False

        try:
            # ローカルLLMで簡易応答を生成（デモ）
            response = self._generate_demo_response(message)

            # アシスタント応答を表示
            self.chat_history.append({'role': 'assistant', 'content': response})
            self._update_chat_display()
            self._status_label.text = '✓ 完了'

        except Exception as e:
            logger.error(f"メッセージ処理エラー: {e}", exc_info=True)
            error_msg = f"エラー: {str(e)}"
            self.chat_history.append({'role': 'error', 'content': error_msg})
            self._update_chat_display()
            self._status_label.text = '✗ エラーが発生しました'

        finally:
            self._send_btn.enabled = True

    def _generate_demo_response(self, user_message: str) -> str:
        """デモ用の簡易応答を生成"""
        # 簡易的なキーワードマッチングで応答を生成
        lower_msg = user_message.lower()

        responses = {
            'automl': '🤖 AutoML機能について：\n\nChemAIのAutoMLエンジンは以下の特徴があります：\n1. **複数モデルの自動評価**: Random Forest, XGBoost, LightGBM, CatBoost等\n2. **交差検証**: KFold, StratifiedKFold等で信頼性の高い評価\n3. **ハイパーパラメータ最適化**: Optuna/Bayes SearchによるPA自動チューニング\n\n推奨される使用フロー：\n- データをアップロード\n- 目的変数を選択\n- AutoMLを実行\n- 結果をSHAPで解釈',

            '記述子': '🧪 化学記述子について：\n\nChemAIでは複数の記述子計算エンジンをサポートしています：\n\n**標準搭載:**\n- RDKit: 物理化学特性（MolWt, LogP, TPSA等）\n- フィンガープリント: Morgan, MACCS等\n\n**オプション:**\n- Mordred: 1800+のQSAR記述子\n- COSMO-RS: 溶媒和自由エネルギー\n\n推奨：分子の性質に応じて複数の記述子セットを試すことが重要です',

            '可視化': '📊 データ可視化について：\n\nVisualizationタブで以下が利用可能：\n- **PCA**: 高速な次元削減\n- **t-SNE**: 非線形な分類境界を可視化\n- **UMAP**: バランス型の投影\n- **相関ヒートマップ**: 特徴量間の相関を表示\n\n使用時のポイント：\n1. 事前に特徴量を標準化\n2. 次元削減後のプロット色分けでグループ分けを視覚化',

            'shap': '🔍 SHAP解釈について：\n\nAutoML実行後、結果表示タブでSHAP値を確認可能：\n- **特徴量重要度**: 各特徴量がモデル予測に与える影響\n- **Dependenceプロット**: 特徴量と予測値の関係性\n- **Waterfallプロット**: 個別予測の説明\n\n化学的意義：\n- HOMO-LUMOギャップが重要 → 電子状態の寄与が大\n- ログP（脂溶性）が重要 → 疎水性相互作用が支配的',

            '化学': '🧬 化学データの分析について：\n\nChemAIはMI（Materials Informatics）に最適化：\n\n1. **SMILES入力対応**: 分子構造を自動解析\n2. **化学的妥当性チェック**: 異常値検出\n3. **記述子の化学的解釈**: 各記述子の物理化学的意味\n\n推奨フロー：\nSMILES入力 → 記述子生成 → AutoML → SHAP解釈 → 新分子設計',
        }

        # キーワードマッチング
        for keyword, response_text in responses.items():
            if keyword in lower_msg:
                return response_text

        # デフォルト応答
        return f"""ご質問ありがとうございます。

ChemAI ML Studioについて、以下の機能をサポートしています：
- 📁 **Data Upload**: CSV/Excel形式のデータアップロード
- 🤖 **AutoML**: 自動機械学習による最適モデル探索
- 📊 **Visualization**: PCA/t-SNE/UMAP等の可視化
- 🔍 **SHAP解釈**: SHAPによるモデル解釈性確保
- 💾 **Export**: PDFレポート・モデル保存

詳しく知りたいテーマについて、以下のいずれかで質問してください：
- 「AutoML」「記述子」「可視化」「SHAP」「化学」

あなたの質問: "{user_message}"

より詳細な情報が必要な場合は、左上の⚙️ボタンからLLM設定を確認し、外部の高精度LLM（Claude, ChatGPT等）をお試しください。"""

    def _update_chat_display(self):
        """チャット履歴を画面に表示"""
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

        # スクロール最下部へ
        self._chat_display.scroll_to('bottom')

    def _clear_chat(self):
        """チャット履歴をクリア"""
        self.chat_history.clear()
        self._update_chat_display()
        self._status_label.text = ''
        ui.notify('✓ チャット履歴をクリアしました', type='positive')
