"""
frontend_nicegui/pages/llm_assistant_tab.py
LLMアシスタント画面
"""
from nicegui import ui
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class LLMAssistantPage:
    """LLMアシスタントページ"""
    
    def __init__(self, llm_config, llm_dialog):
        self.llm_config = llm_config
        self.llm_dialog = llm_dialog
    
    def render(self):
        """ページを描画"""
        with ui.column().classes('w-full max-w-4xl mx-auto p-4'):
            ui.label('✨ LLM アシスタント').classes('text-2xl font-bold mb-4')
            
            # 外部チャット用プロンプト生成
            with ui.card().classes('w-full'):
                ui.label('🔐 セキュア環境向け: 外部高精度LLM用プロンプト生成').classes('font-bold')
                ui.label('ローカルで実行できない場合、このプロンプトをコピーして外部チャットで使用').classes('text-sm text-gray-600 mb-2')
                
                self._prompt_area = ui.textarea().props('readonly outlined autogrow').classes('w-full h-48 font-mono text-sm')
                
                with ui.row():
                    ui.button('プロンプト生成', on_click=self._generate_external_prompt)
                    ui.button('コピー', on_click=lambda: ui.clipboard.write(self._prompt_area.value))
            
            # 設定へのリンク
            with ui.row().classes('justify-end mt-2'):
                self.llm_dialog.create_trigger_button(ui.row(), label='⚙️ 設定')
    
    def _generate_external_prompt(self):
        """外部チャット用プロンプトを生成"""
        prompt = """# ChemAI Data Analysis Assistant

あなたは化学データ分析の専門家アシスタントです。
以下のコンテキストとデータに基づいて、分析支援を行ってください。

## セッション情報
- アプリ: ChemAI ML Studio
- ユーザーの分析目的: 化学データの自動解析
- 現在のステップ: data_upload

## 分析対象データ
データ読み込み済み

## 現在の問題・質問
このデータから最適な分析方針を提案してください

## 期待する回答形式
Markdown形式で、見出しを使って構造化

## 制約事項
- 化学的妥当性を最優先
- 数値計算は単位・有効数字に注意
- コードを提示する場合は、必ず説明を付与
"""
        self._prompt_area.value = prompt
        ui.notify('プロンプトを生成しました。コピーして外部チャットで使用してください')
