"""
frontend_nicegui/components/llm_config_dialog.py
LLM設定ダイアログ（目立たない配置：設定メニュー内など）
"""
from nicegui import ui
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class LLMConfigDialog:
    """
    LLM設定を管理するダイアログコンポーネント
    """
    
    def __init__(self, 
                 config: Optional[object] = None,
                 on_config_change: Optional[Callable] = None):
        """
        Args:
            config: 初期設定
            on_config_change: 設定変更時のコールバック
        """
        self.config = config
        self.on_config_change = on_config_change
        self._dialog: Optional[ui.dialog] = None
        
        self._setup()
    
    def _setup(self):
        """ダイアログをセットアップ"""
        with ui.dialog().props('persistent').classes('w-96') as self._dialog:
            with ui.card().classes('w-full'):
                # 見出し
                with ui.row().classes('w-full justify-between items-center mb-2'):
                    ui.label('LLM設定').classes('text-lg font-bold')
                    ui.button('×', on_click=self._dialog.close).props('flat dense round')
                
                # 動作モード選択
                ui.label('動作モード').classes('font-bold text-sm mt-2')
                self._mode_select = ui.radio(
                    options=[
                        ('prompt_only', 'プロンプトのみ生成（セキュア推奨）'),
                        ('local', 'ローカルLLMを使用'),
                        ('api', '外部APIを使用')
                    ],
                    value='prompt_only',
                    on_change=self._on_mode_change
                ).props('dense')
                
                # API設定
                self._api_section = ui.column()
                self._api_section.visible = False
                with self._api_section:
                    ui.label('API設定').classes('font-bold text-sm mt-2')
                    self._api_endpoint = ui.input('APIエンドポイント', placeholder='https://api.openai.com/v1')
                    self._api_key = ui.input('APIキー', password=True, placeholder='sk-...')
                    self._model_name = ui.input('モデル名', placeholder='gpt-4o-mini')
                
                # 詳細設定
                with ui.expansion('詳細設定', icon='settings').props('dense'):
                    self._temperature = ui.slider(min=0, max=1, step=0.1, value=0.1, label='Temperature')
                    self._max_tokens = ui.number('Max Tokens', value=2000, min=100, max=8000)
                    self._enable_exec = ui.switch('LLM生成コードの自動実行', value=False)
                
                # 保存ボタン
                with ui.row().classes('w-full justify-end mt-4'):
                    ui.button('キャンセル', on_click=self._dialog.close).props('flat')
                    ui.button('保存', on_click=self._save_config, color='primary')
    
    def _on_mode_change(self, e):
        """モード変更時のUI更新"""
        self._api_section.visible = (e.value == 'api')
    
    def _save_config(self):
        """設定を保存"""
        try:
            if self.config:
                self.config.mode = self._mode_select.value
                self.config.api_endpoint = self._api_endpoint.value or None
                self.config.api_key = self._api_key.value or None
                self.config.model_name = self._model_name.value
                self.config.temperature = self._temperature.value
                self.config.max_tokens = int(self._max_tokens.value)
                self.config.enable_code_execution = self._enable_exec.value
                
                if hasattr(self.config, 'save'):
                    self.config.save()
            
            if self.on_config_change and self.config:
                self.on_config_change(self.config)
            
            ui.notify('設定を保存しました', type='positive')
            self._dialog.close()
            
        except Exception as ex:
            logger.error(f"設定保存エラー: {ex}")
            ui.notify(f'エラー: {str(ex)}', type='negative')
    
    def open(self):
        """ダイアログを開く"""
        self._dialog.open()
    
    def close(self):
        """ダイアログを閉じる"""
        self._dialog.close()
    
    def create_trigger_button(self, parent, icon: str = 'psychology', 
                             label: str = 'LLM設定') -> ui.button:
        """設定ダイアログを開くトリガーボタンを作成"""
        btn = ui.button(icon=icon, text=label, on_click=self.open).props('flat dense')
        btn.classes('text-gray-500 hover:text-gray-700')
        return btn
