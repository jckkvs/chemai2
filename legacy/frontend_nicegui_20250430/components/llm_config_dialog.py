"""
frontend_nicegui/components/llm_config_dialog.py
LLM設定ダイアログ（目立たない配置：設定メニュー内など）
修正版: ui.slider の label 引数削除、ローカルLLM設定拡張
"""
from nicegui import ui
from typing import Optional, Callable
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class LLMConfigDialog:
    """
    LLM設定を管理するダイアログコンポーネント
    動作モード：prompt_only / local / api
    デフォルト：local（ローカルLLM）
    """

    def __init__(self,
                 config: Optional[object] = None,
                 on_config_change: Optional[Callable] = None):
        """
        Args:
            config: 初期設定（LLMConfigオブジェクト）
            on_config_change: 設定変更時のコールバック
        """
        self.config = config
        self.on_config_change = on_config_change
        self._dialog: Optional[ui.dialog] = None

        # ウィジェット参照
        self._mode_select: Optional[ui.radio] = None
        self._api_section: Optional[ui.column] = None
        self._local_section: Optional[ui.column] = None
        self._prompt_section: Optional[ui.column] = None

        # API設定ウィジェット
        self._api_endpoint: Optional[ui.input] = None
        self._api_key: Optional[ui.input] = None
        self._api_model_name: Optional[ui.input] = None

        # ローカルLLM設定ウィジェット
        self._local_model_select: Optional[ui.select] = None
        self._local_download_btn: Optional[ui.button] = None
        self._local_download_status: Optional[ui.label] = None
        self._auto_download_check: Optional[ui.switch] = None

        # 詳細設定ウィジェット
        self._temperature: Optional[ui.slider] = None
        self._max_tokens: Optional[ui.number] = None
        self._enable_exec: Optional[ui.switch] = None

        self._setup()

    def _get_model_catalog(self) -> list[dict]:
        """HFモデルカタログを取得"""
        try:
            from backend.llm.providers.hf_provider import HF_MODEL_CATALOG
            return HF_MODEL_CATALOG
        except Exception as e:
            logger.warning(f"HF_MODEL_CATALOGの取得に失敗: {e}")
            return []

    def _setup(self):
        """ダイアログをセットアップ"""
        with ui.dialog().props('persistent maximized').classes('w-96') as self._dialog:
            with ui.card().classes('w-full max-w-2xl'):
                # 見出し
                with ui.row().classes('w-full justify-between items-center mb-2'):
                    ui.label('LLM設定').classes('text-lg font-bold')
                    ui.button('×', on_click=self._dialog.close).props('flat dense round')

                # 動作モード選択
                ui.label('動作モード').classes('font-bold text-sm mt-2')
                self._mode_select = ui.radio(
                    options={
                        'local': 'ローカルLLMを使用（推奨・デフォルト）',
                        'prompt_only': 'プロンプトのみ生成（セキュア）',
                        'api': '外部APIを使用（OpenAI等）'
                    },
                    value='local',  # デフォルト：ローカルLLM
                    on_change=self._on_mode_change
                ).props('dense')

                ui.separator().classes('my-2')

                # --- ローカルLLM設定 ---
                self._local_section = ui.column()
                self._local_section.visible = True  # デフォルト表示
                with self._local_section:
                    ui.label('ローカルLLM設定').classes('font-bold text-sm mt-2')

                    # モデル選択
                    catalog = self._get_model_catalog()
                    model_options = {m['id']: m.get('label', m['id']) for m in catalog}
                    if not model_options:
                        model_options = {'jckkvs/bonsai-8b-1.58bit': 'Bonsai 8B（軽量・ローカル推論）'}

                    self._local_model_select = ui.select(
                        options=model_options,
                        label='モデル選択',
                        value=self.config.local_model_id if self.config else 'jckkvs/bonsai-8b-1.58bit'
                    ).classes('w-full').props('dense filled dark')

                    # モデル情報表示
                    self._local_model_info = ui.label('').classes('text-xs text-gray-500')

                    # ダウンロードボタンとステータス
                    with ui.row().classes('w-full items-center gap-2'):
                        self._local_download_btn = ui.button(
                            'モデルをダウンロード',
                            icon='download',
                            on_click=self._download_model
                        ).props('outline color=cyan dense')

                        self._local_download_status = ui.label('').classes('text-xs text-gray-500')

                    # 自動ダウンロード設定
                    self._auto_download_check = ui.switch(
                        '初回起動時に自動ダウンロード',
                        value=self.config.auto_download_on_first_run if self.config else True
                    ).props('dense')

                    # ローカルLLM詳細設定
                    with ui.expansion('詳細設定', icon='settings').props('dense'):
                        self._local_temp = ui.slider(
                            label='Temperature',
                            min=0, max=1, step=0.1,
                            value=self.config.local_temperature if self.config else 0.7
                        ).props('dense')
                        self._local_max_tokens = ui.number(
                            'Max Tokens',
                            value=self.config.local_max_tokens if self.config else 1024,
                            min=64, max=4096
                        ).props('dense')

                # --- API設定 ---
                self._api_section = ui.column()
                self._api_section.visible = False
                with self._api_section:
                    ui.label('API設定').classes('font-bold text-sm mt-2')
                    self._api_endpoint = ui.input(
                        'APIエンドポイント',
                        placeholder='https://api.openai.com/v1',
                        value=self.config.api_endpoint or '' if self.config else ''
                    ).props('dense filled dark')

                    self._api_key = ui.input(
                        'APIキー',
                        password=True,
                        placeholder='sk-...',
                        value=self.config.api_key or '' if self.config else ''
                    ).props('dense filled dark')

                    self._api_model_name = ui.input(
                        'モデル名',
                        placeholder='gpt-4o-mini',
                        value=self.config.model_name or '' if self.config else ''
                    ).props('dense filled dark')

                # --- プロンプトのみ生成モード設定 ---
                self._prompt_section = ui.column()
                self._prompt_section.visible = False
                with self._prompt_section:
                    ui.label('プロンプトのみ生成モード').classes('font-bold text-sm mt-2')
                    ui.label(
                        'LLMを実際には呼び出さず、外部LLMに投げる用のプロンプトのみを生成します。'
                        'セキュリティを重視する場合に推奨します。'
                    ).classes('text-xs text-gray-500')

                    self._prompt_template_select = ui.select(
                        options={'standard': '標準', 'chemistry': '化学特化', 'code': 'コード生成'},
                        label='プロンプトテンプレート',
                        value='standard'
                    ).props('dense filled dark')

                ui.separator().classes('my-2')

                # 共通詳細設定
                with ui.expansion('共通詳細設定', icon='tune').props('dense'):
                    # 修正: ui.slider に label 引数はないため、別途ラベルを追加
                    ui.label('Temperature（APIモード用）').classes('text-xs text-gray-500')
                    self._temperature = ui.slider(min=0, max=1, step=0.1, value=0.1).props('dense')

                    self._max_tokens = ui.number('Max Tokens（APIモード用）', value=2000, min=100, max=8000).props('dense')
                    self._enable_exec = ui.switch('LLM生成コードの自動実行', value=False).props('dense')

                # 保存ボタン
                with ui.row().classes('w-full justify-end mt-4'):
                    ui.button('キャンセル', on_click=self._dialog.close).props('flat')
                    ui.button('保存', on_click=self._save_config, color='primary')

    def _on_mode_change(self, e):
        """モード変更時のUI更新"""
        mode = e.value
        self._local_section.visible = (mode == 'local')
        self._api_section.visible = (mode == 'api')
        self._prompt_section.visible = (mode == 'prompt_only')

    async def _download_model(self):
        """モデルをダウンロード"""
        if not self._local_model_select:
            return

        model_id = self._local_model_select.value
        if not model_id:
            ui.notify('モデルを選択してください', type='warning')
            return

        try:
            from backend.llm.providers.hf_provider import download_model_async, is_model_downloaded

            # 既にダウンロード済みか確認
            if is_model_downloaded(model_id):
                ui.notify(f'モデル {model_id} は既にダウンロード済みです', type='info')
                return

            # ダウンロード開始
            self._local_download_btn.disable()
            self._local_download_status.text = 'ダウンロード中...'

            # トークン取得
            token = ''
            try:
                from backend.llm.providers.hf_provider import get_hf_token
                token = get_hf_token()
            except Exception:
                pass

            def _on_progress(prog):
                if hasattr(prog, 'fraction'):
                    pct = int(prog.fraction * 100)
                    self._local_download_status.text = f'ダウンロード中... {pct}%'
                elif hasattr(prog, 'message'):
                    self._local_download_status.text = prog.message

            download_model_async(model_id, token, on_progress=_on_progress)

            # 簡易的に数秒待機（実際はスレッドで進捗監視）
            import asyncio
            await asyncio.sleep(3)

            self._local_download_status.text = 'ダウンロード完了！'
            ui.notify(f'モデル {model_id} のダウンロードが完了しました', type='positive')

        except Exception as e:
            logger.error(f"モデルダウンロードエラー: {e}")
            ui.notify(f'ダウンロードエラー: {str(e)}', type='negative')
        finally:
            if self._local_download_btn:
                self._local_download_btn.enable()

    def _save_config(self):
        """設定を保存"""
        try:
            if not self.config:
                ui.notify('設定オブジェクトがありません', type='negative')
                return

            # 共通設定
            mode = self._mode_select.value
            self.config.mode = mode

            # API設定
            if self._api_endpoint:
                self.config.api_endpoint = self._api_endpoint.value or None
            if self._api_key:
                self.config.api_key = self._api_key.value or None
            if self._api_model_name:
                self.config.model_name = self._api_model_name.value or 'gpt-4o-mini'

            # ローカルLLM設定
            if self._local_model_select:
                self.config.local_model_id = self._local_model_select.value
            if hasattr(self.config, 'auto_download_on_first_run') and self._auto_download_check:
                self.config.auto_download_on_first_run = self._auto_download_check.value
            if hasattr(self.config, 'local_temperature') and hasattr(self, '_local_temp'):
                self.config.local_temperature = self._local_temp.value
            if hasattr(self.config, 'local_max_tokens') and hasattr(self, '_local_max_tokens'):
                self.config.local_max_tokens = int(self._local_max_tokens.value)

            # 共通詳細設定
            if self._temperature:
                self.config.temperature = self._temperature.value
            if self._max_tokens:
                self.config.max_tokens = int(self._max_tokens.value)
            if self._enable_exec:
                self.config.enable_code_execution = self._enable_exec.value

            # 保存
            if hasattr(self.config, 'save'):
                self.config.save()
            elif hasattr(self.config, 'save_config'):
                self.config.save_config()

            # LLMSettings にも反映（バックエンド同期）
            try:
                from backend.llm.config import LLMSettings, save_settings
                settings = LLMSettings(
                    operation_mode=mode,
                    preferred_model=self.config.local_model_id if mode == 'local' else None,
                    default_temperature=self.config.local_temperature if mode == 'local' else self.config.temperature,
                    default_max_tokens=self.config.local_max_tokens if mode == 'local' else self.config.max_tokens,
                    auto_download=self.config.auto_download_on_first_run if mode == 'local' else True,
                    auto_download_on_first_run=self.config.auto_download_on_first_run if hasattr(self.config, 'auto_download_on_first_run') else True,
                    api_endpoint=self.config.api_endpoint if mode == 'api' else None,
                    api_key=self.config.api_key if mode == 'api' else None,
                    api_model_name=self.config.model_name if mode == 'api' else 'gpt-4o-mini',
                    prompt_template=self._prompt_template.value if hasattr(self, '_prompt_template') else 'standard',
                )
                save_settings(settings)
            except Exception as sync_ex:
                logger.warning(f"LLMSettings同期スキップ: {sync_ex}")

            if self.on_config_change and self.config:
                self.on_config_change(self.config)

            ui.notify('設定を保存しました', type='positive')
            self._dialog.close()

        except Exception as ex:
            logger.error(f"設定保存エラー: {ex}")
            ui.notify(f'エラー: {str(ex)}', type='negative')

    def open(self):
        """ダイアログを開く"""
        # 設定値をUIに反映
        if self.config:
            if self._mode_select:
                self._mode_select.value = self.config.mode or 'local'
                self._on_mode_change(type('E', (), {'value': self.config.mode or 'local'})())

            if self._local_model_select and hasattr(self.config, 'local_model_id'):
                self._local_model_select.value = self.config.local_model_id

            if self._auto_download_check and hasattr(self.config, 'auto_download_on_first_run'):
                self._auto_download_check.value = self.config.auto_download_on_first_run

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
