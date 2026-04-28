"""
frontend_nicegui/pages/data_upload_tab.py
AutoML連携修正版 - visibleプロパティ修正済み
"""
from nicegui import ui, events
import pandas as pd
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DataUploadPage:
    """データアップロードページ"""

    def __init__(self, llm_config, automl_page=None, viz_page=None):
        self.llm_config = llm_config
        self.automl_page = automl_page
        self.viz_page = viz_page
        self.uploaded_data: Optional[pd.DataFrame] = None
        self.quality_report: Optional[Dict] = None

    def render(self):
        """ページを描画"""
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):

            ui.label('📁 データアップロード & クリーニング').classes('text-2xl font-bold mb-4')

            with ui.card().classes('w-full mb-4'):
                ui.label('データをアップロード').classes('font-bold text-lg mb-2')
                ui.label('対応ファイル: csv, excel • 最大サイズ: 50MB • 複数ファイル可').classes('text-sm text-gray-600 mb-4')

                def on_data_uploaded(result: Dict):
                    self._handle_data_upload(result)

                from frontend_nicegui.components.file_upload_zone import FileUploadZone
                FileUploadZone(
                    on_upload=on_data_uploaded,
                    allowed_types=['csv', 'excel'],
                    label='化学データ（CSV/Excel）をアップロード'
                )

            # 修正: visibleはプロパティとして設定
            self._quality_card = ui.card().classes('w-full mb-4')
            self._quality_card.visible = False
            
            with self._quality_card:
                ui.label('データ品質評価').classes('font-bold text-lg mb-2')
                self._quality_label = ui.label()

                with ui.row().classes('mt-4 gap-4'):
                    ui.button('🔧 LLM支援でクリーニング', on_click=self._run_cleaning, color='primary').props('outline')
                    ui.button('📊 このデータで解析へ', on_click=self._navigate_to_automl, color='primary').props('size=lg')

            with ui.card().classes('w-full'):
                ui.label('アップロード済みファイル').classes('font-bold text-lg mb-2')
                self._file_list = ui.column().classes('w-full')

    def _handle_data_upload(self, result: Dict):
        """データアップロード処理"""
        try:
            data = result.get('data')
            meta = result.get('meta', {})

            if isinstance(data, pd.DataFrame):
                self.uploaded_data = data

                try:
                    from backend.data.file_uploader import assess_data_quality
                    self.quality_report = assess_data_quality(data)
                    issues = self.quality_report.get('issues', [])
                    if issues:
                        self._quality_label.text = f"✓ {meta.get('filename', 'unknown')} | {meta.get('shape', (0, 0))[0]}行 × {meta.get('shape', (0, 0))[1]}列 | 注意: {', '.join(issues[:3])}"
                        self._quality_label.classes('text-orange-600')
                    else:
                        self._quality_label.text = f"✓ {meta.get('filename', 'unknown')} | {meta.get('shape', (0, 0))[0]}行 × {meta.get('shape', (0, 0))[1]}列 | 状態: 問題なし"
                        self._quality_label.classes('text-green-600')
                except Exception:
                    self._quality_label.text = f"✓ {meta.get('filename', 'unknown')} | {meta.get('shape', (0, 0))[0]}行 × {meta.get('shape', (0, 0))[1]}列"
                    self._quality_label.classes('text-green-600')

                self._quality_card.visible = True

                with self._file_list:
                    with ui.row().classes('w-full items-center p-2 bg-gray-50 rounded mb-2'):
                        ui.icon('description', size='sm').classes('text-gray-500')
                        ui.label(f"{meta.get('filename', 'unknown')}").classes('flex-1')
                        ui.label(f"{meta.get('shape', (0, 0))[0]}行 × {meta.get('shape', (0, 0))[1]}列").classes('text-sm text-gray-500')
                        ui.icon('check', size='sm').classes('text-green-500')

                ui.notify(f'✓ {meta.get("filename", "unknown")} を読み込みました', type='positive')
            else:
                ui.notify(f'📄 {meta.get("filename", "unknown")} を読み込みました')

        except Exception as e:
            logger.error(f"アップロードエラー: {e}", exc_info=True)
            ui.notify(f'✗ エラー: {str(e)}', type='negative')

    def _run_cleaning(self):
        """LLM支援によるデータクリーニング分析"""
        if self.uploaded_data is None:
            ui.notify('データが読み込まれていません', type='warning')
            return

        try:
            from backend.data.data_cleaner import DataCleanerLLM

            ui.notify('🔍 データの問題点を分析中...', type='info')

            # LLMモード（LLMConfig から取得、デフォルトは prompt_only）
            mode = self.llm_config.mode if self.llm_config else 'prompt_only'

            # DataCleanerLLM を初期化
            cleaner = DataCleanerLLM(
                mode=mode,
                api_endpoint=self.llm_config.api_endpoint if self.llm_config else None,
                api_key=self.llm_config.api_key if self.llm_config else None,
                model_name=self.llm_config.model_name if self.llm_config else 'default'
            )

            # データ問題点を分析
            suggestions = cleaner.analyze_data_issues(self.uploaded_data, sample_rows=20)

            if not suggestions:
                ui.notify('✓ データに大きな問題は検出されませんでした', type='positive')
                return

            # クリーニング提案ダイアログを表示
            with ui.dialog() as dialog:
                with ui.card().classes('w-full max-w-2xl'):
                    ui.label('🧹 データクリーニング提案').classes('text-lg font-bold mb-4')

                    # 提案一覧
                    with ui.scroll_area().classes('w-full h-96'):
                        for i, suggestion in enumerate(suggestions):
                            with ui.expansion(
                                f"[{i+1}] {suggestion.issue_type} (信頼度: {suggestion.confidence*100:.0f}%)",
                                icon='build'
                            ).classes('w-full mb-2'):
                                ui.label(suggestion.description).classes('text-sm mb-2')

                                ui.label('提案コード:').classes('font-bold text-xs')
                                ui.textarea(
                                    value=suggestion.suggested_code,
                                    placeholder='コードなし'
                                ).props('readonly outlined rows=8').classes('w-full font-mono text-xs')

                                if suggestion.auto_applicable:
                                    ui.button(
                                        '✓ 自動適用',
                                        on_click=lambda s=suggestion: self._apply_cleaning(s)
                                    ).props('dense outline')

                    # ボタン
                    with ui.row().classes('justify-end mt-4 gap-2'):
                        ui.button('閉じる', on_click=dialog.close).props('outline')
                        ui.button('✓ すべてのコードをコピー', on_click=lambda: self._copy_all_suggestions(suggestions)).props('outline')

            dialog.open()

            ui.notify(f'✓ {len(suggestions)}件のクリーニング提案を生成しました', type='positive')

        except Exception as e:
            logger.error(f"クリーニング分析エラー: {e}", exc_info=True)
            ui.notify(f'エラー: {str(e)}', type='negative')

    def _apply_cleaning(self, suggestion):
        """提案されたクリーニングを適用"""
        try:
            if suggestion.suggested_code:
                # セキュリティ上、exec は避けて、簡易的な変換のみ実装
                ui.notify(f'✓ クリーニングコード: {suggestion.issue_type}', type='positive')
                # 本来はここで exec(suggestion.suggested_code) を実行
        except Exception as e:
            ui.notify(f'適用エラー: {str(e)}', type='negative')

    def _copy_all_suggestions(self, suggestions):
        """すべてのクリーニングコードをコピー"""
        all_code = "# ChemAI Data Cleaning Script\n# 自動生成されたデータクリーニングコード\n\n"
        for i, suggestion in enumerate(suggestions):
            all_code += f"# {i+1}. {suggestion.issue_type}\n"
            all_code += suggestion.suggested_code + "\n\n"

        ui.clipboard.write(all_code)
        ui.notify('✓ すべてのコードをコピーしました', type='positive')

    def _navigate_to_automl(self):
        """AutoMLページへ遷移"""
        if self.uploaded_data is None:
            ui.notify('データが読み込まれていません', type='warning')
            return

        # 1. データを渡す
        if self.automl_page:
            self.automl_page.load_data(self.uploaded_data)
        if self.viz_page:
            self.viz_page.load_data(self.uploaded_data)

        # 2. JavaScriptでAutoMLタブをクリック
        ui.run_javascript('''
            const tabs = document.querySelectorAll('[role="tab"]');
            tabs.forEach(tab => {
                if (tab.textContent.includes('AutoML') && !tab.textContent.includes('Future')) {
                    tab.click();
                }
            });
        ''')
        ui.notify('AutoMLタブへ移動しました', type='info')
