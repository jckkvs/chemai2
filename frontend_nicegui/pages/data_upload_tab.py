"""
frontend_nicegui/pages/data_upload_tab.py
データアップロード・クリーニング画面
"""
from nicegui import ui
import pandas as pd
from typing import Dict, Optional

from backend.data.file_uploader import assess_data_quality
from backend.data.data_cleaner import DataCleanerLLM
from frontend_nicegui.components.file_upload_zone import FileUploadZone
from backend.config.llm_settings import LLMConfig

class DataUploadPage:
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.quality_card = None
        self.quality_label = None
        self.clean_btn = None
        self.current_df = None
        self.current_filename = None

    def render(self):
        with ui.column().classes('w-full p-4'):
            ui.label('データアップロード & クリーニング').classes('text-2xl font-bold mb-4')
            
            # アップロードゾーン
            with ui.card().classes('w-full mb-4'):
                self.upload_zone = FileUploadZone(
                    on_upload=self._handle_data_upload,
                    allowed_types=['csv', 'excel', 'pptx', 'docx']
                )
            
            # 品質評価表示
            self.quality_card = ui.card().classes('w-full mt-4')
            self.quality_card.visible = False

            with self.quality_card:
                ui.label('データ品質評価').classes('font-bold text-lg')
                self.quality_label = ui.label()
                with ui.row():
                    self.clean_btn = ui.button('LLM支援でクリーニング', on_click=self._run_cleaning).props('outline')
                    ui.button('このデータで解析へ', on_click=lambda: ui.notify('解析ページへ移行します（未実装）')).props('flat')

    def _handle_data_upload(self, result: Dict):
        """アップロード完了時の処理"""
        data = result['data']
        meta = result['meta']
        self.current_filename = meta['filename']
        
        if isinstance(data, pd.DataFrame):
            self.current_df = data
            # 品質評価
            quality = assess_data_quality(data)
            issues_str = ", ".join(quality['issues']) if quality['issues'] else '問題なし'
            self.quality_label.text = f"✓ {meta['filename']} | 状態: {issues_str}"
            self.quality_card.visible = True
            
            if not quality['is_clean']:
                self.clean_btn.visible = True
            else:
                self.clean_btn.visible = False
        else:
            # 文書テキストの場合
            ui.notify(f'📄 {meta["filename"]} ({len(str(data))}文字) を読み込みました')
            self.quality_card.visible = False

    def _run_cleaning(self):
        """LLM支援クリーニングを実行"""
        if self.current_df is None:
            return
            
        with ui.notify('クリーニング提案を生成中...', spinner=True, type='info'):
            cleaner = DataCleanerLLM(mode=self.llm_config.mode)
            # 実際にはここでLLMを呼び出し、提案をUIに表示する
            # 現時点ではモック的な動作
            suggestions = cleaner.analyze_data_issues(self.current_df)
            
            if not suggestions:
                ui.notify('特に修正の必要は見つかりませんでした')
                return
                
            with ui.dialog() as dialog, ui.card():
                ui.label('クリーニング提案').classes('text-lg font-bold')
                for s in suggestions:
                    with ui.expansion(f"{s.issue_type}: {s.description}", icon='build').classes('w-full'):
                        ui.code(s.suggested_code, language='python')
                
                with ui.row():
                    ui.button('自動適用（信頼度高のみ）', on_click=lambda: self._apply_auto(cleaner, suggestions, dialog))
                    ui.button('閉じる', on_click=dialog.close).props('flat')
            dialog.open()

    def _apply_auto(self, cleaner, suggestions, dialog):
        new_df, report = cleaner.apply_cleaning(self.current_df, suggestions, auto_apply=True)
        self.current_df = new_df
        applied_count = len(report['applied'])
        ui.notify(f'{applied_count} 件の修正を適用しました')
        # 再評価
        self._handle_data_upload({'data': self.current_df, 'meta': {'filename': self.current_filename}})
        dialog.close()
