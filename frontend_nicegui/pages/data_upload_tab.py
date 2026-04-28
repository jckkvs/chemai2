"""
frontend_nicegui/pages/data_upload_tab.py
AutoML連携修正版
"""
from nicegui import ui, events
import pandas as pd
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DataUploadPage:
    """データアップロードページ"""
    
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.uploaded_data: Optional[pd.DataFrame] = None
        self.quality_report: Optional[Dict] = None
    
    def render(self):
        """ページを描画"""
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):
            
            ui.label('📁 データアップロード & クリーニング').classes('text-2xl font-bold mb-4')
            
            with ui.card().classes('w-full mb-4'):
                ui.label('データをアップロード').classes('font-bold text-lg mb-2')
                ui.label('対応ファイル: csv, excel  • 最大サイズ: 50MB  • 複数ファイル可').classes('text-sm text-gray-600 mb-4')
                
                def on_data_uploaded(result: Dict):
                    self._handle_data_upload(result)
                
                from frontend_nicegui.components.file_upload_zone import FileUploadZone
                FileUploadZone(
                    on_upload=on_data_uploaded,
                    allowed_types=['csv', 'excel'],
                    label='化学データ（CSV/Excel）をアップロード'
                )
            
            self._quality_card = ui.card().classes('w-full mb-4').visible(False)
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
                        ui.icon('description').classes('text-gray-500')
                        ui.label(f"{meta.get('filename', 'unknown')}").classes('flex-1')
                        ui.label(f"{meta.get('shape', (0, 0))[0]}行 × {meta.get('shape', (0, 0))[1]}列").classes('text-sm text-gray-500')
                        ui.icon('check').classes('text-green-500')
                
                ui.notify(f'✓ {meta.get("filename", "unknown")} を読み込みました', type='positive')
            else:
                ui.notify(f'📄 {meta.get("filename", "unknown")} を読み込みました')
                
        except Exception as e:
            logger.error(f"アップロードエラー: {e}", exc_info=True)
            ui.notify(f'✗ エラー: {str(e)}', type='negative')
    
    def _run_cleaning(self):
        if self.uploaded_data is None:
            ui.notify('データが読み込まれていません', type='warning')
            return
        ui.notify('クリーニング機能は開発中です', type='info')
    
    def _navigate_to_automl(self):
        """AutoMLページへ遷移"""
        if self.uploaded_data is None:
            ui.notify('データが読み込まれていません', type='warning')
            return
        
        try:
            from frontend_nicegui.main import navigate_to_automl
            navigate_to_automl(self.uploaded_data)
        except Exception as e:
            logger.error(f"遷移エラー: {e}")
            ui.notify('AutoMLページへの遷移に失敗しました', type='negative')
