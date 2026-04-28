"""
frontend_nicegui/pages/data_upload_tab.py
AutoML連携版
"""
from nicegui import ui, events
import pandas as pd
from typing import Optional, Dict, List
import logging

# コンポーネント
from frontend_nicegui.components.file_upload_zone import FileUploadZone
from backend.data.file_uploader import assess_data_quality
from backend.data.data_cleaner import DataCleanerLLM

logger = logging.getLogger(__name__)


class DataUploadPage:
    """データアップロードページ"""
    
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.uploaded_data: Optional[pd.DataFrame] = None
        self.quality_report: Optional[Dict] = None
        self._quality_card = None
        self._quality_label = None
        self._file_list = None
        self._clean_btn = None

    def render(self):
        """ページを描画"""
        with ui.column().classes('w-full max-w-6xl mx-auto p-4'):
            
            # ヘッダー
            ui.label('📁 データアップロード & クリーニング').classes('text-2xl font-bold mb-4')
            
            # アップロードゾーン
            with ui.card().classes('w-full mb-4'):
                ui.label('データをアップロード').classes('font-bold text-lg mb-2')
                
                ui.label(
                    '対応ファイル: csv, excel, pptx, docx  • 最大サイズ: 50MB  • 複数ファイル可'
                ).classes('text-sm text-gray-600 mb-4')
                
                # ファイルアップロードコンポーネント
                def on_data_uploaded(result: Dict):
                    self._handle_data_upload(result)
                
                FileUploadZone(
                    on_upload=on_data_uploaded,
                    allowed_types=['csv', 'excel'],
                    label='化学データ（CSV/Excel）をアップロード'
                )
            
            # データ品質評価
            self._quality_card = ui.card().classes('w-full mb-4')
            self._quality_card.visible = False
            with self._quality_card:
                ui.label('データ品質評価').classes('font-bold text-lg mb-2')
                self._quality_label = ui.label()
                
                # クリーニングボタンと解析ボタン
                with ui.row().classes('mt-4 gap-4'):
                    self._clean_btn = ui.button(
                        '🔧 LLM支援でクリーニング',
                        on_click=self._run_cleaning,
                        color='primary'
                    ).props('outline')
                    
                    ui.button(
                        '📊 このデータで解析へ',
                        on_click=self._navigate_to_automl,
                        color='primary'
                    ).props('size=lg')
            
            # アップロード済みファイル一覧
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
                
                # 品質評価
                self.quality_report = assess_data_quality(data)
                
                # UI更新
                issues = self.quality_report.get('issues', [])
                if issues:
                    self._quality_label.text = (
                        f"✓ {meta.get('filename', 'unknown')} | "
                        f"{len(data)}行 × {len(data.columns)}列 | "
                        f"注意点あり"
                    )
                    self._quality_label.classes('text-orange-600')
                else:
                    self._quality_label.text = (
                        f"✓ {meta.get('filename', 'unknown')} | "
                        f"{len(data)}行 × {len(data.columns)}列 | "
                        f"状態: 良好"
                    )
                    self._quality_label.classes('text-green-600')
                
                self._quality_card.visible = True
                
                # ファイル一覧に追加
                with self._file_list:
                    with ui.row().classes('w-full items-center p-2 bg-gray-50 rounded mb-2'):
                        ui.icon('description').classes('text-gray-500')
                        ui.label(f"{meta.get('filename', 'unknown')}").classes('flex-1')
                        ui.label(f"{len(data)}行 × {len(data.columns)}列").classes('text-sm text-gray-500')
                        ui.icon('check').classes('text-green-500')
                
                ui.notify(f'✓ {meta.get("filename", "unknown")} を読み込みました', type='positive')
                
            else:
                ui.notify(f'📄 {meta.get("filename", "unknown")} を読み込みました')
                
        except Exception as e:
            logger.error(f"アップロードエラー: {e}", exc_info=True)
            ui.notify(f'✗ エラー: {str(e)}', type='negative')
    
    def _run_cleaning(self):
        """LLM支援クリーニングを実行"""
        if self.uploaded_data is None:
            ui.notify('データが読み込まれていません', type='warning')
            return
        
        # クリーニング処理（実際には DataCleanerLLM を使用）
        ui.notify('LLMクリーニング提案を生成中...', spinner=True)
        # ここにクリーニングロジックを実装
    
    def _navigate_to_automl(self):
        """AutoMLページへ遷移"""
        if self.uploaded_data is None:
            ui.notify('データが読み込まれていません', type='warning')
            return
        
        # main.py の automl_page インスタンスにデータを渡す
        from frontend_nicegui.main import automl_page, navigate_to_automl
        automl_page.load_data(self.uploaded_data)
        
        # タブを切り替え
        navigate_to_automl(self.uploaded_data)
        
        # JavaScriptでタブを切り替え（UI上の整合性のため）
        ui.run_javascript('''
            const tabs = document.querySelectorAll('.q-tab');
            for (const tab of tabs) {
                if (tab.innerText.includes('AutoML')) {
                    tab.click();
                    break;
                }
            }
        ''')
