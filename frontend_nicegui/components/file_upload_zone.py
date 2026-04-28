"""
frontend_nicegui/components/file_upload_zone.py
修正版: NiceGUI 3.0+ に対応 (UploadEventArgumentsの修正)
"""
from nicegui import ui, events
from pathlib import Path
from typing import Callable, Optional, List, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FileUploadZone:
    """
    NiceGUI用ファイルアップロードコンポーネント
    """
    
    SUPPORTED_TYPES = {
        'csv': ['.csv'],
        'excel': ['.xlsx', '.xls', '.xlsm'],
    }
    
    def __init__(self, 
                 on_upload: Callable[[Dict], None],
                 allowed_types: Optional[List[str]] = None,
                 max_file_size_mb: float = 50,
                 multiple: bool = True,
                 label: str = 'データをアップロード'):
        self.on_upload = on_upload
        self.allowed_types = allowed_types or list(self.SUPPORTED_TYPES.keys())
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.multiple = multiple
        self.label = label
        self._uploaded_files: List[Dict] = []
        
        self._render()
    
    def _render(self):
        """コンポーネントを描画"""
        with ui.card().classes('w-full'):
            ui.label(self.label).classes('text-lg font-bold mb-2')
            
            with ui.row().classes('text-sm text-gray-600 mb-2'):
                ui.label(f"対応ファイル: {', '.join(self.allowed_types)}")
                ui.label(f"• 最大サイズ: {self.max_file_size // (1024*1024)}MB")
            
            with ui.element('div').classes('border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors'):
                ui.icon('upload_file', size='2em').classes('text-gray-400 mb-2')
                ui.label('ファイルをドラッグ＆ドロップ または クリックして選択').classes('text-gray-600')
                
                self._uploader = ui.upload(
                    on_upload=self._handle_upload,
                    on_rejected=self._handle_rejected,
                    multiple=self.multiple,
                    max_file_size=self.max_file_size,
                    accepted_files=','.join(
                        ext for t in self.allowed_types 
                        for ext in self.SUPPORTED_TYPES.get(t, [])
                    )
                ).classes('hidden')
                
                ui.button('ファイルを選択', on_click=self._uploader.run).classes('mt-2')
            
            self._progress = ui.linear_progress(value=0, show_value=True).classes('w-full mt-2').props('color=primary')
            self._progress.visible = False
            self._message = ui.label().classes('text-sm mt-2')
            
            self._file_list = ui.column().classes('mt-4')
            self._render_file_list()
    
    async def _handle_upload(self, e: events.UploadEventArguments):
        """アップロード処理 (修正: e.file.read() 使用)"""
        try:
            self._progress.visible = True
            self._progress.value = 0.3
            self._message.text = 'ファイル読み込み中...'
            
            # NiceGUI 3.0+ では e.content ではなく e.file を使用
            content = await e.file.read()
            file_name = e.file.name
            # sizeはプロパティの場合と非同期メソッドの場合があるが、通常は読み込み後に判明
            file_size = len(content)
            
            self._progress.value = 0.6
            
            file_type = self._detect_type(file_name)
            self._progress.value = 0.8
            
            # 読み込み処理
            try:
                from backend.data.file_uploader import read_csv_smart, read_excel_smart
                if file_type == 'csv':
                    data = read_csv_smart(content)
                elif file_type == 'excel':
                    data = read_excel_smart(content)
                else:
                    raise ValueError(f"サポートされていないファイルタイプ: {file_type}")
            except ImportError:
                # バックエンドがない場合の簡易読み込み
                import io
                if file_type == 'csv':
                    data = pd.read_csv(io.BytesIO(content))
                elif file_type == 'excel':
                    data = pd.read_excel(io.BytesIO(content))
                else:
                    raise ValueError("バックエンドモジュールが必要です")
            
            self._progress.value = 1.0
            
            meta = {
                'filename': file_name,
                'size': file_size,
                'type': file_type,
                'shape': data.shape if isinstance(data, pd.DataFrame) else None,
            }
            
            result = {'file': e, 'data': data, 'meta': meta}
            self.on_upload(result)
            
            self._uploaded_files.append(result)
            self._render_file_list()
            
            self._message.text = f'✓ {file_name} を読み込みました'
            self._message.classes('text-green-600')
            
        except Exception as ex:
            logger.error(f"アップロードエラー: {ex}")
            self._message.text = f'✗ エラー: {str(ex)}'
            self._message.classes('text-red-600')
        finally:
            self._progress.visible = False
    
    def _handle_rejected(self, e: events.UiEventArguments):
        """アップロード拒否時の処理"""
        self._message.text = '✗ ファイル形式またはサイズが許可されていません'
        self._message.classes('text-red-600')
    
    def _detect_type(self, filename: str) -> str:
        """ファイル名からタイプを判定"""
        suffix = Path(filename).suffix.lower()
        for file_type, extensions in self.SUPPORTED_TYPES.items():
            if suffix in extensions and file_type in self.allowed_types:
                return file_type
        return 'unknown'
    
    def _render_file_list(self):
        """アップロード済みファイル一覧を表示"""
        self._file_list.clear()
        if not self._uploaded_files:
            return
        
        with self._file_list:
            ui.label('アップロード済みファイル').classes('font-bold text-sm mb-1')
            for item in self._uploaded_files:
                meta = item['meta']
                with ui.row().classes('items-center text-sm p-2 bg-gray-50 rounded mb-2'):
                    ui.icon('description', size='sm').classes('text-gray-500')
                    ui.label(f"{meta['filename']}").classes('flex-1')
                    if meta.get('shape'):
                        ui.label(f"{meta['shape'][0]}行 × {meta['shape'][1]}列").classes('text-gray-500')
                    ui.icon('check', size='sm').classes('text-green-500')
    
    def clear(self):
        self._uploaded_files.clear()
        self._render_file_list()
        self._message.text = ''
