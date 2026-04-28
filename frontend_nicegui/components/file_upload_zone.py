"""
frontend_nicegui/components/file_upload_zone.py
Drag & Drop ファイルアップロードコンポーネント
既存UIと共存：既存のデータ読み込み機能は維持し、拡張として追加
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
    - ドラッグ＆ドロップ対応
    - 複数ファイル対応
    - ファイルタイプ自動判定
    - アップロード進捗表示
    """
    
    SUPPORTED_TYPES = {
        'csv': ['.csv'],
        'excel': ['.xlsx', '.xls', '.xlsm'],
        'pptx': ['.pptx'],
        'docx': ['.docx'],
    }
    
    def __init__(self, 
                 on_upload: Callable[[Dict], None],
                 allowed_types: Optional[List[str]] = None,
                 max_file_size_mb: float = 50,
                 multiple: bool = True,
                 label: str = 'データをアップロード'):
        """
        Args:
            on_upload: アップロード完了時のコールバック
                       引数: {'file': UploadedFile, 'data': DataFrame|str, 'meta': Dict}
            allowed_types: 許可するファイルタイプ ['csv', 'excel', 'pptx', 'docx']
            max_file_size_mb: 最大ファイルサイズ（MB）
            multiple: 複数ファイルアップロードを許可
            label: アップロードゾーンの表示ラベル
        """
        self.on_upload = on_upload
        self.allowed_types = allowed_types or list(self.SUPPORTED_TYPES.keys())
        self.max_file_size = max_file_size_mb * 1024 * 1024  # bytes
        self.multiple = multiple
        self.label = label
        self._uploaded_files: List[Dict] = []
        
        self._render()
    
    def _render(self):
        """コンポーネントを描画"""
        with ui.card().classes('w-full'):
            # 見出し
            ui.label(self.label).classes('text-lg font-bold mb-2')
            
            # 説明テキスト
            with ui.row().classes('text-sm text-gray-600 mb-2'):
                ui.label(f"対応ファイル: {', '.join(self.allowed_types)}")
                ui.label(f"• 最大サイズ: {int(self.max_file_size // (1024*1024))}MB")
                if self.multiple:
                    ui.label("• 複数ファイル可")
            
            # アップロードエリア
            with ui.element('div').classes('border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors'):
                ui.icon('upload_file', size='2em').classes('text-gray-400 mb-2')
                ui.label('ファイルをドラッグ＆ドロップ または クリックして選択').classes('text-gray-600')
                
                # 実際のアップロードコンポーネント
                self._uploader = ui.upload(
                    on_upload=self._handle_upload,
                    on_rejected=self._handle_rejected,
                    multiple=self.multiple,
                    max_file_size=self.max_file_size,
                    auto_upload=True,
                    label=self.label
                ).classes('w-full')

            
            # 進捗表示
            self._progress = ui.linear_progress(value=0, show_value=True).classes('w-full mt-2').props('color=primary')
            self._progress.visible = False
            
            # 成功/エラーメッセージ
            self._message = ui.label().classes('text-sm mt-2')
            
            # 一覧表示
            self._file_list = ui.column().classes('mt-4')
            self._render_file_list()
    
    async def _handle_upload(self, e: events.UploadEventArguments):
        """アップロード処理"""
        try:
            # 進捗表示
            self._progress.visible = True
            self._progress.value = 0.3
            self._message.text = 'ファイル読み込み中...'
            
            # ファイル内容を読み込み (NiceGUI 3.0.0+ API)
            content = await e.file.read()
            self._progress.value = 0.6
            
            # ファイル情報の取得
            file_name = e.file.name
            file_size = e.file.size()  # NiceGUI 3.8.0 では同期メソッド
            
            # 拡張子からタイプ判定
            file_type = self._detect_type(file_name)
            self._progress.value = 0.8
            
            # 読み込み処理（backendの関数を呼び出し）
            from backend.data.file_uploader import (
                read_csv_smart, read_excel_smart, read_document_to_text
            )
            
            if file_type in ['csv']:
                data = read_csv_smart(content)
            elif file_type in ['excel']:
                data = read_excel_smart(content)
            elif file_type in ['pptx', 'docx']:
                # backend/data/file_uploader.py の期待する文字列に変換
                internal_type = 'powerpoint' if file_type == 'pptx' else 'word'
                data = read_document_to_text(content, internal_type)
            else:
                # Default to text
                data = content.decode('utf-8', errors='ignore')
            
            self._progress.value = 1.0
            
            # メタ情報作成
            meta = {
                'filename': file_name,
                'size': file_size,
                'type': file_type,
                'shape': data.shape if isinstance(data, pd.DataFrame) else None,
            }
            
            # コールバック呼び出し
            result = {'file': e, 'data': data, 'meta': meta}
            self.on_upload(result)
            
            # 一覧に追加
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
                with ui.row().classes('items-center text-sm p-2 bg-gray-50 rounded'):
                    ui.icon('description', size='sm').classes('text-gray-500')
                    ui.label(f"{meta['filename']}").classes('flex-1')
                    if meta.get('shape'):
                        ui.label(f"{meta['shape'][0]}行 × {meta['shape'][1]}列").classes('text-gray-500')
                    ui.icon('check', size='sm').classes('text-green-500')
    
    def clear(self):
        """アップロード履歴をクリア"""
        self._uploaded_files.clear()
        self._render_file_list()
        self._message.text = ''
