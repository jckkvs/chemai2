"""
frontend_nicegui/components/file_upload_zone.py
Drag & Drop file upload component for NiceGUI
Corrected: ui.upload() usage - removed .run() call and fixed trigger method
"""
from nicegui import ui, events
from pathlib import Path
from typing import Callable, Optional, List, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FileUploadZone:
    """
    NiceGUI file upload component with drag & drop support
    """
    
    SUPPORTED_EXTENSIONS = {
        'csv': ['.csv'],
        'excel': ['.xlsx', '.xls', '.xlsm'],
    }
    
    def __init__(self,
                 on_upload: Callable[[Dict], None],
                 allowed_types: Optional[List[str]] = None,
                 max_file_size_mb: float = 50,
                 multiple: bool = True,
                 label: str = 'Upload Data'):
        self.on_upload = on_upload
        self.allowed_types = allowed_types or list(self.SUPPORTED_EXTENSIONS.keys())
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.max_file_size_mb = max_file_size_mb
        self.multiple = multiple
        self.label = label
        self._uploaded_files: List[Dict] = []

        self._render()
    
    def _render(self):
        """Render the component"""
        with ui.card().classes('w-full'):
            ui.label(self.label).classes('text-lg font-bold mb-2')
            
            with ui.row().classes('text-sm text-gray-600 mb-2'):
                ui.label(f"Supported: {', '.join(self.allowed_types)}")
                ui.label(f"• Max: {int(self.max_file_size_mb)}MB")
            
            # Drop zone area
            with ui.element('div').classes('relative border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors cursor-pointer'):
                ui.icon('upload_file', size='2em').classes('text-gray-400 mb-2')
                ui.label('Click to select files or drag & drop').classes('text-gray-600')
                
                # auto_upload=True: upload immediately on selection (no hidden submit button needed)
                # max_file_size omitted: Python-side validation avoids Quasar unit ambiguity
                self._uploader = ui.upload(
                    on_upload=self._handle_upload,
                    on_rejected=self._handle_rejected,
                    multiple=self.multiple,
                    auto_upload=True,
                ).classes('absolute inset-0 opacity-0 cursor-pointer')
            
            # Progress bar
            self._progress = ui.linear_progress(value=0, show_value=True).classes('w-full mt-2').props('color=primary')
            self._progress.visible = False
            
            # Status message
            self._message = ui.label().classes('text-sm mt-2')
            
            # File list
            self._file_list = ui.column().classes('mt-4')
            self._render_file_list()
    
    async def _handle_upload(self, e: events.UploadEventArguments):
        """Handle file upload"""
        try:
            self._progress.visible = True
            self._progress.value = 30
            self._message.text = 'Reading file...'
            
            # Read file content
            content = await e.file.read()
            file_name = e.file.name
            file_size = len(content)

            self._progress.value = 60

            # Python-side size validation
            if file_size > self.max_file_size_bytes:
                raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB (max {int(self.max_file_size_mb)}MB)")

            # Detect file type
            file_type = self._detect_type(file_name)

            if file_type not in self.allowed_types:
                raise ValueError(f"Unsupported file type: {Path(file_name).suffix} (allowed: {self.allowed_types})")
            
            self._progress.value = 80
            
            # Read data based on type
            try:
                from backend.data.file_uploader import read_csv_smart, read_excel_smart
                if file_type == 'csv':
                    data = read_csv_smart(content)
                elif file_type == 'excel':
                    data = read_excel_smart(content)
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
            except ImportError:
                import io
                if file_type == 'csv':
                    data = pd.read_csv(io.BytesIO(content))
                else:
                    raise ValueError("Backend module required")
            
            self._progress.value = 100
            
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
            
            self._message.text = f'✓ Loaded: {file_name}'
            self._message.classes('text-green-600')
            
        except Exception as ex:
            logger.error(f"Upload error: {ex}")
            self._message.text = f'✗ Error: {str(ex)}'
            self._message.classes('text-red-600')
        finally:
            self._progress.visible = False
    
    def _handle_rejected(self, e: events.UiEventArguments):
        """Handle rejected upload (fired by Quasar client-side validation)"""
        self._message.text = '✗ Upload rejected by browser (file may exceed limits)'
        self._message.classes('text-red-600')
        logger.warning(f"Upload rejected event: {e}")
    
    def _detect_type(self, filename: str) -> str:
        """Detect file type from extension"""
        suffix = Path(filename).suffix.lower()
        for file_type, extensions in self.SUPPORTED_EXTENSIONS.items():
            if suffix in extensions and file_type in self.allowed_types:
                return file_type
        return 'unknown'
    
    def _render_file_list(self):
        """Render uploaded files list"""
        self._file_list.clear()
        if not self._uploaded_files:
            return
        
        with self._file_list:
            ui.label('Uploaded Files').classes('font-bold text-sm mb-1')
            for item in self._uploaded_files:
                meta = item['meta']
                with ui.row().classes('items-center text-sm p-2 bg-gray-50 rounded'):
                    ui.icon('description', size='sm').classes('text-gray-500')
                    ui.label(f"{meta['filename']}").classes('flex-1')
                    if meta.get('shape'):
                        ui.label(f"{meta['shape'][0]} rows × {meta['shape'][1]} cols").classes('text-gray-500')
                    ui.icon('check', size='sm').classes('text-green-500')
    
    def clear(self):
        """Clear uploaded files"""
        self._uploaded_files.clear()
        self._render_file_list()
        self._message.text = ''
