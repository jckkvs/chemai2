"""
frontend_nicegui/components/file_upload_zone.py
Drag & Drop file upload component for NiceGUI
Corrected: removed unsupported 'accepted_files' parameter from ui.upload()
"""
from nicegui import ui, events
from pathlib import Path
from typing import Callable, Optional, List, Dict, Union
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
        'powerpoint': ['.pptx'],
        'word': ['.docx'],
    }
    
    def __init__(self, 
                 on_upload: Callable[[Dict], None],
                 allowed_types: Optional[List[str]] = None,
                 max_file_size_mb: float = 50,
                 multiple: bool = True,
                 label: str = 'Upload Data'):
        """
        Args:
            on_upload: Callback function when file is uploaded
            allowed_types: List of allowed file types ['csv', 'excel', etc.]
            max_file_size_mb: Maximum file size in MB
            multiple: Allow multiple file selection
            label: Display label for the upload zone
        """
        self.on_upload = on_upload
        self.allowed_types = allowed_types or list(self.SUPPORTED_EXTENSIONS.keys())
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.multiple = multiple
        self.label = label
        self._uploaded_files: List[Dict] = []
        
        self._render()
    
    def _render(self):
        """Render the component"""
        with ui.card().classes('w-full'):
            # Header
            ui.label(self.label).classes('text-lg font-bold mb-2')
            
            # Info text
            with ui.row().classes('text-sm text-gray-600 mb-2'):
                ui.label(f"Supported: {', '.join(self.allowed_types)}")
                ui.label(f"• Max: {self.max_file_size // (1024*1024)}MB")
            
            # Drop zone
            with ui.element('div').classes('border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors'):
                ui.icon('upload_file', size='2em').classes('text-gray-400 mb-2')
                ui.label('Drag & drop files or click to select').classes('text-gray-600')
                
                # Hidden upload component - corrected: removed 'accepted_files' parameter
                self._uploader = ui.upload(
                    on_upload=self._handle_upload,
                    on_rejected=self._handle_rejected,
                    multiple=self.multiple,
                    max_file_size=self.max_file_size
                ).classes('hidden')
                
                # Trigger button
                ui.button('Select Files', on_click=self._uploader.run).classes('mt-2')
            
            # Progress bar
            self._progress = ui.linear_progress(value=0, show_value=True).classes('w-full mt-2').props('color=primary')
            self._progress.visible = False
            
            # Status message
            self._message = ui.label().classes('text-sm mt-2')
            
            # File list
            self._file_list = ui.column().classes('mt-4')
            self._render_file_list()
    
    async def _handle_upload(self, e: events.UploadEventArguments):
        """Handle file upload - corrected for NiceGUI 3.0+"""
        try:
            # Show progress
            self._progress.visible = True
            self._progress.value = 30
            self._message.text = 'Reading file...'
            
            # Read file content - corrected: use e.file instead of e.content
            content = await e.file.read()
            file_name = e.file.name
            file_size = len(content)  # Use length of content since it's already read
            
            self._progress.value = 60
            
            # Detect file type
            file_type = self._detect_type(file_name)
            
            # Check if type is allowed
            if file_type not in self.allowed_types:
                raise ValueError(f"Unsupported file type: {file_type}")
            
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
                # Fallback without backend
                import io
                if file_type == 'csv':
                    data = pd.read_csv(io.BytesIO(content))
                elif file_type == 'excel':
                    data = pd.read_excel(io.BytesIO(content))
                else:
                    raise ValueError("Backend module required for this file type")
            
            self._progress.value = 100
            
            # Create metadata
            meta = {
                'filename': file_name,
                'size': file_size,
                'type': file_type,
                'shape': data.shape if isinstance(data, pd.DataFrame) else None,
            }
            
            # Call callback
            result = {'file': e, 'data': data, 'meta': meta}
            self.on_upload(result)
            
            # Update file list
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
        """Handle rejected upload"""
        self._message.text = '✗ File type or size not allowed'
        self._message.classes('text-red-600')
    
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
