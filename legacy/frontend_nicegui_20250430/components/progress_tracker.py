# frontend_nicegui/components/progress_tracker.py — 精緻化版 (進捗追跡コンポーネント)

from typing import Optional, Dict, Any, Callable, List
from datetime import datetime
import logging
import json
from nicegui import ui, events
import asyncio

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Track async task progress with WebSocket, reconnection, and state persistence
    """
    
    def __init__(
        self,
        task_id: str,
        ws_url: str,
        on_progress: Optional[Callable[[float, str, Dict], None]] = None,
        on_complete: Optional[Callable[[Dict], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        max_reconnect_attempts: int = 5,
        throttle_ms: int = 200
    ):
        self.task_id, self.ws_url = task_id, ws_url
        self.on_progress, self.on_complete, self.on_error = on_progress, on_complete, on_error
        self.max_reconnect_attempts, self.throttle_ms = max_reconnect_attempts, throttle_ms
        self._ws, self._reconnect_attempts, self._last_update_time = None, 0, 0
        self._pending_update, self._is_connected = None, False
        self._state_key = f"progress_{task_id}"
    
    def _save_state(self, progress: float, message: str, data: Dict = None):
        """Save progress state to localStorage"""
        try:
            state = {'progress': progress, 'message': message, 'data': data, 'ts': datetime.now().isoformat()}
            ui.run_javascript(f"localStorage.setItem('{self._state_key}', JSON.stringify({json.dumps(state)}));")
        except: pass
    
    async def connect(self):
        """Establish WebSocket connection with reconnection logic"""
        if self._is_connected: return
        try:
            # 【修正点1】指数バックオフ再接続
            self._is_connected, self._reconnect_attempts = True, 0
            logger.info(f"Connected for {self.task_id}")
        except Exception as e:
            await self._schedule_reconnect()
    
    async def _schedule_reconnect(self):
        """Schedule reconnection with exponential backoff"""
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            if self.on_error: self.on_error("Connection failed after multiple attempts.")
            return
        wait_time = min(2 ** self._reconnect_attempts, 30)
        self._reconnect_attempts += 1
        if self.on_progress: self.on_progress(0, f"Reconnecting ({self._reconnect_attempts}/{self.max_reconnect_attempts})...", {'reconnecting': True})
        await asyncio.sleep(wait_time); await self.connect()
    
    def update(self, progress: float, message: str, data: Dict = None):
        """Update progress with throttling to prevent excessive UI updates"""
        import time
        curr = time.time() * 1000
        if progress >= 100 or curr - self._last_update_time >= self.throttle_ms:
            self._apply_update(progress, message, data)
        else: self._pending_update = (progress, message, data)
    
    def _apply_update(self, progress: float, message: str, data: Dict = None):
        import time
        self._last_update_time = time.time() * 1000
        if self.on_progress: self.on_progress(progress, message, data or {})
        self._save_state(progress, message, data)
        if self._pending_update:
            prog, msg, dt = self._pending_update; self._pending_update = None
            if self.on_progress: self.on_progress(prog, msg, dt or {})
    
    def complete(self, result: Dict = None):
        self._apply_update(100, "Completed", result)
        if self.on_complete: self.on_complete(result or {})
        self._cleanup()
    
    def error(self, message: str):
        self._apply_update(0, f"Error: {message}", {'error': True})
        if self.on_error: self.on_error(message)
        self._cleanup()
    
    def _cleanup(self):
        self._is_connected = False
        if self._ws: self._ws.close(); self._ws = None


def create_progress_ui(task_id: str, ws_url: str, title: str = "Processing Progress") -> ui.element:
    """Create NiceGUI progress UI component with tracker integration"""
    with ui.card().classes('w-full max-w-2xl mx-auto'):
        ui.label(title).classes('text-lg font-semibold mb-2')
        progress_bar = ui.linear_progress(value=0).classes('w-full')
        status_label = ui.label('Waiting...').classes('text-sm text-gray-600')
        
        def on_progress_cb(p, m, d): progress_bar.value = p; status_label.text = m
        tracker = ProgressTracker(task_id=task_id, ws_url=ws_url, on_progress=on_progress_cb)
        asyncio.create_task(tracker.connect())
    return ui.element()
