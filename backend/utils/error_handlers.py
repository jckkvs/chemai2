"""
backend/utils/error_handlers.py
安全な実行のためのデコレータとユーティリティ。
"""
import functools
import traceback
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

def safe_execute(func: Callable) -> Callable:
    """
    安全な関数実行デコレータ。
    例外をキャッチし、ログに詳細を記録し、ユーザーに通知（NiceGUIが利用可能な場合）します。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"実行エラー ({func.__name__}): {str(e)}"
            error_detail = traceback.format_exc()
            
            logger.error(f"❌ {error_msg}")
            logger.error(error_detail)
            
            # NiceGUI の通知を試みる（実行環境に NiceGUI があり、ループが回っている場合）
            try:
                from nicegui import ui
                ui.notify(error_msg, type="negative", timeout=10000)
            except Exception:
                # NiceGUI がコンテキスト外、または未インストールの場合はスキップ
                pass
            
            # None を返して graceful degradation
            return None
    return wrapper
