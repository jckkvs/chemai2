from nicegui import ui
from frontend_nicegui.utils import app_state
from pathlib import Path
import json

HISTORY_FILE = Path(__file__).parent.parent.parent / "test_results_history.json"


def get_latest_test_summary() -> dict:
    """Get latest test result summary for display."""
    if not HISTORY_FILE.exists():
        return None
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        if history:
            return history[-1]
    except (json.JSONDecodeError, IOError):
        pass
    return None


def page_welcome() -> None:
    """Welcome page - just the start button, nothing else"""
    with ui.column().classes('w-full min-h-screen items-center justify-center gap-8').style('background-color: #111827;'):
        # Title
        with ui.column().classes('items-center gap-2'):
            ui.label('🧪 ChemAI2').classes('text-5xl font-bold text-white')
            ui.label('材料開発を支援します').classes('text-xl text-gray-300')

        # Just one big action
        ui.button(
            'データを入れる',
            icon='upload',
            on_click=lambda: app_state.navigate_to('data_upload')
        ).props('size=lg color=primary').classes('px-12 py-6 text-xl')

        # Test result summary at bottom
        summary = get_latest_test_summary()
        if summary:
            with ui.card().classes('bg-gray-800 p-4 mt-8').style('border: 1px solid #374151;'):
                with ui.row().classes('items-center gap-4'):
                    ui.label('📊 最新テスト結果:').classes('text-sm text-gray-400')
                    ui.label(f"✅ {summary['passed']} 成功").classes('text-sm text-green-400')
                    ui.label(f"❌ {summary['failed']} 失敗").classes('text-sm text-red-400')
                    ui.label(f"⚠️ {summary['errors']} エラー").classes('text-sm text-yellow-400')
                    ui.label(f"⏭ {summary['skipped']} スキップ").classes('text-sm text-gray-500')
                    ui.button('詳細グラフ', icon='bar_chart',
                            on_click=lambda: app_state.navigate_to('settings')
                            ).props('flat color=gray size=sm dense')
