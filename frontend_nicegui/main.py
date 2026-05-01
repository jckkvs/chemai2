"""
ChemAI2 - NiceGUI Frontend
アプリケーション仕様書に基づく意思決定支援アプリケーション
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from nicegui import ui, run
except ImportError:
    print("NiceGUI not installed. Please install with: pip install nicegui")
    sys.exit(1)

# Import pages
from frontend_nicegui.pages import (
    page_welcome,
    page_data_upload,
    page_llm_interview,
    page_preprocessing,
    page_eda,
    page_decision_support,
    page_ml_modeling,
    page_doe,
    page_results,
    page_settings,
)

# Import utils
from frontend_nicegui.utils import app_state

# Page routing
PAGES = {
    'welcome': page_welcome,
    'data_upload': page_data_upload,
    'llm_interview': page_llm_interview,
    'preprocessing': page_preprocessing,
    'eda': page_eda,
    'decision': page_decision_support,
    'ml_modeling': page_ml_modeling,
    'doe': page_doe,
    'results': page_results,
    'settings': page_settings,
}


def create_navigation_drawer() -> None:
    """Create the main navigation drawer"""
    with ui.left_drawer(top_corner=True, bottom_corner=True).props('bordered') as drawer:
        drawer.classes('bg-gray-900 text-white')

        with ui.column().classes('w-full gap-2 p-4'):
            ui.label('ChemAI2').classes('text-2xl font-bold text-white mb-4')

            # Beginner mode toggle
            with ui.row().classes('w-full items-center justify-between mb-4'):
                ui.label('初心者モード').classes('text-sm text-gray-300')
                beginner_toggle = ui.switch(value=app_state.beginner_mode)
                beginner_toggle.classes('text-white')
                from frontend_nicegui.utils.beginner_mode import beginner_mode_toggle_changed
                beginner_toggle.on_value_change(
                    lambda e: beginner_mode_toggle_changed(e.value)
                )

            nav_items = [
                ('welcome', '🏠 ホーム', 'welcome'),
                ('data_upload', '📊 データを入れる', 'data'),
                ('decision', '🎯 次はどう動く？', 'decision'),
            ]

            for page_id, label, icon in nav_items:
                btn = ui.button(
                    label,
                    icon=icon,
                    on_click=lambda _, pid=page_id: app_state.navigate_to(pid)
                ).classes('w-full justify-start text-left').props('no-caps')

                if page_id == app_state.current_page:
                    btn.classes('bg-blue-600', remove='bg-transparent')
                else:
                    btn.classes('bg-transparent hover:bg-gray-800')

    return drawer


def get_current_step() -> int:
    """Get current step number (1-5) based on app_state"""
    if not app_state.data_loaded:
        return 1
    if app_state.target_column is None:
        return 2
    # Check if EDA is done (has viewed EDA page)
    if app_state.current_page in ['eda', 'llm_interview', 'decision', 'ml_modeling', 'doe', 'results']:
        return 3
    if app_state.current_page in ['llm_interview', 'decision', 'ml_modeling', 'doe', 'results']:
        return 4
    if app_state.current_page in ['decision', 'ml_modeling', 'doe', 'results']:
        return 5
    return 2


def render_progress_bar() -> None:
    """Render progress bar showing current step"""
    steps = [
        ('❶', 'データを入れる', app_state.data_loaded),
        ('❷', '目標を決める', app_state.target_column is not None),
        ('❸', 'データを眺める', app_state.current_page in ['eda', 'llm_interview', 'decision']),
        ('❹', '条件を提案させる', app_state.current_page in ['decision', 'ml_modeling', 'doe']),
        ('❺', '結果を確認する', app_state.current_page in ['results']),
    ]

    current_step = get_current_step()

    with ui.card().classes('w-full bg-gray-800 p-3 mb-4').style('border: 1px solid #374151;'):
        with ui.row().classes('w-full items-center gap-2'):
            ui.label(f'ステップ {current_step}/5').classes('text-xs text-gray-400 mr-4')

            for idx, (icon, label, done) in enumerate(steps, 1):
                if idx == current_step:
                    ui.label(f'{icon} {label}').classes('text-sm text-blue-300 font-bold')
                elif done:
                    ui.label(f'{icon} {label} ✅').classes('text-sm text-green-400')
                else:
                    ui.label(f'{icon} {label}').classes('text-sm text-gray-600')

                if idx < len(steps):
                    ui.label('→').classes('text-gray-600')


def build_page(page_id: str, page_func) -> None:
    """Build a page with common layout (drawer + top bar + content)"""
    drawer = create_navigation_drawer()

    with ui.column().classes('w-full min-h-screen').style('background-color: #111827;'):
        # Progress bar
        render_progress_bar()

        # Top bar
        with ui.row().classes('w-full bg-gray-900 p-4 items-center shadow-lg'):
            ui.button(icon='menu', on_click=lambda: drawer.toggle()).props('flat color=white')
            ui.label('ChemAI2').classes('text-xl font-bold text-white')
            ui.space()
            ui.button(icon='settings', on_click=lambda: app_state.navigate_to('settings')).props('flat color=white')

        # Page content
        page_func()


def main() -> None:
    """Main entry point - register all page routes with shared layout"""

    # Register a route for each page with shared layout
    for page_id, page_func in PAGES.items():
        # Create a closure that captures page_id and page_func
        def make_handler(pid, func):
            def handler():
                app_state.current_page = pid
                build_page(pid, func)
            return handler
        ui.page(f'/{page_id}')(make_handler(page_id, page_func))

    # Default route
    @ui.page('/')
    def index():
        page_func = PAGES.get(app_state.current_page, page_welcome)
        build_page(app_state.current_page, page_func)

    # Run the app
    run(
        title='ChemAI2 - 意思決定支援',
        host='0.0.0.0',
        port=8080,
        reload=False,
    )


if __name__ == '__main__':
    main()
