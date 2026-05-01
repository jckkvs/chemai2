"""
Utilities for ChemAI2 frontend
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / 'backend'
if backend_path.exists():
    sys.path.insert(0, str(backend_path.parent))

# State management
class AppState:
    """Global application state"""
    def __init__(self):
        self.current_page: str = 'welcome'
        self.data_loaded: bool = False
        self.data_df = None
        self.data_path: str = ""
        self.target_column: str = None
        self.feature_columns: list = []
        self.model_trained: bool = False
        self.model_results: dict = {}
        self.llm_config: dict = {
            'provider': 'local',
            'model': '',
            'api_key': '',
            'api_base': '',
        }
        self.experiment_design: dict = {}
        self.selected_features: list = []

        # Decision support state
        self.data_sufficiency: dict = {}
        self.achievement_prob: float = 0.0
        self.recommended_actions: list = []

        # Beginner mode
        self.beginner_mode: bool = False

        # Domain knowledge is now managed by DomainKnowledgeManager
        # self.domain_knowledge is accessed via domain_knowledge global instance

    def reset(self):
        """Reset state for new project"""
        self.__init__()

    def navigate_to(self, page: str) -> None:
        """Navigate to a specific page using NiceGUI routing"""
        self.current_page = page
        from nicegui import ui
        ui.navigate.to(f'/{page}')

# Global state instance
app_state = AppState()

def format_number(num: float, precision: int = 4) -> str:
    """Format number with given precision"""
    return f"{num:.{precision}f}"

def notify(message: str, type: str = 'info'):
    """Show notification"""
    from nicegui import ui
    ui.notify(message, type=type)
