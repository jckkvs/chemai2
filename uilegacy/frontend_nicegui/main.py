"""
frontend_nicegui/main.py - ChemAI2 Decision Support Application
Based on アプリケーション仕様書.md (2026-04-30)

Core flow: Data Upload → LLM Interview → Preprocessing → EDA → Decision Support → (ML/DOE as needed)
"""
from nicegui import ui, app
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure app
app.title = 'ChemAI2 - Decision Support Platform'
app.favicon = '🧪'

# Shared state for the application
state: Dict[str, Any] = {
    # Data
    'df': None,
    'filename': None,
    'target_col': None,
    'smiles_col': None,
    'task_type': 'regression',
    'exclude_cols': [],
    'group_col': None,
    'time_col': None,
    # Domain knowledge
    'domain_knowledge': {},
    'user_constraints': [],
    # Features
    'selected_descriptors': [],
    'precalc_done': False,
    'precalc_df': None,
    # Pipeline
    'cv_key': 'auto',
    'cv_folds': 5,
    'num_scaler': 'standard',
    'num_imputer': 'median',
    'cat_encoder': 'onehot',
    'selected_models': [],
    'model_params': {},
    # Monotonicity
    'monotonicity_constraints': {
        '_global': {'default_direction': 'none', 'default_strength': 0.5, 'default_sigma': 3.0},
        '_by_feature': {},
    },
    # Flags
    'do_eda': True,
    'do_prep': True,
    'do_eval': True,
    # Results
    'automl_result': None,
    'decision_report': None,
}

# Page instances
pages: Dict[str, Optional[object]] = {}


@ui.page('/')
def main_page():
    """Main page with decision-support oriented workflow."""

    # Initialize page instances
    # LLM config (shared)
    llm_config = None
    try:
        from frontend_nicegui.components.llm_config_dialog import LLMConfigDialog
        llm_config = type('LLMConfig', (), {
            'mode': 'prompt_only',
            'api_endpoint': None,
            'api_key': None,
            'model_name': 'default',
            'temperature': 0.1,
            'max_tokens': 2000,
            'enable_code_execution': False,
        })()
    except Exception:
        pass

    # EDA page
    try:
        from frontend_nicegui.pages.eda_page import EDAPage
        pages['eda'] = EDAPage()
    except Exception as e:
        print(f"Failed to import EDAPage: {e}")
        pages['eda'] = None

    # Decision Support page
    try:
        from frontend_nicegui.pages.decision_support_page import DecisionSupportPage
        pages['decision'] = DecisionSupportPage()
    except Exception as e:
        print(f"Failed to import DecisionSupportPage: {e}")
        pages['decision'] = None

    # Preprocessing page
    try:
        from frontend_nicegui.pages.preprocessing_page import PreprocessingPage
        pages['preprocessing'] = PreprocessingPage()
    except Exception as e:
        print(f"Failed to import PreprocessingPage: {e}")
        pages['preprocessing'] = None

    # DOE page
    try:
        from frontend_nicegui.pages.doe_page import DOEPPage
        pages['doe'] = DOEPPage()
    except Exception as e:
        print(f"Failed to import DOEPPage: {e}")
        pages['doe'] = None

    # Results page
    try:
        from frontend_nicegui.pages.results_page import ResultsPage
        pages['results'] = ResultsPage()
    except Exception as e:
        print(f"Failed to import ResultsPage: {e}")
        pages['results'] = None

    # Settings page
    try:
        from frontend_nicegui.pages.settings_page import SettingsPage
        pages['settings'] = SettingsPage()
    except Exception as e:
        print(f"Failed to import SettingsPage: {e}")
        pages['settings'] = None

    # Connect data upload to EDA/Decision pages
    if pages.get('data_upload') and pages.get('eda'):
        pass  # Will connect via shared state

    # Import and instantiate page classes
    # Data Upload
    try:
        from frontend_nicegui.pages.data_upload_tab import DataUploadPage
        pages['data_upload'] = DataUploadPage(llm_config=llm_config)
    except Exception as e:
        print(f"Failed to import DataUploadPage: {e}")
        pages['data_upload'] = None

    # Visualization (shared between tabs)
    viz_page = None
    try:
        from frontend_nicegui.pages.visualization_tab import VisualizationPage
        viz_page = VisualizationPage()
        pages['visualization'] = viz_page
    except Exception as e:
        print(f"Failed to import VisualizationPage: {e}")
        pages['visualization'] = None

    # AutoML
    automl_page = None
    try:
        from frontend_nicegui.pages.automl_page import AutoMLPage
        automl_page = AutoMLPage(viz_page=viz_page)
        pages['automl'] = automl_page
    except Exception as e:
        print(f"Failed to import AutoMLPage: {e}")
        pages['automl'] = None

    # Connect data upload to other pages
    if pages.get('data_upload'):
        pages['data_upload'].automl_page = automl_page
        pages['data_upload'].viz_page = viz_page

    # LLM Assistant
    try:
        from frontend_nicegui.pages.llm_assistant_tab import LLMAssistantPage
        pages['llm_assistant'] = LLMAssistantPage(llm_config=llm_config, llm_dialog=None)
    except Exception as e:
        print(f"Failed to import LLMAssistantPage: {e}")
        pages['llm_assistant'] = None

    # Interview page
    try:
        from frontend_nicegui.pages.interview_page import InterviewPage
        pages['interview'] = InterviewPage()
    except Exception as e:
        print(f"Failed to import InterviewPage: {e}")
        pages['interview'] = None

    # Header
    with ui.header().classes('bg-white shadow-sm h-16'):
        with ui.row().classes('w-full justify-between items-center px-4'):
            with ui.row().classes('items-center gap-2'):
                ui.label('🧪 ChemAI2').classes('text-xl font-bold text-primary')
                ui.label('Decision Support Platform').classes('text-sm text-gray-500')
            with ui.row().classes('gap-2'):
                ui.button(icon='help', on_click=lambda: ui.navigate.to('/help')).props('flat dense')

    # Main tabs based on specification 10.2
    with ui.tabs().classes('w-full bg-white shadow-sm') as tabs:
        welcome_tab = ui.tab('Welcome', icon='home')
        data_tab = ui.tab('Data Upload', icon='cloud_upload')
        interview_tab = ui.tab('LLM Interview', icon='psychology')
        preprocessing_tab = ui.tab('Preprocessing', icon='tune')
        eda_tab = ui.tab('EDA', icon='insights')
        decision_tab = ui.tab('Decision Support', icon='gavel')
        ml_tab = ui.tab('ML Modeling', icon='model_training')
        doe_tab = ui.tab('DOE', icon='science')
        results_tab = ui.tab('Results', icon='description')
        settings_tab = ui.tab('Settings', icon='settings')

    # Tab panels
    with ui.tab_panels(tabs, value=welcome_tab).classes('w-full max-w-7xl mx-auto mt-4') as tab_panels:

        # Welcome tab
        with ui.tab_panel(welcome_tab):
            render_welcome()

        # Data Upload tab
        with ui.tab_panel(data_tab):
            if pages.get('data_upload'):
                pages['data_upload'].render()
            else:
                ui.label('Data Upload page failed to load').classes('text-red-500')

        # LLM Interview tab
        with ui.tab_panel(interview_tab):
            if pages.get('interview'):
                pages['interview'].render()
            elif pages.get('llm_assistant'):
                pages['llm_assistant'].render()
            else:
                ui.label('LLM Interview page failed to load').classes('text-red-500')

        # Preprocessing tab
        with ui.tab_panel(preprocessing_tab):
            if pages.get('preprocessing'):
                pages['preprocessing'].render()
            else:
                render_preprocessing()

        # EDA tab
        with ui.tab_panel(eda_tab):
            if pages.get('eda'):
                pages['eda'].render()
            else:
                render_eda()

        # Decision Support tab (core)
        with ui.tab_panel(decision_tab):
            if pages.get('decision'):
                pages['decision'].render()
            else:
                render_decision_support()

        # ML Modeling tab
        with ui.tab_panel(ml_tab):
            if pages.get('automl'):
                pages['automl'].render()
            else:
                ui.label('ML Modeling page failed to load').classes('text-red-500')

        # DOE tab
        with ui.tab_panel(doe_tab):
            if pages.get('doe'):
                pages['doe'].render()
            else:
                render_doe()

        # Results tab
        with ui.tab_panel(results_tab):
            if pages.get('results'):
                pages['results'].render()
            else:
                render_results()

        # Settings tab
        with ui.tab_panel(settings_tab):
            if pages.get('settings'):
                pages['settings'].render()
            else:
                render_settings()

    # Footer
    with ui.footer().classes('bg-gray-100'):
        with ui.row().classes('w-full justify-center py-2 gap-2'):
            ui.label('ChemAI2 - Materials Informatics Decision Support').classes('text-xs text-gray-600')


def render_welcome():
    """Welcome/guide page shown on first visit."""
    with ui.card().classes('w-full max-w-3xl mx-auto mt-8'):
        ui.label('🧪 Welcome to ChemAI2').classes('text-2xl font-bold text-primary mb-4')

        ui.markdown("""
        ### Core Philosophy: Decision Support over Analysis

        This platform is designed to help you **make decisions**, not just analyze data.

        **Your Workflow:**
        1. **Data Upload** → Upload your SMILES/table data
        2. **LLM Interview** → Define your decision goals and share domain knowledge
        3. **Preprocessing** → Select features and configure preprocessing
        4. **EDA** → Read and understand your data patterns
        5. **Decision Support** → Get actionable recommendations

        **Four Core Decision Scenarios:**
        - 🎯 Next experiment proposal
        - 🎯 Target achievement feasibility
        - 🎯 Data sufficiency judgment
        - 🎯 Theme withdrawal decision
        """)

        with ui.row().classes('w-full justify-center mt-4'):
            ui.button('Get Started →', on_click=lambda: ui.navigate.to('/#data')).props('color=primary size=lg')


def render_preprocessing():
    """Preprocessing and feature selection page."""
    ui.label('Preprocessing & Feature Selection').classes('text-2xl font-bold mb-4')

    ui.markdown("""
    Configure preprocessing and select features for your analysis.

    - Feature selection based on domain knowledge
    - Preprocessing configuration (scaling, imputation, encoding)
    - Correlation analysis and feature importance
    """)

    ui.label('Preprocessing functionality - to be implemented').classes('text-gray-500')


def render_eda():
    """EDA page - where users read and understand their data."""
    ui.label('Exploratory Data Analysis').classes('text-2xl font-bold mb-4')

    ui.markdown("""
    **Read your data thoroughly before proceeding to modeling.**

    - Basic statistics and visualization
    - Interactive filtering (slider, dropdown)
    - Data drill-down and annotation
    - SMILES hover and structure visualization
    - LLM-guided EDA reading support
    """)

    # Link to visualization page if available
    if pages.get('visualization'):
        ui.button('Open Visualization', on_click=lambda: ui.navigate.to('/#eda')).props('outline')
    else:
        ui.label('EDA functionality - to be implemented').classes('text-gray-500')


def render_decision_support():
    """Decision Support page - THE CORE of the application."""
    ui.label('🎯 Decision Support').classes('text-2xl font-bold mb-4')

    ui.markdown("""
    ### Core Decision Support

    Based on your data and goals, the system will help you decide:

    **① Next Experiment Proposal**
    - What experiments should you run next?
    - Optimal conditions for your target

    **② Target Achievement Feasibility**
    - Can you achieve your target value?
    - Realistic target value suggestions

    **③ Data Sufficiency Judgment**
    - Is current data enough for reliable prediction?
    - What additional data is needed?

    **④ Theme Withdrawal Decision**
    - Should you continue or withdraw from this theme?
    - Direction change suggestions
    """)

    ui.label('Decision Support functionality - to be implemented').classes('text-gray-500')


def render_doe():
    """Design of Experiments page - used when data is insufficient."""
    ui.label('Design of Experiments (DOE)').classes('text-2xl font-bold mb-4')

    ui.markdown("""
    Experimental design for data augmentation.

    *This page is activated only when Decision Support determines data is insufficient.*
    """)

    ui.label('DOE functionality - to be implemented').classes('text-gray-500')


def render_results():
    """Results and Decision Report page."""
    ui.label('Results & Decision Report').classes('text-2xl font-bold mb-4')

    ui.markdown("""
    View decision reports and experiment proposals.

    - Decision report generation
    - Next experiment list (CSV/Excel export)
    - Executive summary for meetings
    """)

    ui.label('Results functionality - to be implemented').classes('text-gray-500')


def render_settings():
    """Settings page - LLM config, domain knowledge management."""
    ui.label('Settings').classes('text-2xl font-bold mb-4')

    ui.markdown("""
    - LLM provider selection
    - Domain knowledge management
    - UMA weight path configuration
    - Feature engine settings
    """)

    # LLM Config button
    try:
        from frontend_nicegui.components.llm_config_dialog import LLMConfigDialog
        dialog = LLMConfigDialog()
        dialog.create_trigger_button(ui.row(), label='LLM設定')
    except Exception:
        ui.label('Settings functionality - to be implemented').classes('text-gray-500')


@ui.page('/help')
def help_page():
    """Help page."""
    with ui.header().classes('items-center'):
        ui.link('← Back', '/').classes('text-white q-mr-md')
        ui.label('❓ Help - ChemAI2').classes('text-h6')

    ui.markdown("""
    ## ChemAI2 Decision Support Platform

    ### Core Flow
    1. **Data Upload**: Upload your SMILES/tabular data
    2. **LLM Interview**: Set decision goals, share domain knowledge
    3. **Preprocessing**: Feature selection and preprocessing
    4. **EDA**: Read and understand data patterns
    5. **Decision Support**: Get actionable recommendations

    ### Key Features
    - **LLM-assisted**: AI helps throughout the workflow
    - **Decision-oriented**: Analysis is a means, decision is the goal
    - **Domain knowledge integration**: Your expertise is fully utilized
    - **Backend asset reuse**: All backend/ assets are leveraged
    """)


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        title='ChemAI2 - Decision Support Platform',
        host='0.0.0.0',
        port=8080,
        reload=False,
        storage_secret='chemai2_secret_key_2026'
    )
