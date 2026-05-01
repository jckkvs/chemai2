"""
frontend_nicegui/main.py
ChemAI MI Studio — LLM-assisted workflow + full SMILES/EDA/Inverse features from chemai2_qwen.
Base: chemai2_cc (LLM Interview/Assistant) + chemai2_qwen (SMILES features, EDA, Inverse, etc.)
"""
from nicegui import ui, app
import sys
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@ui.page('/')
def main_page():
    """Main page with LLM-assisted workflow and full feature integration."""

    # ── Shared state dictionary (minimal set from chemai2_qwen) ──
    state = {
        # Data
        "df": None,
        "filename": None,
        "target_col": None,
        "smiles_col": None,
        "task_type": "regression",
        "exclude_cols": [],
        "group_col": None,
        "time_col": None,
        "weight_col": None,
        # SMILES descriptors
        "precalc_df": None,
        "precalc_done": False,
        "selected_descriptors": [],
        "calc_summary": {},
        "_applied_recommendation": None,
        # Pipeline: CV
        "cv_key": "auto",
        "cv_folds": 5,
        "timeout": 300,
        # Pipeline: preprocessing
        "num_scaler": "standard",
        "num_imputer": "median",
        "num_transform": "none",
        "cat_encoder": "onehot",
        "cat_imputer": "most_frequent",
        # Pipeline: feature engineering
        "do_polynomial": False,
        "feature_selector": "none",
        # Pipeline: models
        "selected_models": [],
        "model_params": {},
        # Monotonicity constraints
        "monotonicity_constraints": {
            "_global": {"default_direction": "none", "default_strength": 0.5, "default_sigma": 3.0, "apply_to_new_features": True},
            "_by_feature": {},
            "_by_set": {}
        },
        "feature_classification": {},
        "feature_stats": {},
        "monotonic_constraints": {},
        # Pipeline: flags
        "do_eda": True,
        "do_prep": True,
        "do_eval": True,
        "do_pca": True,
        "do_shap": True,
        # Results
        "automl_result": None,
        "pipeline_result": None,
        "metric_evaluator": None,
        "metric_cache": {},
        "available_categories": [],
    }

    # ── LLM config ──
    # ── Tab change handler ──
    def _on_tab_change(e):
        """Handle tab value change - refresh containers as needed."""
        tab_val = getattr(e, 'value', None) or str(e)

        # Refresh containers based on tab value
        refresh_map = {
            "eda":        "_refresh_eda",
            "results":    "_refresh_results",
            "inverse":    "_refresh_inverse",
            "llm":        "_refresh_llm_report",
            "export":     "_refresh_export",
            "comparison": "_refresh_experiment_comparison",
            "mixture":    "_refresh_mixture",
            "computation": "_refresh_computation",
            "quantum":     "_refresh_quantum",
        }

        if tab_val in refresh_map:
            key = refresh_map[tab_val]
            fn = state.get(key)
            if fn:
                try:
                    fn()
                except Exception as exc:
                    logger.warning(f"Tab refresh {key} failed: {exc}")

        # Also handle interview tab navigation
        if tab_val == "interview":
            pass  # No refresh needed, handled by set_data

    tab_panels.on_value_change(_on_tab_change)

    from backend.config.llm_settings import LLMConfig
    from backend.llm.config import LLMSettings, load_settings, save_settings
    from frontend_nicegui.components.llm_config_dialog import LLMConfigDialog

    llm_config = LLMConfig.load().get_effective_config()
    # デフォルトモードが 'local' になっていることを確認
    if not hasattr(llm_config, 'mode') or not llm_config.mode:
        llm_config.mode = 'local'

    # LLMConfig.mode と LLMSettings.operation_mode を同期
    try:
        llm_settings = load_settings()
        if llm_config.mode:
            llm_settings.operation_mode = llm_config.mode
            save_settings(llm_settings)
    except Exception as e:
        logger.warning(f"LLMSettings同期スキップ: {e}")

    llm_dialog = LLMConfigDialog(config=llm_config, on_config_change=lambda c: c.save())

    # ── Page instances (chemai2_cc originals) ──
    from frontend_nicegui.pages.visualization_tab import VisualizationPage
    from frontend_nicegui.pages.automl_page import AutoMLPage
    from frontend_nicegui.pages.doe_page import DoEPage
    from frontend_nicegui.pages.interview_page import InterviewPage
    from frontend_nicegui.pages.llm_assistant_tab import LLMAssistantPage

    viz_page = VisualizationPage()
    automl_page = AutoMLPage(viz_page=viz_page)
    doe_page = DoEPage()
    interview_page = InterviewPage(
        navigate_to_automl=lambda: tab_panels.set_value(auto_ml_tab),
        navigate_to_doe=lambda: tab_panels.set_value(doe_tab),
        navigate_to_llm=lambda: tab_panels.set_value(llm_tab),
    )
    assistant_page = LLMAssistantPage(llm_config=llm_config, llm_dialog=llm_dialog, state=state)

    # ── Header ──
    with ui.header().classes('bg-white shadow-sm'):
        with ui.row().classes('w-full justify-between items-center px-4'):
            ui.label('ChemAI MI Studio').classes('text-xl font-bold text-primary')
            with ui.row().classes('gap-2'):
                ui.label('v1.0.0').classes('text-xs text-gray-500')
                llm_dialog.create_trigger_button(ui.row(), label='⚙️', icon='settings')

    # ── Tab creation (extended) ──
    with ui.tabs().classes('w-full bg-white shadow-sm') as t:
        data_tab = ui.tab('Data & SMILES', icon='cloud_upload')
        interview_tab = ui.tab('LLM Interview', icon='psychology')
        eda_tab = ui.tab('EDA', icon='insights')
        auto_ml_tab = ui.tab('AutoML & Results', icon='model_training')
        llm_tab = ui.tab('LLM Assistant', icon='auto_awesome')
        inverse_tab = ui.tab('Inverse', icon='find_replace')
        settings_tab = ui.tab('Settings', icon='tune')
        doe_tab = ui.tab('DOE', icon='science')

    # ── Tab panels ──
    with ui.tab_panels(t, value=data_tab).classes('w-full max-w-7xl mx-auto mt-4') as tab_panels:


        # ── Data & SMILES Tab (integrated from chemai2_qwen) ──
        with ui.tab_panel(data_tab):
            # Original DataUploadPage for file upload
            def go_to_interview():
                if data_page and data_page.uploaded_data is not None:
                    interview_page.set_data(data_page.uploaded_data)
                tab_panels.set_value(interview_tab)

            def go_to_automl():
                tab_panels.set_value(auto_ml_tab)

            data_page = DataUploadPage(
                llm_config=llm_config,
                automl_page=automl_page,
                viz_page=viz_page,
                navigate_to_automl=go_to_automl,
                navigate_to_interview=go_to_interview,
            )
            data_page.render()

            # SMILES descriptor selection (from chemai2_qwen data_tab.py)
            ui.separator().classes('my-4')
            ui.label('⚗️ SMILES Features & Column Roles').classes('text-lg font-bold mb-2')

            try:
                from frontend_nicegui.components.data_tab import render_data_tab
                render_data_tab(state)
            except Exception as e:
                logger.warning(f"render_data_tab failed: {e}")
                # Fallback: simplified descriptor plugin UI
                with ui.expansion('⚗️ SMILES Feature Selection', icon='science').classes('w-full'):
                    try:
                        from frontend_nicegui.components.descriptor_plugins_ui import render_descriptor_plugins
                        render_descriptor_plugins(state)
                    except Exception as e2:
                        logger.warning(f"descriptor_plugins failed: {e2}")
                        ui.label('SMILES descriptor UI requires data with SMILES column.').classes('text-gray-500')

        # ── LLM Interview Tab (extended) ──
        with ui.tab_panel(interview_tab):
            interview_page.render()

        # ── EDA Tab (from chemai2_qwen) ──
        with ui.tab_panel(eda_tab):
            _eda_container = ui.column().classes('w-full')
            state["_eda_container"] = _eda_container

            def _build_eda():
                _eda_container.clear()
                with _eda_container:
                    try:
                        from frontend_nicegui.components.eda_panel import render_eda_panel
                        render_eda_panel(state)
                    except Exception as e:
                        logger.warning(f"EDA panel failed: {e}")
                        ui.label('EDA panel requires data to be loaded first.').classes('text-gray-500')

            _build_eda()
            state["_refresh_eda"] = _build_eda

        # ── AutoML & Results Tab ──
        with ui.tab_panel(auto_ml_tab):
            automl_page.render()

            ui.separator().classes('my-4')
            ui.label('📊 Results').classes('text-lg font-bold mb-2')

            _results_container = ui.column().classes('w-full')
            state["_results_container"] = _results_container

            def _build_results():
                _results_container.clear()
                with _results_container:
                    try:
                        from frontend_nicegui.components.results_tab import render_results_tab
                        render_results_tab(state)
                    except Exception as e:
                        logger.warning(f"Results tab failed: {e}")
                        ui.label('Results will appear after AutoML completes.').classes('text-gray-500')

            _build_results()
            state["_refresh_results"] = _build_results

        # ── LLM Assistant Tab (container pattern) ──
        with ui.tab_panel(llm_tab):
            assistant_page.render()

            # Report & Interpretation container (lazy loading)
            _llm_report_container = ui.column().classes('w-full')
            state["_llm_report_container"] = _llm_report_container

            def _build_llm_report():
                _llm_report_container.clear()
                with _llm_report_container:
                    # Report Generation
                    ui.separator().classes('my-4')
                    ui.label('📝 Report Generation').classes('text-lg font-bold mb-2')
                    try:
                        from frontend_nicegui.components.report_generator import render_report_tab
                        render_report_tab(state)
                    except Exception as e:
                        logger.warning(f"Report generator failed: {e}")
                        ui.label('Report generation requires completed analysis.').classes('text-gray-500')

                    # Analysis Explanation
                    ui.separator().classes('my-4')
                    ui.label('🔍 Analysis Explanation').classes('text-lg font-bold mb-2')
                    try:
                        from frontend_nicegui.components.interpretation_panel import render_interpretation_panel
                        automl_result = state.get("automl_result")
                        if automl_result:
                            render_interpretation_panel(automl_result, state)
                        else:
                            ui.label('Analysis explanation will appear after AutoML completes.').classes('text-gray-500')
                    except Exception as e:
                        logger.warning(f"Interpretation panel failed: {e}")
                        ui.label('Analysis explanation module not available.').classes('text-gray-500')

            _build_llm_report()  # Initial render
            state["_refresh_llm_report"] = _build_llm_report

        # ── Inverse Analysis Tab (from chemai2_qwen) ──
        with ui.tab_panel(inverse_tab):
            _inverse_container = ui.column().classes('w-full')
            state["_inverse_container"] = _inverse_container

            def _build_inverse():
                _inverse_container.clear()
                with _inverse_container:
                    try:
                        from frontend_nicegui.components.inverse_tab import render_inverse_panel
                        render_inverse_panel(state)
                    except Exception as e:
                        logger.warning(f"Inverse tab failed: {e}")
                        ui.label('Inverse analysis requires completed model.').classes('text-gray-500')

            _build_inverse()
            state["_refresh_inverse"] = _build_inverse

        # ── Settings Tab (from chemai2_qwen) ──
        with ui.tab_panel(settings_tab):
            # Settings checker
            try:
                from frontend_nicegui.components.settings_checker import render_settings_checker
                render_settings_checker(state)
            except Exception:
                pass

            ui.separator().classes('q-my-sm')

            # Leakage check
            try:
                from frontend_nicegui.components.leakage_check_ui import render_leakage_check_panel
                render_leakage_check_panel(state)
            except Exception:
                pass

            ui.separator().classes('q-my-sm')

            # CV config
            try:
                from frontend_nicegui.components.cv_config_ui import render_cv_config
                render_cv_config(state)
            except Exception:
                pass

            ui.separator().classes('q-my-sm')

            # Pipeline config
            try:
                from frontend_nicegui.components.pipeline_config_ui import render_pipeline_config
                render_pipeline_config(state)
            except Exception:
                pass

            ui.separator().classes('q-my-sm')

            # Post-analysis config
            try:
                from frontend_nicegui.components.post_analysis_config import render_post_analysis_config
                render_post_analysis_config(state)
            except Exception:
                pass

            ui.separator().classes('q-my-sm')

            # Monotonicity config
            try:
                from frontend_nicegui.components.monotonicity_config import render_monotonicity_config
                render_monotonicity_config(state)
            except Exception:
                pass

        # ── DOE Tab (existing) ──
        with ui.tab_panel(doe_tab):
            doe_page.render()

    # ── Footer ──
    with ui.footer().classes('bg-gray-100'):
        with ui.row().classes('w-full justify-center py-2 gap-2'):
            ui.label('ChemAI MI Studio - Materials Informatics Platform').classes('text-xs text-gray-600')
            ui.separator().props('vertical')
            ui.button('❓ Help', on_click=lambda: ui.navigate.to('/help'),).props('flat dense size=xs')


# ── Help page (existing) ──
@ui.page('/help')
def help_page():
    ui.add_head_html('''
    <style>
    body { font-family: sans-serif; padding: 20px; max-width: 900px; margin: 0 auto; }
    h1 { color: #1976D2; }
    </style>
    ''')
    with ui.header().classes('items-center'):
        ui.link('← Back', '/').classes('text-white q-mr-md')
        ui.label('❓ Help - ChemAI MI Studio').classes('text-h6')

    with ui.column().classes('q-pa-lg q-gutter-md').style('max-width:900px;margin:0 auto;'):
        ui.label('ChemAI MI Studio').classes('text-h4')
        ui.markdown("""
## How to use

### LLM-assisted workflow (shortest path)
1. **Data & SMILES**: Upload CSV/Excel (or use sample benchmark)
2. **LLM Interview**: AI interviews you to understand analysis goals
3. **LLM Assistant**: AI helps with analysis planning and reporting
4. **AutoML & Results**: Automatic model training and evaluation
5. **Inverse**: Explore conditions for target properties

### SMILES Features (from chemai2_qwen)
- 14 descriptor engines: RDKit, Mordred, MolAI, XTB, etc.
- Automatic feature computation with progress tracking
- LLM-assisted feature selection

### EDA & Visualization
- Distribution plots, correlation heatmaps, scatter plots
- Outlier detection, clustering analysis
- PCA/t-SNE/UMAP dimensionality reduction

## UI Design Philosophy
- **LLM-assisted**: AI helps throughout the workflow
- **Progressive Disclosure**: Simple for beginners, detailed settings for experts
- **Full integration**: chemai2_cc (LLM) + chemai2_qwen (features/EDA/inverse)
""")


# ── Entry point ──
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='ChemAI MI Studio',
        host='0.0.0.0',
        port=8080,
        reload=False,
        storage_secret='chemai2_secret_key_2026'
    )
