"""
Beginner Mode Automation for ChemAI2

When enabled, LLM guides the user through the entire workflow:
data upload → preprocessing → EDA → ML modeling → results.
"""

from __future__ import annotations

from typing import Callable, Optional

from frontend_nicegui.utils import app_state


# Step callbacks (set by pages when they finish)
_step_callbacks: dict[str, Callable] = {}


def register_step_callback(step: str, callback: Callable) -> None:
    """Register a callback for when a step completes.

    Parameters
    ----------
    step : str
        Step name: 'data_loaded', 'preprocessing_done', 'eda_done',
        'ml_done', 'results_ready'.
    callback : callable
        Function to call when step completes.
    """
    _step_callbacks[step] = callback


def trigger_step(step: str, *args, **kwargs) -> None:
    """Trigger a step callback if registered.

    Parameters
    ----------
    step : str
        Step name that completed.
    """
    if step in _step_callbacks:
        _step_callbacks[step](*args, **kwargs)


def run_beginner_workflow() -> None:
    """Run the full beginner workflow if beginner_mode is enabled."""
    if not app_state.beginner_mode:
        return

    # Step 1: Data upload (user does this manually in UI)
    # When data is loaded, trigger next step
    if app_state.data_loaded:
        # Step 2: Auto-preprocess (use defaults)
        _auto_preprocess()
    else:
        app_state.navigate_to('data_upload')


def _auto_preprocess() -> None:
    """Auto-run preprocessing with default settings."""
    if not app_state.data_loaded:
        return
    # TODO: Call backend preprocessing with default settings
    # For now, just navigate to preprocessing page
    app_state.navigate_to('preprocessing')


def _auto_eda() -> None:
    """Auto-run EDA and generate LLM summary."""
    if not app_state.data_loaded:
        return
    # TODO: Call backend EDA, then LLM summary
    app_state.navigate_to('eda')


def _auto_ml() -> None:
    """Auto-select best model using LLM recommendation."""
    if not app_state.data_loaded or not app_state.target_column:
        return
    # TODO: Call backend ML with LLM-recommended settings
    app_state.navigate_to('ml_modeling')


def _show_results() -> None:
    """Show results with LLM interpretation."""
    app_state.navigate_to('results')


def beginner_mode_toggle_changed(enabled: bool) -> None:
    """Handle beginner mode toggle.

    Parameters
    ----------
    enabled : bool
        Whether beginner mode is enabled.
    """
    app_state.beginner_mode = enabled
    if enabled:
        run_beginner_workflow()
