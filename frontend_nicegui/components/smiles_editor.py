"""
SMILES Editor Component for ChemAI2
Provides SMILES input with real-time structure preview and validation.
"""

from __future__ import annotations

from typing import Optional

from nicegui import ui

from backend.chem.smiles_visualizer import validate_and_preview_smiles


def smiles_editor(
    label: str = 'SMILES',
    placeholder: str = '例: CCO (エタノール)',
    value: str = '',
    on_change: Optional[callable] = None,
    size: tuple[int, int] = (300, 300),
) -> dict:
    """Create a SMILES editor with structure preview.

    Parameters
    ----------
    label : str, default='SMILES'
        Label for the input field.
    placeholder : str, default='例: CCO (エタノール)'
        Placeholder text for the input.
    value : str, default=''
        Initial SMILES value.
    on_change : callable, optional
        Callback when SMILES changes. Receives dict with 'valid', 'smiles', 'svg', etc.
    size : tuple[int, int], default=(300, 300)
        Preview image size (width, height).

    Returns
    -------
    dict
        Contains 'input' (ui.input), 'preview' (ui.html), 'props' (ui.label for properties).
    """

    # State
    state = {
        'valid': False,
        'smiles': value,
        'svg': None,
        'formula': None,
        'mw': None,
        'error': None,
    }

    # Build UI
    with ui.card().classes('w-full p-4 gap-4'):
        # Input row
        smi_input = ui.input(
            label=label,
            placeholder=placeholder,
            value=value,
        ).classes('w-full')

        # Preview area
        with ui.row().classes('w-full items-start gap-4'):
            preview = ui.html('').classes('border rounded p-2 bg-white')
            with ui.column().classes('gap-2'):
                formula_label = ui.label('').classes('text-sm text-gray-600')
                mw_label = ui.label('').classes('text-sm text-gray-600')
                error_label = ui.label('').classes('text-sm text-red-500')

    # Update preview
    def update_preview():
        smi = smi_input.value.strip()
        state['smiles'] = smi
        if not smi:
            preview.content = ''
            formula_label.text = ''
            mw_label.text = ''
            error_label.text = ''
            state.update({'valid': False, 'error': None, 'svg': None})
            return
        result = validate_and_preview_smiles(smi, size=size)
        state.update(result)
        if result['valid']:
            if result['svg']:
                preview.content = result['svg'].decode('utf-8')
            formula_label.text = f"分子式: {result['formula']}" if result['formula'] else ''
            mw_label.text = f"分子量: {result['mw']:.2f}" if result['mw'] else ''
            error_label.text = ''
        else:
            preview.content = ''
            formula_label.text = ''
            mw_label.text = ''
            error_label.text = result['error'] or 'Invalid SMILES'

        if on_change:
            on_change(state)

    smi_input.on_value_change(update_preview)

    # Initial preview
    if value:
        update_preview()

    return {
        'input': smi_input,
        'preview': preview,
        'formula': formula_label,
        'mw': mw_label,
        'error': error_label,
        'state': state,
    }
