# prototyping/mi_autonomous_studio_v2.py
import os
import sys
import json
from pathlib import Path
from nicegui import ui, app

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from backend.llm.hardware_detector import detect_hardware
from backend.llm.model_selector import select_optimal_model
from backend.llm.snippets import SnippetLibrary, SnippetAssembler

# Initialize components
library = SnippetLibrary()
assembler = SnippetAssembler(library)
hardware = detect_hardware()
model_config = select_optimal_model(profile=hardware)

# UI State
state = {
    "hardware": hardware.to_dict(),
    "model": model_config,
    "snippets": library.get_all_summaries(),
    "selected_snippets": [],
    "execution_plan": ""
}

def refresh_plan():
    state["execution_plan"] = assembler.assemble(state["selected_snippets"])
    plan_display.set_content(f"```python\n{state['execution_plan']}\n```")

@ui.page('/')
def main_page():
    ui.dark_mode(True)
    ui.colors(primary='#00d4ff', secondary='#7b2ff7', accent='#4ade80')

    with ui.header().classes('items-center justify-between bg-slate-900 border-b border-white/10 p-4'):
        with ui.row().classes('items-center gap-4'):
            ui.icon('auto_awesome', size='2rem').classes('text-primary')
            ui.label('MI Autonomous Studio v2').classes('text-2xl font-bold hero-gradient')
        
        with ui.row().classes('items-center gap-6'):
            ui.label(f"Environment: {state['hardware']['env_id']}").classes('text-sm font-mono bg-white/5 px-3 py-1 rounded')
            ui.label(f"Model: {state['model'].repo_id}").classes('text-sm text-secondary')

    with ui.row().classes('w-full h-[calc(100vh-80px)] no-wrap gap-0'):
        # --- Left Panel: Concierge Chat & Snippets ---
        with ui.column().classes('w-1/3 p-6 border-r border-white/10 gap-6 overflow-y-auto'):
            ui.label('🤖 AI Concierge').classes('text-xl font-bold')
            
            with ui.card().classes('w-full p-4 bg-white/5 border border-white/10'):
                ui.markdown('**Research Intent**')
                query_input = ui.textarea(
                    placeholder='Describe your research goal (e.g., "Analyze polymer solubility using RDKit and Linear Tree")'
                ).classes('w-full')
                ui.button('Generate Plan', icon='bolt', on_click=lambda: ui.notify("Analyzing intent...")).classes('w-full btn-primary')

            ui.separator().classes('my-4')
            
            ui.label('📦 Golden Snippets').classes('text-lg font-bold')
            for snippet in state["snippets"]:
                with ui.card().classes('w-full p-3 mb-2 hover-bounce cursor-pointer bg-white/5 border border-white/5'):
                    with ui.row().classes('items-center justify-between w-full'):
                        ui.label(snippet['name']).classes('font-bold')
                        ui.checkbox(on_change=lambda e, sid=snippet['id']: 
                            (state["selected_snippets"].append(sid) if e.value else state["selected_snippets"].remove(sid), refresh_plan())
                        )
                    ui.label(snippet['description']).classes('text-xs text-grey-400')

        # --- Right Panel: Plan & Execution ---
        with ui.column().classes('w-2/3 p-6 bg-slate-950/50 gap-6 overflow-y-auto'):
            ui.label('📝 Execution Plan (Validated MI Code)').classes('text-xl font-bold')
            
            global plan_display
            plan_display = ui.markdown('```python\n# Select snippets to build your plan\n```').classes('w-full p-4 bg-black/40 rounded border border-white/5 font-mono text-sm')
            
            with ui.row().classes('w-full gap-4'):
                ui.button('Validate Plan', icon='fact_check').props('outline').classes('flex-grow')
                ui.button('Execute in Sandbox', icon='play_circle').classes('flex-grow btn-primary')
            
            with ui.expansion('📊 Decision Support Visualization', icon='insights', value=True).classes('w-full border border-white/10 rounded'):
                ui.label('Results will appear here after execution...').classes('text-grey-500 p-4 italic')

    # CSS for premium look
    ui.add_head_html('''
    <style>
    .hero-gradient {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .btn-primary {
        background: linear-gradient(135deg, #00d4ff, #7b2ff7) !important;
        color: white !important;
    }
    .hover-bounce:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.08) !important;
    }
    </style>
    ''')

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title='MI Autonomous Studio v2', port=8086, dark=True)
