"""
frontend_nicegui/components/eda_tab.py

EDA統合タブ — Core EDA, Data Sandbox, Advanced EDAを統合
"""
from nicegui import ui
from frontend_nicegui.components.eda_panel import render_eda_panel
from frontend_nicegui.components.data_sandbox import render_data_sandbox
from frontend_nicegui.components.eda_advanced_panel import render_eda_advanced_panel
from frontend_nicegui.components.data_dialogue_panel import render_data_dialogue

def render_eda_tab(state: dict):
    """
    EDAワークスペースのメインエントリーポイント
    """
    df = state.get("df")
    if df is None:
        with ui.card().classes("full-width q-pa-xl items-center justify-center text-center glass-card"):
            ui.icon("analytics", size="xl", color="grey-7")
            ui.label("データが読み込まれていません").classes("text-h6 text-grey-5")
            ui.label("「📁 データ管理」でCSVファイルをアップロードしてください。").classes("text-grey-6")
        return

    # 頂上にダイアログ（サジェスト）を表示
    render_data_dialogue(state)

    with ui.tabs().classes("full-width q-mt-md").props("dense no-caps active-color=cyan indicator-color=cyan") as eda_subtabs:
        tab_core = ui.tab("core_eda", label="📊 基本解析 (Core)", icon="insights")
        tab_sandbox = ui.tab("sandbox", label="🧪 Data Sandbox", icon="science")
        tab_adv = ui.tab("advanced", label="🔬 物理・矛盾解析", icon="biotech")
        
    with ui.tab_panels(eda_subtabs, value="core_eda").classes("full-width bg-transparent"):
        with ui.tab_panel("core_eda"):
            render_eda_panel(state)
            
        with ui.tab_panel("sandbox"):
            render_data_sandbox(state)
            
        with ui.tab_panel("advanced"):
            render_eda_advanced_panel(state)
