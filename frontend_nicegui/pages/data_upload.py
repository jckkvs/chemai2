from nicegui import ui
from frontend_nicegui.utils import app_state
from frontend_nicegui.components.domain_knowledge_input import domain_knowledge_button
import pandas as pd
from pathlib import Path
from io import BytesIO


def is_mixture_data(df: pd.DataFrame, file_name: str = "") -> tuple:
    """Detect if data is mixture type (multiple SMILES + WT% columns).
    Returns (is_mixture, description).
    """
    smiles_cols = [c for c in df.columns if 'smiles' in c.lower()]
    wt_cols = [c for c in df.columns if any(x in c.lower() for x in ['wt%', 'weight', 'ratio'])]

    # Check filename
    if 'mixture' in file_name.lower():
        return True, f"混合系データです（ファイル名: {file_name}）"

    # Check for multiple SMILES columns (mixture indicator)
    if len(smiles_cols) >= 2:
        desc = f"混合系データの可能性があります（SMILES列: {len(smiles_cols)}件"
        if wt_cols:
            desc += f"、WT%列: {len(wt_cols)}件"
        desc += "）。大丈夫ですか？"
        return True, desc

    return False, ""


def show_mixture_confirmation(df: pd.DataFrame, file_name: str = ""):
    """Show confirmation dialog for mixture data."""
    is_mix, message = is_mixture_data(df, file_name)

    if not is_mix:
        return False

    ui.notify(message, type='warning', timeout=5)

    with ui.dialog() as dialog:
        with ui.card().classes('w-full max-w-2xl p-6').style('background-color: #1f2937; color: #f9fafb; border: 1px solid #f59e0b;'):
            ui.label('⚠️ 混合系データの確認').classes('text-xl font-bold text-amber-400 mb-4')

            ui.label(message).classes('text-sm text-gray-300 mb-4')

            ui.label('混合系データでは複数のSMILES列とWT%列があります。').classes('text-xs text-gray-400 mb-2')
            ui.label('処理時は各成分の組成比を考慮した特徴量が生成されます。').classes('text-xs text-gray-400 mb-4')

            with ui.row().classes('w-full gap-2'):
                ui.button('OK、大丈夫です', icon='check', on_click=dialog.close).props('color=primary')

                def on_cancel():
                    ui.notify('データをクリアしました', type='info')
                    app_state.data_df = None
                    app_state.data_loaded = False
                    dialog.close()
                    app_state.navigate_to('data_upload')

                ui.button('キャンセル', icon='close', on_click=on_cancel).props('flat color=gray')

    dialog.open()
    return True


# Sample data path
SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'samples'

def handle_file_upload(event) -> None:
    """Handle file upload event (click or drag & drop)"""
    try:
        file_name = event.name
        file_content = event.content  # bytes

        if file_name.endswith('.csv'):
            df = pd.read_csv(BytesIO(file_content))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(BytesIO(file_content))
        else:
            ui.notify('CSVまたはExcelファイルを選んでください', type='warning')
            return

        app_state.data_df = df
        app_state.data_loaded = True
        app_state.data_path = file_name

        ui.notify(f'{len(df)}行のデータを読み込みました', type='positive')

        # Check for mixture data
        if show_mixture_confirmation(df, file_name):
            # Don't navigate yet - let user confirm first
            return

        # Beginner mode: auto-navigate to EDA
        if app_state.beginner_mode:
            ui.notify('初心者モード：まずはデータを眺めてみましょう', type='info')
            app_state.navigate_to('eda')
        else:
            app_state.navigate_to('data_upload')

    except Exception as e:
        ui.notify(f'読み込めませんでした: {str(e)}', type='negative')

def load_sample_data(sample_name: str) -> None:
    """Load sample data"""
    try:
        sample_path = SAMPLE_DATA_DIR / f'{sample_name}.csv'
        if not sample_path.exists():
            sample_path = Path(__file__).parent.parent.parent / 'data' / f'{sample_name}.csv'

        if not sample_path.exists():
            ui.notify(f'サンプルが見つかりません: {sample_name}', type='warning')
            return

        df = pd.read_csv(sample_path)
        app_state.data_df = df
        app_state.data_loaded = True
        app_state.data_path = str(sample_path)

        ui.notify(f'サンプルデータを読み込みました', type='positive')

        # Check for mixture data
        if show_mixture_confirmation(df, f'{sample_name}.csv'):
            # Don't navigate yet - let user confirm first
            return

        # Beginner mode: auto-navigate to EDA
        if app_state.beginner_mode:
            ui.notify('初心者モード：まずはデータを眺めてみましょう', type='info')
            app_state.navigate_to('eda')
        else:
            app_state.navigate_to('data_upload')

    except Exception as e:
        ui.notify(f'エラー: {str(e)}', type='negative')

def page_data_upload() -> None:
    # Beginner mode check
    if not hasattr(app_state, 'beginner_mode'):
        app_state.beginner_mode = False
    """Data upload page - minimal, just what user needs"""
    with ui.card().classes('w-full max-w-3xl mx-auto p-8 relative').style('background-color: #111827; color: #F9FAFB; border: none;'):
        # Domain knowledge button
        domain_knowledge_button('data_upload')

        # Data not loaded - show upload
        if app_state.data_df is None:
            ui.label('データを入れてください').classes('text-2xl font-bold text-white mb-6')

            # Upload section
            ui.upload(
                label='ファイルを選択またはドラッグアンドドロップ（CSV/Excel）',
                on_upload=handle_file_upload,
                auto_upload=True
            ).classes('w-full').props('accept=.csv,.xls,.xlsx drag')

            ui.separator().classes('bg-gray-700 my-6')

            # Sample data - simple
            ui.label('サンプルを試す').classes('text-lg text-gray-300 mb-4')

            for sample_id, label in [
                ('tabular_50_safe', 'テーブルデータ'),
                ('test_smiles', 'SMILESデータ'),
            ]:
                ui.button(
                    label,
                    icon='science',
                    on_click=lambda _, sid=sample_id: load_sample_data(sid)
                ).classes('w-full mb-2').props('flat color=gray-300')

        # Data loaded - show what to do next
        else:
            df = app_state.data_df
            ui.label('データを読み込みました').classes('text-2xl font-bold text-white mb-2')
            ui.label(f'{len(df)}行 × {len(df.columns)}列').classes('text-gray-300 mb-4')

            # Show beginner mode status
            if app_state.beginner_mode:
                ui.label('💡 初心者モード有効：まずはデータを眺めましょう').classes('text-sm text-blue-200 mb-2')

            # What to do next - select target
            if app_state.target_column is None:
                ui.label('次は：予測したい物性を選んでください').classes('text-lg text-blue-300 mb-4')

                ui.select(
                    options=list(df.columns),
                    label='予測したい物性（列名）',
                    value=None,
                    on_change=lambda e: setattr(app_state, 'target_column', e.value)
                ).classes('w-full')

                # After selecting target, show next button
                if app_state.target_column:
                    ui.button(
                        '次へ：AIに相談する →',
                        icon='arrow_forward',
                        on_click=lambda: app_state.navigate_to('llm_interview')
                    ).props('color=primary size=lg').classes('w-full mt-4')

            else:
                ui.label(f'予測対象: {app_state.target_column}').classes('text-green-300 mb-4')
                ui.button(
                    '次へ：AIに相談する →',
                    icon='arrow_forward',
                    on_click=lambda: app_state.navigate_to('llm_interview')
                ).props('color=primary size=lg').classes('w-full')
