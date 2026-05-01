from nicegui import ui
from frontend_nicegui.utils import app_state
from frontend_nicegui.components.domain_knowledge_input import domain_knowledge_button
from frontend_nicegui.components.variable_helper import show_variable_dialog
from frontend_nicegui.components.smiles_thumbnails import show_smiles_preview
import pandas as pd
import numpy as np
from typing import Dict, Any


def page_eda() -> None:
    """EDA page - just enough to understand data"""

    if not app_state.data_loaded or app_state.data_df is None:
        with ui.card().classes('w-full max-w-3xl mx-auto p-8 relative').style('background-color: #111827; color: #F9FAFB; border: none;'):
            domain_knowledge_button('eda')
            ui.label('データがありません').classes('text-2xl font-bold text-white mb-4')
            ui.button(
                'データを入れる →',
                icon='upload',
                on_click=lambda: ui.navigate.to('/#data_upload')
            ).props('color=primary size=lg')
        return

    df = app_state.data_df

    with ui.card().classes('w-full max-w-3xl mx-auto p-8').style('background-color: #111827; color: #F9FAFB; border: none;'):

        # Beginner mode guidance
        if app_state.beginner_mode:
            with ui.card().classes('w-full bg-blue-900 p-4 mb-6').style('border: 1px solid #3b82f6;'):
                ui.label('💡 初心者モード：データを眺める').classes('text-lg font-bold text-blue-200 mb-2')
                ui.label('このステップでは、データの基本統計・可視化を行い、パターンを把握します。').classes('text-sm text-blue-100')
                ui.label('以下の項目を順に確認してください：').classes('text-sm text-blue-100 mt-2')
                ui.label('  1. インタラクティブ・フィルタリングでデータを絞り込む').classes('text-xs text-blue-200')
                ui.label('  2. 相関の高い変数を確認する').classes('text-xs text-blue-200')
                ui.label('  3. 分布を確認して外れ値がないかチェックする').classes('text-xs text-blue-200')
                ui.button(
                    '次へ：AIに相談する →',
                    icon='arrow_forward',
                    on_click=lambda: app_state.navigate_to('llm_interview')
                ).props('color=blue size=sm').classes('mt-2')

        ui.label('データを眺める').classes('text-2xl font-bold text-white mb-6').tooltip('EDA: データの基本統計・可視化を行い、パターンを把握するステップです（機械学習の専門用語では「探索的データ分析」と呼ばれます）')

        # SMILES structure thumbnails
        if app_state.data_df is not None:
            show_smiles_preview(app_state.data_df, max_per_col=10)

        # Interactive filters section
        ui.separator().classes('bg-gray-700 my-4')
        ui.label('🔍 インタラクティブ・フィルタリング').classes('text-lg text-blue-300 mb-4')

        # Initialize filtered_df in app_state
        if not hasattr(app_state, 'filtered_df') or app_state.filtered_df is None:
            app_state.filtered_df = df.copy()
        if not hasattr(app_state, 'filter_settings') or app_state.filter_settings is None:
            app_state.filter_settings = {'col': None, 'min': None, 'max': None}

        # Filter UI
        with ui.expansion('フィルター設定', icon='filter').classes('w-full bg-gray-800').style('border: 1px solid #374151;'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            if numeric_cols:
                ui.label('数値変数のフィルター').classes('text-sm text-gray-300 mb-2')

                filter_col = ui.select(
                    options=numeric_cols,
                    label='フィルターする列',
                    value=app_state.filter_settings.get('col') or (numeric_cols[0] if numeric_cols else None)
                ).classes('w-full mb-2')

                if filter_col.value and filter_col.value in df.columns:
                    col_min = float(df[filter_col.value].min())
                    col_max = float(df[filter_col.value].max())
                    current_min = app_state.filter_settings.get('min', col_min)
                    current_max = app_state.filter_settings.get('max', col_max)

                    range_filter = ui.slider(
                        min=col_min,
                        max=col_max,
                        value={'min': current_min, 'max': current_max},
                        label=f'{filter_col.value} の範囲'
                    ).classes('w-full')

            if categorical_cols:
                ui.separator().classes('bg-gray-700 my-2')
                ui.label('カテゴリ変数のフィルター').classes('text-sm text-gray-300 mb-2')

                for cat_col in categorical_cols[:3]:
                    unique_vals = df[cat_col].unique().tolist()
                    ui.select(
                        options=['全て'] + unique_vals,
                        label=cat_col,
                        value='全て'
                    ).classes('w-full mb-1')

            def apply_filters():
                filtered = df.copy()
                # Apply numeric filter
                if filter_col.value and filter_col.value in df.columns:
                    min_val = range_filter.value['min']
                    max_val = range_filter.value['max']
                    filtered = filtered[
                        (filtered[filter_col.value] >= min_val) &
                        (filtered[filter_col.value] <= max_val)
                    ]
                    app_state.filter_settings = {
                        'col': filter_col.value,
                        'min': min_val,
                        'max': max_val
                    }

                app_state.filtered_df = filtered
                ui.notify(f'{len(filtered)}件にフィルターされました', type='positive')
                ui.navigate.to('/#eda', reload=True)

            ui.button('フィルター適用', icon='check', on_click=apply_filters).props('color=primary').classes('mt-2')

        # Show filtered data stats
        display_df = app_state.filtered_df if app_state.filtered_df is not None else df
        ui.label(f'表示中: {len(display_df)}件 / 全{len(df)}件').classes('text-sm text-gray-400 mb-4')

        # Helper to create variable label with "?" button
        def var_label_with_help(col_name: str, label_text: str = None):
            """Show variable name with '?' button for explanation."""
            display_name = label_text or col_name
            with ui.row().classes('items-center gap-1'):
                ui.label(display_name).classes('text-sm text-gray-300')
                ui.button(icon='help', on_click=lambda _, c=col_name: show_variable_dialog(c)
                ).props('flat dense size=xs color=gray').tooltip('この変数について説明する')

        # Simple stats
        if app_state.target_column and app_state.target_column in df.columns:
            target = df[app_state.target_column]
            ui.label(f'予測対象「{app_state.target_column}」の傾向').classes('text-lg text-blue-300 mb-4')

            with ui.row().classes('w-full gap-4 mb-6'):
                for label, value, col in [
                    ('平均', f"{target.mean():.4f}", app_state.target_column),
                    ('最小値', f"{target.min():.4f}", app_state.target_column),
                    ('最大値', f"{target.max():.4f}", app_state.target_column),
                ]:
                    with ui.card().classes('flex-1 bg-gray-800 p-3').style('border: 1px solid #374151;'):
                        var_label_with_help(col, label)
                        ui.label(value).classes('text-xl font-bold text-white')

        # Data drill-down: show sample details
        ui.separator().classes('bg-gray-700 my-4')
        ui.label('📊 データドリルダウン（サンプル詳細）').classes('text-lg text-blue-300 mb-4')

        # Show first 10 rows with clickable samples
        preview_df = display_df.head(10)
        with ui.scroll_area().classes('w-full h-64'):
            ui.table(
                columns=list(preview_df.columns),
                rows=[list(row) for _, row in preview_df.iterrows()],
                title=f'表示サンプル（最初の{len(preview_df)}件）'
            ).classes('w-full')

        # Scatter plot if possible
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2 and app_state.target_column in numeric_cols:
            ui.separator().classes('bg-gray-700 my-4')
            ui.label('分布を確認').classes('text-lg text-white mb-4')

            with ui.row().classes('w-full gap-4'):
                with ui.column().classes('flex-1'):
                    var_label_with_help(numeric_cols[0] if len(numeric_cols) > 0 else '', '横軸')
                    x_col = ui.select(
                        options=numeric_cols,
                        label='横軸',
                        value=numeric_cols[0] if len(numeric_cols) > 0 else None
                    ).classes('w-full')

                with ui.column().classes('flex-1'):
                    var_label_with_help(app_state.target_column, '縦軸（予測対象）')
                    y_col = ui.select(
                        options=numeric_cols,
                        label='縦軸（予測対象）',
                        value=app_state.target_column
                    ).classes('w-full')

            def show_plot():
                try:
                    import plotly.express as px
                    fig = px.scatter(df, x=x_col.value, y=y_col.value, title='分布')
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e0f0', size=11)
                    )
                    ui.plotly(fig).classes('w-full')
                except Exception as e:
                    ui.notify(f'プロットできません: {str(e)}', type='warning')

            ui.button('表示', icon='bar_chart', on_click=lambda: show_plot()).props('color=primary').classes('mt-2')

        # Next step
        ui.separator().classes('bg-gray-700 my-6')
        ui.label('確認が終わったら').classes('text-lg text-gray-300 mb-4')
        ui.button(
            'AIに相談する →',
            icon='arrow_forward',
            on_click=lambda: ui.navigate.to('/#llm_interview')
        ).props('color=primary size=lg').classes('w-full')
