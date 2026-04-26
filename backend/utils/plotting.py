# backend/utils/plotting.py — 精緻化版 (Plotly可視化ユーティリティ)

from typing import Optional, Union, List, Dict, Any, Literal
from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# 【修正点1】グローバル設定の初期化フラグ
_PLOTTING_INITIALIZED = False


def initialize_plotting(
    template: str = 'plotly_white',
    font_family: Optional[str] = None,
    font_size: int = 12,
    color_sequence: Optional[List[str]] = None,
    width: int = 1000,
    height: int = 600,
    scale: float = 1.0
):
    """
    Initialize Plotly with consistent theme and font settings (idempotent)
    """
    global _PLOTTING_INITIALIZED
    
    if _PLOTTING_INITIALIZED:
        logger.debug("Plotting already initialized. Updating settings only.")
    else:
        _PLOTTING_INITIALIZED = True
        logger.info("Initializing Plotly with consistent settings")
    
    if font_family is None:
        font_family = _detect_best_japanese_font()
    
    px.defaults.template = template
    go.layout.Template.layout.font.family = font_family
    go.layout.Template.layout.font.size = font_size
    
    if color_sequence:
        px.defaults.color_discrete_sequence = color_sequence
    
    px.defaults.width = width
    px.defaults.height = height
    px.defaults.renderers.default = 'browser'


def _detect_best_japanese_font() -> str:
    """
    Detect best available Japanese font with fallback chain
    """
    import subprocess
    import sys
    
    candidate_fonts = [
        'Noto Sans JP', 'NotoSansJP', 'IPAexGothic', 'IPAexMincho',
        'Meiryo', 'Yu Gothic', 'Hiragino Sans', 'sans-serif'
    ]
    
    if sys.platform.startswith('linux'):
        try:
            result = subprocess.run(['fc-list', ':family'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                installed = result.stdout.lower()
                for font in candidate_fonts:
                    if font.lower() in installed: return font
        except Exception: pass
    
    try:
        from matplotlib import font_manager
        available = [f.name for f in font_manager.fontManager.ttflist]
        for font in candidate_fonts:
            if font in available: return font
    except ImportError: pass
    
    return 'sans-serif'


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    title: Optional[str] = None,
    log_x: bool = False,
    log_y: bool = False,
    trendline: Optional[Literal['ols', 'lowess']] = None,
    max_points: int = 10000,
    sampling_method: Literal['random', 'stratified', 'first'] = 'random',
    **kwargs
) -> go.Figure:
    """
    Create scatter plot with automatic downsampling for large datasets
    """
    if len(df) > max_points:
        df_plot = _downsample_dataframe(df, max_points, sampling_method, color)
    else:
        df_plot = df.copy()
    
    if log_x and (df_plot[x] <= 0).any(): log_x = False
    if log_y and (df_plot[y] <= 0).any(): log_y = False
    
    fig = px.scatter(
        df_plot, x=x, y=y, color=color, size=size,
        hover_data=hover_data, title=title,
        log_x=log_x, log_y=log_y, trendline=trendline, **kwargs
    )
    
    fig.update_layout(
        template=px.defaults.template,
        font=dict(family=go.layout.Template.layout.font.family, size=go.layout.Template.layout.font.size)
    )
    return fig


def _downsample_dataframe(
    df: pd.DataFrame,
    target_size: int,
    method: Literal['random', 'stratified', 'first'],
    stratify_column: Optional[str] = None
) -> pd.DataFrame:
    """Downsample DataFrame while preserving distribution characteristics"""
    if len(df) <= target_size: return df.copy()
    rng = np.random.default_rng(42)
    
    if method == 'first': return df.head(target_size).copy()
    elif method == 'random':
        indices = rng.choice(len(df), size=target_size, replace=False)
        return df.iloc[indices].copy()
    elif method == 'stratified' and stratify_column and stratify_column in df.columns:
        from sklearn.model_selection import train_test_split
        _, df_sample = train_test_split(df, train_size=target_size, stratify=df[stratify_column], random_state=42, shuffle=True)
        return df_sample.copy()
    else:
        indices = rng.choice(len(df), size=target_size, replace=False)
        return df.iloc[indices].copy()


def export_figure(
    fig: go.Figure,
    output_path: Union[str, Path],
    format: Literal['png', 'jpg', 'svg', 'pdf', 'html'] = 'png',
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 2.0,
    max_file_size_mb: float = 10.0
) -> Path:
    """Export Plotly figure with size control and format validation"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format = format.lower().lstrip('.')
    if format == 'jpeg': format = 'jpg'
    
    try:
        write_kwargs = {'format': format}
        if width: write_kwargs['width'] = width
        if height: write_kwargs['height'] = height
        if format in ('png', 'jpg') and scale: write_kwargs['scale'] = scale
        
        if format == 'html':
            fig.write_html(str(output_path), include_plotlyjs='cdn', auto_open=False)
        else:
            fig.write_image(str(output_path), **write_kwargs)
        
        if output_path.stat().st_size / (1024 * 1024) > max_file_size_mb and format in ('png', 'jpg'):
            write_kwargs['scale'] = max(1.0, scale * 0.5)
            fig.write_image(str(output_path), **write_kwargs)
        return output_path
    except Exception as e:
        logger.error(f"Export failed: {e}")
        if format != 'html':
            html_path = output_path.with_suffix('.html')
            fig.write_html(str(html_path), include_plotlyjs='cdn', auto_open=False)
            return html_path
        raise


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
    annot: bool = True,
    colorscale: str = 'RdBu_r',
    max_columns: int = 30,
    **kwargs
) -> go.Figure:
    """Create correlation heatmap with automatic column selection"""
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    if len(numeric_cols) > max_columns:
        variances = df[numeric_cols].var()
        numeric_cols = variances.nlargest(max_columns).index.tolist()
    
    if len(numeric_cols) < 2: return go.Figure().add_annotation(text="Insufficient data")
    
    corr = df[numeric_cols].corr(method=method)
    fig = px.imshow(corr, text_auto='.2f' if annot else False, aspect='auto', color_continuous_scale=colorscale, **kwargs)
    fig.update_layout(template=px.defaults.template, font=dict(family=go.layout.Template.layout.font.family, size=go.layout.Template.layout.font.size), title='Correlation Matrix')
    return fig
