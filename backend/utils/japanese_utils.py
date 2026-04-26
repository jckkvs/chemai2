# backend/utils/japanese_utils.py — guikit-learn 互換の日本語処理

"""
Japanese language support utilities for ChemAI ML Studio

Provides:
- Auto-detection of Japanese fonts (NotoSansJP, IPAex, Meiryo)
- Consistent font configuration for matplotlib/plotly/reportlab
- Japanese-safe text rendering for reports and visualizations

Compatible with guikit-learn's japanize_matplotlib integration.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# 日本語フォント候補（優先順位付き）
JAPANESE_FONT_CANDIDATES: List[str] = [
    'Noto Sans JP',          # Google Noto (Linux/Windows/macOS)
    'NotoSansJP-Regular',    # Alternative name
    'IPAexGothic',           # IPAフォント (Linux)
    'IPAexMincho',
    'Meiryo',                # Windows標準
    'Yu Gothic',             # Windows 10+
    'Hiragino Sans',         # macOS標準
    'Hiragino Kaku Gothic',
    'sans-serif',            # 最終フォールバック
]


def detect_japanese_font() -> str:
    """
    Detect best available Japanese font with fallback chain
    """
    # 1. matplotlib font_manager で利用可能フォントを検出
    try:
        from matplotlib import font_manager
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        for candidate in JAPANESE_FONT_CANDIDATES:
            if candidate in available_fonts: return candidate
    except ImportError: pass
    
    # 2. Linux: fc-list でシステムフォントを検出
    if sys.platform.startswith('linux'):
        try:
            result = subprocess.run(['fc-list', ':family'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                installed = result.stdout.lower()
                for candidate in JAPANESE_FONT_CANDIDATES:
                    if candidate.lower() in installed: return candidate
        except Exception: pass
    
    # 3. Windows: registry check
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts") as key:
                for i in range(winreg.QueryInfoKey(key)[1]):
                    font_name = winreg.EnumValue(key, i)[0]
                    for candidate in JAPANESE_FONT_CANDIDATES:
                        if candidate.lower() in font_name.lower(): return candidate
        except Exception: pass
    
    return 'sans-serif'


def configure_matplotlib_japanese(font_name: Optional[str] = None):
    """Configure matplotlib for Japanese text rendering"""
    if font_name is None: font_name = detect_japanese_font()
    matplotlib.rcParams['font.family'] = font_name
    matplotlib.rcParams['axes.unicode_minus'] = False
    if not hasattr(plt, '_chemai_japanese_configured'):
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        plt._chemai_japanese_configured = True
        logger.info(f"Configured matplotlib for Japanese: {font_name}")


def configure_plotly_japanese(font_name: Optional[str] = None) -> Dict[str, Any]:
    """Get Plotly layout configuration for Japanese text"""
    if font_name is None: font_name = detect_japanese_font()
    return {
        'font': dict(family=font_name, size=12),
        'title': dict(font=dict(family=font_name, size=14)),
        'xaxis': dict(titlefont=dict(family=font_name)),
        'yaxis': dict(titlefont=dict(family=font_name)),
        'legend': dict(font=dict(family=font_name)),
    }


def configure_reportlab_japanese(pdf_path: Path, font_name: Optional[str] = None) -> str:
    """Register Japanese font for ReportLab PDF generation"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if font_name is None: font_name = detect_japanese_font()
    
    font_paths: Dict[str, List[str]] = {
        'Noto Sans JP': ['/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf', '/usr/share/fonts/noto-cjk/NotoSansJP-Regular.otf'],
        'IPAexGothic': ['/usr/share/fonts/opentype/ipaexfont/IPAexGothic.ttf'],
        'Meiryo': ['C:/Windows/Fonts/meiryo.ttc', 'C:/Windows/Fonts/meiryo.ttf'],
    }
    
    for font_path in font_paths.get(font_name, []):
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name
            except: pass
    return 'Helvetica'


def japanize_plotly_figure(fig, font_name: Optional[str] = None):
    """Apply Japanese font settings to an existing Plotly figure"""
    if font_name is None: font_name = detect_japanese_font()
    fig.update_layout(configure_plotly_japanese(font_name))
    return fig


def get_japanese_font_info() -> Dict[str, Any]:
    """Get detailed information about detected Japanese font"""
    font_name = detect_japanese_font()
    return {'font_name': font_name, 'available': font_name != 'sans-serif', 'platform': sys.platform}
