# tests/test_japanese_utils.py — 日本語フォントユーティリティテスト

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import matplotlib
import matplotlib.pyplot as plt

from backend.utils.japanese_utils import (
    detect_japanese_font,
    configure_matplotlib_japanese,
    configure_plotly_japanese,
    configure_reportlab_japanese,
    japanize_plotly_figure,
    get_japanese_font_info,
    JAPANESE_FONT_CANDIDATES,
)

class TestFontDetection:
    def test_detect_japanese_font_returns_string(self):
        result = detect_japanese_font()
        assert isinstance(result, str)
    
    def test_detect_japanese_font_in_candidates_or_fallback(self):
        result = detect_japanese_font()
        assert result in JAPANESE_FONT_CANDIDATES + ['sans-serif']

class TestMatplotlibConfiguration:
    def test_configure_matplotlib_sets_rcparams(self):
        configure_matplotlib_japanese('TestFont')
        assert matplotlib.rcParams['font.family'] == 'TestFont'
        assert matplotlib.rcParams['axes.unicode_minus'] is False

class TestPlotlyConfiguration:
    def test_configure_plotly_returns_layout_dict(self):
        result = configure_plotly_japanese('TestFont')
        assert result['font']['family'] == 'TestFont'
    
    def test_japanize_plotly_figure_modifies_in_place(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        japanize_plotly_figure(fig, 'TestFont')
        assert fig.layout.font.family == 'TestFont'

class TestReportLabConfiguration:
    def test_configure_reportlab_fallback_when_file_not_found(self, tmp_path):
        result = configure_reportlab_japanese(tmp_path / 'output.pdf', 'NonExistentFont')
        assert result == 'Helvetica'

class TestFontInfo:
    def test_get_japanese_font_info_returns_dict(self):
        result = get_japanese_font_info()
        assert 'font_name' in result
        assert 'available' in result
