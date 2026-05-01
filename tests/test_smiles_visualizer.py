"""
Tests for backend.chem.smiles_visualizer
"""

from __future__ import annotations

import pytest

from backend.chem.smiles_visualizer import (
    smiles_to_svg,
    smiles_to_base64_png,
    validate_and_preview_smiles,
)


class TestSmilesToSvg:
    """Tests for smiles_to_svg"""

    def test_valid_smiles(self):
        svg = smiles_to_svg('CCO')
        assert svg is not None
        assert b'<svg' in svg
        assert b'</svg>' in svg

    def test_invalid_smiles(self):
        svg = smiles_to_svg('invalid_smiles')
        assert svg is None

    def test_kekulize(self):
        svg = smiles_to_svg('c1ccccc1')  # benzene (aromatic)
        assert svg is not None
        # Keuklized should show double bonds
        assert b'class="bond-2"' in svg or b'double' in svg.lower()

    def test_size(self):
        svg = smiles_to_svg('CCO', size=(200, 200))
        assert svg is not None
        assert b'width="200"' in svg or b'height="200"' in svg


class TestSmilesToBase64Png:
    """Tests for smiles_to_base64_png"""

    def test_valid_smiles(self):
        png = smiles_to_base64_png('CCO')
        assert png is not None
        assert png.startswith('data:image/png;base64,')

    def test_invalid_smiles(self):
        png = smiles_to_base64_png('invalid')
        assert png is None


class TestValidateAndPreviewSmiles:
    """Tests for validate_and_preview_smiles"""

    def test_valid_smiles(self):
        result = validate_and_preview_smiles('CCO')
        assert result['valid'] is True
        assert result['error'] is None
        assert result['svg'] is not None
        assert result['formula'] == 'C2H6O'
        assert result['mw'] is not None
        assert 46.0 < result['mw'] < 47.0

    def test_invalid_smiles(self):
        result = validate_and_preview_smiles('invalid')
        assert result['valid'] is False
        assert result['error'] is not None
        assert result['svg'] is None

    def test_empty_smiles(self):
        result = validate_and_preview_smiles('')
        assert result['valid'] is False
