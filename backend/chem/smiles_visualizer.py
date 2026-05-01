"""
SMILES structure visualization using RDKit.

Generates SVG images of molecules for display in NiceGUI.
"""

from __future__ import annotations

import io
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Draw


def smiles_to_svg(
    smiles: str,
    size: tuple[int, int] = (300, 300),
    kekulize: bool = True,
    wedge_bonds: bool = True,
) -> Optional[bytes]:
    """Convert a SMILES string to SVG image bytes.

    Parameters
    ----------
    smiles : str
        SMILES string to visualize.
    size : tuple[int, int], default=(300, 300)
        Image size (width, height) in pixels.
    kekulize : bool, default=True
        Whether to kekulize the molecule (show explicit double bonds).
    wedge_bonds : bool, default=True
        Whether to wedge stereobonds.

    Returns
    -------
    bytes or None
        SVG image bytes, or None if SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if kekulize:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        drawer = Draw.MolDraw2DSVG(size[0], size[1])
        if wedge_bonds:
            drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Remove XML declaration and DOCTYPE for embedding
        svg = svg.replace('<?xml version="1.0" encoding="UTF-8"?>\n', '')
        svg = svg.replace('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n', '')
        return svg.encode('utf-8')
    except Exception:
        return None


def smiles_to_base64_png(
    smiles: str,
    size: tuple[int, int] = (300, 300),
) -> Optional[str]:
    """Convert SMILES to base64-encoded PNG for web display.

    Parameters
    ----------
    smiles : str
        SMILES string to visualize.
    size : tuple[int, int], default=(300, 300)
        Image size (width, height) in pixels.

    Returns
    -------
    str or None
        Base64-encoded PNG string (data:image/png;base64,...), or None if invalid.
    """
    try:
        import base64
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()
        return 'data:image/png;base64,' + base64.b64encode(img_bytes).decode('utf-8')
    except Exception:
        return None


def validate_and_preview_smiles(
    smiles: str,
    size: tuple[int, int] = (300, 300),
) -> dict:
    """Validate SMILES and return preview info.

    Parameters
    ----------
    smiles : str
        SMILES string to check.
    size : tuple[int, int], default=(300, 300)
        Image size for preview.

    Returns
    -------
    dict
        {
            'valid': bool,
            'error': str or None,
            'svg': bytes or None,
            'formula': str or None,
            'mw': float or None,
        }
    """
    result = {
        'valid': False,
        'error': None,
        'svg': None,
        'formula': None,
        'mw': None,
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result['error'] = 'Invalid SMILES string'
            return result
        result['valid'] = True
        result['svg'] = smiles_to_svg(smiles, size=size)
        from rdkit.Chem import rdMolDescriptors
        result['formula'] = rdMolDescriptors.CalcMolFormula(mol)
        result['mw'] = rdMolDescriptors.CalcExactMolWt(mol)
        return result
    except Exception as e:
        result['error'] = str(e)
        return result
