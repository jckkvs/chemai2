"""
backend/chem/utils.py

Utility functions for chemistry-related operations.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def smiles_to_3d_mol(smiles: str, charge: int = 0) -> Optional[object]:
    """
    Convert a SMILES string to a 3D molecule object.

    Args:
        smiles: SMILES string
        charge: Molecular charge

    Returns:
        3D molecule object or None if conversion fails
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

        return mol
    except ImportError:
        logger.warning("RDKit not available for 3D conversion")
        return None
    except Exception as e:
        logger.error(f"Error converting SMILES to 3D: {e}")
        return None
