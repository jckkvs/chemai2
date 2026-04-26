# backend/chem/smiles_utils.py
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import MolStandardize

def validate_smiles_batch(smiles_list: List[str]) -> List[bool]:
    """Validate a batch of SMILES strings"""
    results = []
    for smiles in smiles_list:
        if not isinstance(smiles, str) or not smiles.strip():
            results.append(False)
            continue
        mol = Chem.MolFromSmiles(smiles)
        results.append(mol is not None)
    return results

def standardize_smiles(smiles_list: List[str]) -> List[Optional[str]]:
    """Standardize a batch of SMILES strings using RDKit Standardizer"""
    standardizer = MolStandardize.Standardizer()
    results = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                std_mol = standardizer.standardize(mol)
                results.append(Chem.MolToSmiles(std_mol))
            else:
                results.append(None)
        except:
            results.append(None)
    return results
