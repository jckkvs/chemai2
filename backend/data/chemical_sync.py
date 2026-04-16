"""
backend/data/chemical_sync.py

Chemical Structure Synchronization Engine
"""
import pandas as pd
from typing import Optional, List, Dict
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

import io
import base64

class ChemicalStructureSync:
    """
    SMILES列を検出し、構造表示を管理するクラス。
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.smiles_col = self._detect_smiles_col()
        self._img_cache: Dict[str, str] = {}

    def _detect_smiles_col(self) -> Optional[str]:
        for col in self.df.columns:
            if "smiles" in col.lower():
                return col
        return None

    def get_structure_b64(self, index: int, size: tuple = (300, 300)) -> str:
        """指定したインデックスのSMILESをBase64画像に変換する。"""
        if not HAS_RDKIT or self.smiles_col is None:
            return ""
        
        try:
            smiles = self.df.iloc[index][self.smiles_col]
            if pd.isna(smiles) or not isinstance(smiles, str):
                return ""
            
            if smiles in self._img_cache:
                return self._img_cache[smiles]
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            
            # 2D 座標生成
            AllChem.Compute2DCoords(mol)
            
            img = Draw.MolToImage(mol, size=size)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            
            self._img_cache[smiles] = b64
            return b64
        except Exception:
            return ""

    def get_smiles(self, index: int) -> str:
        if self.smiles_col and index < len(self.df):
            return str(self.df.iloc[index][self.smiles_col])
        return ""
