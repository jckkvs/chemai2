"""
backend/chem/smiles_utils.py — 精緻化版 (validate_smiles_batch / standardize_smiles_batch)

SMILES検証・標準化のエッジケース処理を強化したユーティリティ。
"""

from typing import List, Optional, Literal, Union
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
try:
    from rdkit.Chem import rdMolStandardize
except ImportError:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        rdMolStandardize = None
        logger.warning("rdMolStandardize not available. Standardize features may be limited.")

logger = logging.getLogger(__name__)


def validate_smiles_batch(
    smiles_list: List[Optional[str]],
    require_organic: bool = False,
    min_atoms: int = 1,
    max_atoms: int = 500
) -> List[bool]:
    """
    Validate a batch of SMILES strings with configurable constraints
    
    Args:
        smiles_list: List of SMILES strings (None allowed)
        require_organic: If True, reject molecules containing metals
        min_atoms: Minimum number of heavy atoms required
        max_atoms: Maximum number of heavy atoms allowed
    
    Returns:
        List of bool indicating validity for each input
    """
    results = []
    
    for smi in smiles_list:
        # 【修正点1】None/空文字列の事前チェック
        if not smi or not isinstance(smi, str) or not smi.strip():
            results.append(False)
            continue
        
        smi_stripped = smi.strip()
        
        try:
            # 【修正点1】Molオブジェクト生成とNoneチェック
            mol = Chem.MolFromSmiles(smi_stripped)
            if mol is None:
                results.append(False)
                continue
            
            # 水素追加前の原子数カウント（heavy atomsのみ）
            n_heavy = mol.GetNumHeavyAtoms()
            
            # 【修正点1】原子数範囲チェック
            if n_heavy < min_atoms or n_heavy > max_atoms:
                results.append(False)
                continue
            
            # 【修正点1】有機分子要件チェック（金属元素の存在確認）
            if require_organic:
                has_metal = any(
                    atom.GetAtomicNum() > 36 or  # Beyond Kr
                    atom.GetAtomicNum() in [3, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 
                                           26, 27, 28, 29, 30, 31, 32, 33, 34, 35]  # Common metals
                    for atom in mol.GetAtoms()
                )
                if has_metal:
                    results.append(False)
                    continue
            
            # SanitizeMolで化学的妥当性を最終確認
            Chem.SanitizeMol(mol)
            results.append(True)
            
        except Exception:
            # 【修正点4】例外は個別分子で捕捉しバッチ処理を継続
            results.append(False)
    
    return results


def standardize_smiles_batch(
    smiles_list: List[Optional[str]],
    remove_salts: bool = True,
    normalize_tautomers: bool = True,
    reionize: bool = True,
    uncharge: bool = False,
    stereo_mode: Literal['keep', 'remove', 'explicit'] = 'keep'
) -> List[Optional[str]]:
    """
    Standardize a batch of SMILES with configurable options
    
    Args:
        smiles_list: List of SMILES strings
        remove_salts: Remove salt fragments using RDKit's SaltRemover
        normalize_tautomers: Apply tautomer normalization
        reionize: Apply reionization rules
        uncharge: Remove formal charges
        stereo_mode: How to handle stereochemistry ('keep', 'remove', 'explicit')
    
    Returns:
        List of standardized SMILES (None for invalid inputs)
    """
    results = []
    
    # 【修正点2】初期化はループ外で一度のみ（パフォーマンス）
    remover = SaltRemover.SaltRemover() if remove_salts else None
    
    for smi in smiles_list:
        if not smi or not isinstance(smi, str) or not smi.strip():
            results.append(None)
            continue
        
        smi_stripped = smi.strip()
        
        try:
            # 【修正点1】Mol生成とNoneチェック
            mol = Chem.MolFromSmiles(smi_stripped)
            if mol is None:
                results.append(None)
                continue
            
            # 【修正点2】元構造を破壊しないためcopy()を明示
            mol = Chem.Mol(mol)
            
            # 【修正点3】処理順序最適化: 塩除去→互変異性体→再イオン化→立体化学
            # 1. 塩除去
            if remover:
                mol = remover.StripMol(mol, dontRemoveEverything=True)
                if mol is None or mol.GetNumAtoms() == 0:
                    results.append(None)
                    continue
            
            # 2. 互変異性体・イオン化標準化 (rdMolStandardizeを使用)
            if normalize_tautomers or reionize or uncharge:
                try:
                    # より堅牢な標準化器
                    unbound_mol = rdMolStandardize.Cleanup(mol)
                    if uncharge:
                        unbound_mol = rdMolStandardize.Uncharge().uncharge(unbound_mol)
                    mol = unbound_mol
                except Exception as e:
                    logger.debug(f"rdMolStandardize failed for {smi}: {e}")
            
            # 3. 立体化学処理
            if stereo_mode == 'remove':
                Chem.RemoveStereochemistry(mol)
            elif stereo_mode == 'explicit':
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            
            # 4. 芳香族性再認識
            Chem.Kekulize(mol, clearAromaticFlags=True)
            Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
            
            # 5. 標準SMILES生成
            isomeric = (stereo_mode != 'remove')
            standardized = Chem.MolToSmiles(mol, isomericSmiles=isomeric, canonical=True)
            
            if not standardized:
                results.append(None)
                continue
                
            results.append(standardized)
            
        except Exception as e:
            logger.debug(f"Standardization failed for '{smi}': {type(e).__name__}")
            results.append(None)
    
    return results
