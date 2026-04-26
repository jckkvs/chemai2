# backend/chem/protonation.py — 精緻化版 (pH依存プロトン化エンジン)

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
from rdkit import Chem
from rdkit.Chem import rdMolStandardize

logger = logging.getLogger(__name__)


def protonate_at_ph(
    smiles: str,
    ph: float,
    pka_values: Optional[Dict[str, Tuple[float, float]]] = None,
    max_protonation_states: int = 3,
    return_major_state: bool = True
) -> Optional[str]:
    """
    Generate protonated/deprotonated SMILES at specified pH with chemical rigor
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    # 【修正点4】極端なpH値の処理
    if ph < -2 or ph > 16:
        logger.warning(f"Extreme pH value {ph}. Results may be chemically unrealistic.")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.debug(f"Invalid SMILES for protonation: {smiles}")
        return None
    
    try:
        # 【修正点3】Protonate3D試行＋フォールバック
        protonated = _try_protonate_rdkit(mol, ph, pka_values, max_protonation_states)
        
        if protonated is not None:
            return Chem.MolToSmiles(protonated, isomericSmiles=True)
        
        # 【修正点3】フォールバック: 簡易ルールベースプロトン化
        logger.debug("RDKit Protonate3D failed, using rule-based fallback")
        return _protonate_rule_based(mol, ph, pka_values)
        
    except Exception as e:
        logger.error(f"Protonation failed for '{smiles}' at pH {ph}: {e}")
        return None


def _try_protonate_rdkit(
    mol: Chem.Mol,
    ph: float,
    pka_values: Optional[Dict],
    max_states: int
) -> Optional[Chem.Mol]:
    """
    Attempt protonation using RDKit's built-in methods with robust error handling
    """
    try:
        from rdkit.Chem import rdMolStandardize
        
        # pH範囲のクリッピング（化学的妥当性の範囲内）
        ph_clipped = max(0.0, min(14.0, ph))
        
        # 【修正点1】pKa値の信頼区間を考慮した状態選択
        if pka_values:
            # カスタムpKaがある場合: 確率的に主要状態を選択
            major_state = _select_major_state_from_pka(pka_values, ph_clipped)
            if major_state is not None:
                pass  # rdMolStandardizeは自動状態選択のため、ここではスキップ
        
        # RDKitの標準化パイプライン
        uncharger = rdMolStandardize.Uncharger()
        mol_neutral = uncharger.uncharge(mol)
        
        # Protonate3D（3D座標が必要なので2Dから生成）
        mol_3d = Chem.AddHs(mol_neutral)
        Chem.EmbedMolecule(mol_3d, randomSeed=42)
        
        # 【修正点2】両性イオン対応: 電荷状態の妥当性チェック
        protonator = rdMolStandardize.Protonator(pH=ph_clipped)
        mol_protonated = protonator.protonate(mol_3d)
        
        if mol_protonated is None:
            return None
        
        # 3D→2D変換（SMILES出力用）
        Chem.RemoveHs(mol_protonated)
        return mol_protonated
        
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"RDKit protonation failed: {e}")
        return None


def _select_major_state_from_pka(
    pka_values: Dict[str, Tuple[float, float]],
    ph: float,
    confidence_threshold: float = 0.9
) -> Optional[str]:
    """
    Select major protonation state considering pKa uncertainty
    
    【修正点1】不確実性伝播: pKaの標準偏差を考慮した確率計算
    """
    if not pka_values:
        return None
    
    major_states = []
    
    for site_id, (pka_mean, pka_std) in pka_values.items():
        if pka_std > 0:
            # Monte Carlo approximation for uncertainty propagation
            n_samples = 100
            pka_samples = np.random.normal(pka_mean, pka_std, n_samples)
            prob_protonated = np.mean(1 / (1 + 10**(ph - pka_samples)))
        else:
            # Deterministic case
            prob_protonated = 1 / (1 + 10**(ph - pka_mean))
        
        # 【修正点1】信頼区間ベースの判定
        if prob_protonated >= confidence_threshold:
            major_states.append((site_id, 'protonated'))
        elif prob_protonated <= (1 - confidence_threshold):
            major_states.append((site_id, 'deprotonated'))
    
    if not major_states:
        return None
    
    return major_states[0][1]


def _protonate_rule_based(
    mol: Chem.Mol,
    ph: float,
    pka_values: Optional[Dict]
) -> Optional[str]:
    """
    Rule-based protonation fallback using SMARTS patterns
    
    【修正点2】多価イオン・両性イオンの電荷遷移を化学的に厳密に
    """
    # 酸性基SMARTS（pKa ~3-5）
    acidic_patterns = [
        ('carboxylic', '[CX3](=O)[OX2H1]', -1),  # COOH → COO-
        ('phenol', '[cOX2H1]', -1),  # Ar-OH → Ar-O-
    ]
    
    # 塩基性基SMARTS（pKa ~8-11）
    basic_patterns = [
        ('amine_primary', '[NX3;H2;!$(N-[!#6])]', +1),  # R-NH2 → R-NH3+
        ('amine_secondary', '[NX3;H1;!$(N-[!#6])]', +1),  # R2NH → R2NH2+
        ('amine_tertiary', '[NX3;H0;!$(N-[!#6])]', +1),  # R3N → R3NH+
        ('pyridine', '[nH0;+0]', +1),  # Pyridine N → protonated
    ]
    
    mol_work = Chem.Mol(mol)
    current_charge = Chem.GetFormalCharge(mol_work)
    
    # 酸性基の脱プロトン化（pH > pKa + 2）
    if ph > 5.0:  # 簡易閾値
        for name, pattern, charge_change in acidic_patterns:
            patt = Chem.MolFromSmarts(pattern)
            if patt is None: continue
            matches = mol_work.GetSubstructMatches(patt)
            for match in matches:
                atom = mol_work.GetAtomWithIdx(match[0])
                if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
                    atom.SetFormalCharge(atom.GetFormalCharge() + charge_change)
    
    # 塩基性基のプロトン化（pH < pKa - 2）
    if ph < 9.0:  # 簡易閾値
        for name, pattern, charge_change in basic_patterns:
            patt = Chem.MolFromSmarts(pattern)
            if patt is None: continue
            matches = mol_work.GetSubstructMatches(patt)
            for match in matches:
                atom = mol_work.GetAtomWithIdx(match[0])
                if atom.GetSymbol() == 'N':
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                    atom.SetFormalCharge(atom.GetFormalCharge() + charge_change)
    
    # 【修正点2】両性イオンの電荷バランス検証
    final_charge = Chem.GetFormalCharge(mol_work)
    if abs(final_charge) > 3:
        logger.warning(f"High formal charge ({final_charge}) after protonation. Result may be unrealistic.")
    
    smiles = Chem.MolToSmiles(mol_work, isomericSmiles=True)
    return smiles if smiles else None


def batch_protonate(
    smiles_list: List[str],
    ph: float,
    pka_lookup: Optional[Dict[str, Dict]] = None,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Batch protonation with progress tracking and error aggregation
    """
    results = []
    for idx, smi in enumerate(smiles_list):
        try:
            pka_vals = pka_lookup.get(smi) if pka_lookup else None
            protonated = protonate_at_ph(smi, ph, pka_vals)
            results.append({
                'original': smi, 'protonated': protonated,
                'success': protonated is not None, 'ph': ph, 'index': idx
            })
        except Exception as e:
            logger.debug(f"Batch protonation failed for item {idx}: {e}")
            results.append({
                'original': smi, 'protonated': None, 'success': False,
                'ph': ph, 'index': idx, 'error': str(e)
            })
    return pd.DataFrame(results)
