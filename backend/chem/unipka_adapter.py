# backend/chem/unipka_adapter.py — 精緻化版 (pKa計算実行)

from typing import List, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
import logging
import time
import subprocess
import os
import tempfile
import json
from rdkit import Chem
from rdkit.Chem import rdFreeSASA
from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

def calculate_unipka(
    smiles_list: List[str],
    ph_range: Optional[tuple] = None,
    max_retries: int = 3,
    base_timeout: int = 120,
    fallback_to_rdkit: bool = True
) -> pd.DataFrame:
    """
    Calculate pKa values using UniPKa with robust retry and fallback logic
    
    Args:
        smiles_list: List of SMILES strings
        ph_range: (min_ph, max_ph) for calculation range (default: 0-14)
        max_retries: Maximum retry attempts with exponential backoff
        base_timeout: Initial timeout in seconds
        fallback_to_rdkit: Use RDKit approximation if UniPKa fails
    
    Returns:
        pd.DataFrame with pKa predictions
    """
    if not smiles_list:
        return pd.DataFrame()
    
    ph_min, ph_max = ph_range or (0.0, 14.0)
    results = []
    n_success = 0
    
    for idx, smi in enumerate(smiles_list):
        if not smi or not isinstance(smi, str):
            results.append({'pka': np.nan, 'method': 'none'})
            continue
        
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            results.append({'pka': np.nan, 'method': 'none'})
            continue
        
        # 【修正点2】電荷状態の検証と補正
        formal_charge = Chem.GetFormalCharge(mol)
        if abs(formal_charge) > 3:
            logger.warning(f"High formal charge ({formal_charge}) in {smi}, pKa may be unreliable")
        
        pka_val = np.nan
        method_used = 'unipka'
        
        # 【修正点1】指数バックオフ再試行
        last_error = None
        for attempt in range(max_retries):
            timeout = base_timeout * (2 ** attempt)
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    input_file = os.path.join(tmpdir, "input.smi")
                    output_file = os.path.join(tmpdir, "output.json")
                    
                    with open(input_file, 'w') as f:
                        f.write(smi.strip())
                    
                    cmd = [
                        'unipka', '--input', input_file,
                        '--ph-range', f'{ph_min},{ph_max}',
                        '--output', output_file, '--format', 'json'
                    ]
                    
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True,
                        timeout=timeout, cwd=tmpdir
                    )
                    
                    if proc.returncode != 0:
                        raise RuntimeError(f"UniPKa exit code {proc.returncode}: {proc.stderr[:200]}")
                    
                    if not os.path.exists(output_file):
                        raise FileNotFoundError("UniPKa output file not generated")
                    
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    
                    # 【修正点3】スキーマ検証
                    if 'pka' not in data and 'macro_pka' not in data:
                        raise ValueError("Invalid UniPKa output: missing pKa field")
                    
                    pka_val = float(data.get('pka', data.get('macro_pka', np.nan)))
                    n_success += 1
                    last_error = None
                    break  # Success
                    
            except subprocess.TimeoutExpired:
                last_error = f"Timeout after {timeout}s"
                logger.warning(f"UniPKa attempt {attempt+1} timed out for {smi!r}")
            except Exception as e:
                last_error = str(e)
                logger.debug(f"UniPKa attempt {attempt+1} failed for {smi!r}: {e}")
            
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)
                time.sleep(wait_time)
        
        # 【修正点4】フォールバック処理
        if np.isnan(pka_val) and fallback_to_rdkit and last_error:
            try:
                # RDKitベースの簡易pKa推定（酸性/塩基性基の経験則）
                pka_val = _estimate_pka_rdkit(mol)
                method_used = 'rdkit_fallback'
                logger.info(f"Fallback to RDKit pKa approximation for {smi!r}")
            except Exception as fallback_err:
                logger.error(f"Both UniPKa and fallback failed for {smi!r}: {fallback_err}")
        
        results.append({'pka': pka_val, 'method': method_used})
    
    df = pd.DataFrame(results)
    df.index = range(len(df))
    
    success_rate = df['method'].eq('unipka').sum() / len(df) if len(df) > 0 else 0
    fallback_rate = df['method'].eq('rdkit_fallback').sum() / len(df)
    
    logger.info(
        f"UniPKa completed: {n_success}/{len(smiles_list)} success, "
        f"fallback: {fallback_rate:.1%}"
    )
    return df

def _estimate_pka_rdkit(mol) -> float:
    """
    Rough pKa estimation based on functional group heuristics
    【修正点4】簡易フォールバック用（学術用途ではない）
    """
    # 酸性基の簡易検出
    patt_acidic = Chem.MolFromSmarts('[OX2H]')  # OH groups
    patt_carboxylic = Chem.MolFromSmarts('[$([CX3](=O)[OX2H]),$([CX3](=O)[OX1-])]')
    
    acidic_count = mol.GetSubstructMatches(patt_acidic) if patt_acidic else []
    carboxy_count = mol.GetSubstructMatches(patt_carboxylic) if patt_carboxylic else []
    
    # 経験則: カルボン酸 ~4.5, 一般アルコール ~15-16（水溶液中で解離しにくい）
    if carboxy_count:
        return 4.5
    elif acidic_count:
        return 15.0
    return np.nan

class UniPkaAdapter(BaseChemAdapter):
    """
    Uni-pKa (dptech-corp) による pKa 記述子アダプター。
    """
    def __init__(self, ph_range: tuple = (0.0, 14.0), fallback_to_rdkit: bool = True):
        self.ph_range = ph_range
        self.fallback_to_rdkit = fallback_to_rdkit

    @property
    def name(self) -> str: return "unipka"

    @property
    def description(self) -> str:
        return "Uni-pKa による高精度 pKa / LogD / 溶媒和エネルギー予測"

    def is_available(self) -> bool:
        import shutil
        return shutil.which("unipka") is not None

    def compute(self, smiles_list: List[str], **kwargs: Any) -> DescriptorResult:
        df = calculate_unipka(
            smiles_list,
            ph_range=self.ph_range,
            fallback_to_rdkit=self.fallback_to_rdkit
        )
        
        failed_indices = df[df['method'] == 'none'].index.tolist()
        
        return DescriptorResult(
            descriptors=df[['pka']],
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name
        )

    def get_descriptor_names(self) -> List[str]:
        return ["pka"]

    def get_descriptors_metadata(self) -> List[DescriptorMetadata]:
        return [DescriptorMetadata(name="pka", meaning="Predicted pKa value", is_count=False)]
