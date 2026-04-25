"""
---FEATURE_METADATA---
name: RDKit Basic Descriptors
description: 分子量、ログ P、TPSA、水素結合供与体/受容体など基本的な物性記述子
category: physicochemical
compute_cost: low
recommended_for: [solubility, permeability, toxicity, admet]
returns_meta_features: true
params:
  normalize:
    type: boolean
    default: true
    description: 特徴量を 0-1 に正規化するかどうか
  selected_descriptors:
    type: multi-select
    default: []
    options: [MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount]
    description: 使用する記述子を選択（空の場合は全部計算）
---END_METADATA---
"""

from typing import List, Dict, Optional, Any
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    from rdkit.Chem.Descriptors import MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount
    RDKit_AVAILABLE = True
except ImportError:
    RDKit_AVAILABLE = False

# 利用可能な記述子のマッピング
DESCRIPTOR_MAP = {
    'MolWt': MolWt,
    'LogP': MolLogP,
    'TPSA': TPSA,
    'NumHDonors': NumHDonors,
    'NumHAcceptors': NumHAcceptors,
    'NumRotatableBonds': NumRotatableBonds,
    'RingCount': RingCount,
}

def compute_features(
    smiles_list: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    SMILES リストから RDKit 記述子を計算
    """
    if not RDKit_AVAILABLE:
        return {
            'feature_names': [],
            'feature_matrix': np.zeros((len(smiles_list), 0)),
            'metadata': {'error': 'RDKit not installed', 'computed': 0, 'skipped': len(smiles_list)}
        }
    
    config = config or {}
    normalize = config.get('normalize', True)
    selected = config.get('selected_descriptors', [])
    
    # 使用する記述子を決定
    descriptors_to_use = selected if selected else list(DESCRIPTOR_MAP.keys())
    
    # 特徴量計算
    feature_matrix = []
    feature_names = []
    computed_count = 0
    skipped_count = 0
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            
            row = []
            for desc_name in descriptors_to_use:
                if desc_name in DESCRIPTOR_MAP:
                    try:
                        value = DESCRIPTOR_MAP[desc_name](mol)
                        row.append(float(value) if value is not None else np.nan)
                    except:
                        row.append(np.nan)
            
            feature_matrix.append(row)
            computed_count += 1
            
        except Exception:
            feature_matrix.append([np.nan] * len(descriptors_to_use))
            skipped_count += 1
        
        if not feature_names:
            feature_names = descriptors_to_use.copy()
    
    feature_matrix = np.array(feature_matrix, dtype=float)
    
    # Handle NaNs (simple imputation for the plugin)
    if feature_matrix.size > 0:
        col_means = np.nanmean(feature_matrix, axis=0)
        inds = np.where(np.isnan(feature_matrix))
        feature_matrix[inds] = np.take(col_means, inds[1])
    
    # 正規化
    if normalize and feature_matrix.size > 0:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
    
    return {
        'feature_names': feature_names,
        'feature_matrix': feature_matrix,
        'metadata': {
            'computed': computed_count,
            'skipped': skipped_count,
            'descriptors_used': descriptors_to_use,
            'normalized': normalize
        }
    }
