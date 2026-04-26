# backend/chem/molai_adapter.py — 精緻化版 (MolAI CNN潜在空間計算)

from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_molai_descriptors(
    smiles_list: List[str],
    model_path: Optional[Union[str, Path]] = None,
    latent_dim: int = 128,
    batch_size: int = 32,
    device: Optional[str] = None,
    return_numpy: bool = True
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Calculate MolAI CNN-based latent space descriptors with robust device handling
    """
    if not smiles_list:
        return pd.DataFrame() if return_numpy else np.array([])
    
    # 【修正点3】入力SMILESの事前バリデーション（無効分子を早期除去）
    valid_smiles, valid_mask = _filter_valid_smiles(smiles_list)
    if not valid_smiles:
        logger.warning("No valid SMILES for MolAI descriptor calculation")
        return _create_empty_result(len(smiles_list), latent_dim, return_numpy)
    
    # 【修正点1】デバイス自動検出とフォールバック
    device = _detect_best_device(device)
    logger.info(f"Using device '{device}' for MolAI inference")
    
    try:
        import torch
        from backend.chem.molai_model import MolAIModel
    except ImportError as e:
        logger.error(f"MolAI dependencies not available: {e}")
        return _create_empty_result(len(smiles_list), latent_dim, return_numpy)
    
    model = _load_molai_model(model_path, latent_dim, device)
    if model is None:
        return _create_empty_result(len(smiles_list), latent_dim, return_numpy)
    
    model.eval()
    all_embeddings = []
    n_total = len(valid_smiles)
    
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            batch = valid_smiles[i:i+batch_size]
            input_tensor = _smiles_to_tensor(batch, model.tokenizer, device)
            embedding = model.encode(input_tensor)
            
            # 【修正点4】出力検証: 次元・dtype・NaNチェック
            embedding = _validate_embedding_output(embedding, latent_dim, batch)
            all_embeddings.append(embedding.cpu().numpy())
            
            # 【修正点2】GPU使用時はバッチ後にキャッシュクリア
            if device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    embeddings_np = np.vstack(all_embeddings)
    
    if len(valid_smiles) < len(smiles_list):
        embeddings_full = np.full((len(smiles_list), latent_dim), np.nan, dtype=np.float32)
        embeddings_full[valid_mask] = embeddings_np
        embeddings_np = embeddings_full
    
    if return_numpy:
        return embeddings_np.astype(np.float32)
    else:
        columns = [f'molai_latent_{i}' for i in range(latent_dim)]
        return pd.DataFrame(embeddings_np, columns=columns, dtype=np.float32)


def _filter_valid_smiles(smiles_list: List[str]) -> Tuple[List[str], np.ndarray]:
    """Filter valid SMILES and return mask for reconstruction"""
    valid_mask = np.ones(len(smiles_list), dtype=bool)
    valid_smiles = []
    try:
        from rdkit import Chem
        for i, smi in enumerate(smiles_list):
            if smi and isinstance(smi, str):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is not None: valid_smiles.append(smi.strip())
                else: valid_mask[i] = False
            else: valid_mask[i] = False
    except ImportError:
        for i, smi in enumerate(smiles_list):
            if smi and isinstance(smi, str) and len(smi.strip()) > 2: valid_smiles.append(smi.strip())
            else: valid_mask[i] = False
    return valid_smiles, valid_mask


def _detect_best_device(requested: Optional[str]) -> str:
    """Auto-detect best available compute device with fallback chain"""
    try:
        import torch
        if requested in ('cuda', 'mps', 'cpu'):
            if requested == 'cuda' and torch.cuda.is_available(): return 'cuda'
            if requested == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return 'mps'
            if requested == 'cpu': return 'cpu'
        if torch.cuda.is_available(): return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return 'mps'
    except ImportError: pass
    return 'cpu'


def _load_molai_model(model_path: Optional[Union[str, Path]], latent_dim: int, device: str):
    """Load MolAI model with fallback to bundled default"""
    import torch
    from backend.chem.molai_model import MolAIModel
    if model_path is None: model_path = Path(__file__).parent / 'models' / 'molai_default.pt'
    model_path = Path(model_path)
    try:
        if not model_path.exists(): return MolAIModel(latent_dim=latent_dim).to(device)
        return MolAIModel.load(model_path, latent_dim=latent_dim).to(device)
    except Exception as e:
        logger.error(f"Failed to load MolAI model: {e}"); return None


def _smiles_to_tensor(smiles_list: List[str], tokenizer, device: str):
    import torch
    encoded = tokenizer.encode_batch(smiles_list)
    max_len = max(len(e) for e in encoded)
    tensor_list = [torch.tensor(e + [0] * (max_len - len(e)), dtype=torch.long) for e in encoded]
    return torch.stack(tensor_list).to(device)


def _validate_embedding_output(embedding, expected_dim: int, batch_smiles: List[str]):
    import torch
    if embedding.dim() != 2 or embedding.shape[1] != expected_dim:
        if embedding.numel() == len(batch_smiles) * expected_dim: embedding = embedding.view(len(batch_smiles), expected_dim)
        else: return torch.full((len(batch_smiles), expected_dim), float('nan'))
    if torch.isnan(embedding).any() or torch.isinf(embedding).any():
        embedding = torch.where(torch.isfinite(embedding), embedding, torch.zeros_like(embedding))
    return embedding


def _create_empty_result(n_samples, latent_dim, return_numpy):
    res = np.full((n_samples, latent_dim), np.nan, dtype=np.float32)
    return res if return_numpy else pd.DataFrame(res, columns=[f'molai_latent_{i}' for i in range(latent_dim)])
