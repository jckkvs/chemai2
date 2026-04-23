"""
backend_fastapi/services/chem_service.py
化学記述子計算のサービス層 - 既存 backend.chem.* adapters を統合
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Callable, Optional
import importlib

logger = logging.getLogger(__name__)

class ChemDescriptorService:
    """SMILES記述子計算サービス"""
    
    # エンジン設定マッピング（既存 backend adapters に対応）
    ENGINE_CONFIGS = {
        "rdkit": ("backend.chem.rdkit_adapter", "RDKitAdapter", {"compute_fp": True}),
        "mordred": ("backend.chem.mordred_adapter", "MordredAdapter", {"selected_only": True}),
        "group_contrib": ("backend.chem.group_contrib_adapter", "GroupContribAdapter", {}),
        "descriptastorus": ("backend.chem.descriptastorus_adapter", "DescriptaStorusAdapter", {}),
        "molai": ("backend.chem.molai_adapter", "MolAIAdapter", {"n_components": 6}),
        "skfp": ("backend.chem.skfp_adapter", "SkfpAdapter", {"fp_types": ["ECFP", "MACCS"]}),
        "uma": ("backend.chem.uma_adapter", "UMAAdapter", {}),
        "mol2vec": ("backend.chem.mol2vec_adapter", "Mol2VecAdapter", {}),
        "padel": ("backend.chem.padel_adapter", "PaDELAdapter", {}),
        "molfeat": ("backend.chem.molfeat_adapter", "MolfeatAdapter", {}),
        "xtb": ("backend.chem.xtb_adapter", "XTBAdapter", {}),
        "unipka": ("backend.chem.unipka_adapter", "UniPkaAdapter", {}),
        "cosmo": ("backend.chem.cosmo_adapter", "CosmoAdapter", {}),
        "chemprop": ("backend.chem.chemprop_adapter", "ChempropAdapter", {}),
    }
    
    def compute_descriptors(
        self,
        smiles_list: List[str],
        engines: List[str],
        options: Dict[str, Any],
        on_progress: Optional[Callable[[int, int, str], None]] = None
    ) -> pd.DataFrame:
        """複数エンジンで記述子を計算し、結合して返す"""
        if not smiles_list:
            return pd.DataFrame()
        
        results: List[pd.DataFrame] = []
        total = len(engines)
        for idx, engine in enumerate(engines):
            if engine not in self.ENGINE_CONFIGS:
                logger.warning(f"Unknown engine: {engine}, skipping")
                continue
            module_path, class_name, default_kwargs = self.ENGINE_CONFIGS[engine]
            # user options may override defaults
            kwargs = {**default_kwargs, **options}
            try:
                mod = importlib.import_module(module_path)
                adapter_cls = getattr(mod, class_name)
                adapter = adapter_cls(**kwargs)
                df_eng = adapter.compute(smiles_list)
                if df_eng is not None and not df_eng.empty:
                    # prefix columns to avoid collisions
                    df_eng.columns = [f"{engine}_{col}" if not col.startswith(f"{engine}_") else col for col in df_eng.columns]
                    results.append(df_eng.reset_index(drop=True))
                if on_progress:
                    on_progress(idx + 1, total, f"{engine} 完了")
            except ImportError as e:
                logger.warning(f"Engine {engine} not available: {e}")
            except Exception as e:
                logger.error(f"Engine {engine} calculation failed: {e}", exc_info=True)
        if not results:
            # return empty DF with proper index for SMILES rows
            return pd.DataFrame(index=range(len(smiles_list)))
        df_combined = pd.concat(results, axis=1)
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
        # ensure numeric dtypes where possible
        df_combined = df_combined.apply(pd.to_numeric, errors='coerce')
        return df_combined
    
    def get_available_engines(self) -> List[Dict[str, Any]]:
        """利用可能なエンジン一覧を取得"""
        available = []
        for engine, (module_path, class_name, _) in self.ENGINE_CONFIGS.items():
            try:
                mod = importlib.import_module(module_path)
                getattr(mod, class_name)
                available.append({"key": engine, "name": class_name.replace('Adapter', ''), "available": True})
            except ImportError:
                available.append({"key": engine, "name": class_name.replace('Adapter', ''), "available": False, "reason": "Module not installed"})
        return available

# expose a singleton for quick import
chem_service = ChemDescriptorService()
