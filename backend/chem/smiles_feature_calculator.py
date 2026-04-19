"""
SMILES特徴量の計算・キャッシュ管理モジュール

既存の descriptor_sets.py (DescriptorSet) と連携し、
実際の記述子計算を実行して結果をキャッシュする。
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np

from backend.chem import ADAPTER_REGISTRY, BaseChemAdapter
from backend.chem.descriptor_sets import DescriptorSet
from backend.chem.charge_config import ChargeConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureCalculationResult:
    """特徴量計算の結果を保持"""
    features: pd.DataFrame  # 計算された特徴量 (n_samples x n_features)
    smiles_list: List[str]  # 入力SMILES
    metadata: Dict[str, Any] = field(default_factory=dict)  # 計算メタデータ
    failed_smiles: List[str] = field(default_factory=list)  # 計算失敗したSMILES
    
    @property
    def feature_columns(self) -> List[str]:
        return self.features.columns.tolist()
    
    @property
    def n_samples(self) -> int:
        return len(self.features)
    
    @property
    def n_features(self) -> int:
        return self.features.shape[1]


class SMILESFeatureCalculator:
    """
    SMILESから特徴量を計算・キャッシュするクラス
    
    既存の DescriptorSet 設定に基づき、
    利用可能なアダプターで記述子を計算し、結果をキャッシュする。
    """
    
    CACHE_DIR = Path.home() / ".chemai" / "feature_cache"
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 利用可能なアダプターの事前チェック
        self.available_adapters: Dict[str, BaseChemAdapter] = {}
        for name, adapter_cls in ADAPTER_REGISTRY.items():
            try:
                # 引数なしで初期化可能なもののみ
                adapter = adapter_cls()
                if adapter.is_available():
                    self.available_adapters[name] = adapter
            except Exception as e:
                logger.debug(f"Adapter {name} unavailable: {e}")
    
    def _make_cache_key(
        self,
        smiles_list: List[str],
        descriptor_set: DescriptorSet,
        charge_config: Optional[ChargeConfig] = None,
    ) -> str:
        """キャッシュキーを生成（SMILES + 設定のハッシュ）"""
        # SMILESのソート済みハッシュ
        smiles_hash = hashlib.md5(
            json.dumps(sorted(smiles_list), ensure_ascii=False).encode()
        ).hexdigest()[:16]
        
        # 設定のハッシュ
        set_hash = hashlib.md5(
            json.dumps(descriptor_set.to_dict(), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        
        # 電荷設定のハッシュ
        charge_hash = ""
        if charge_config:
            charge_hash = hashlib.md5(
                json.dumps(charge_config.__dict__, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
        
        return f"{smiles_hash}_{set_hash}_{charge_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        return self.CACHE_DIR / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[FeatureCalculationResult]:
        """キャッシュから結果を読み込む"""
        if not self.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                logger.info(f"Cache hit: {cache_key}")
                return result
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return None
    
    def _save_to_cache(
        self,
        cache_key: str,
        result: FeatureCalculationResult,
    ) -> None:
        """結果をキャッシュに保存"""
        if not self.cache_enabled:
            return
        
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"Cache saved: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _adapter_name_to_flag(self, adapter_name: str) -> Optional[str]:
        """アダプター名 → engine_flags のキーに変換"""
        mapping = {
            "RDKit": None,  # RDKitは常に有効
            "Mordred": "use_mordred",
            "xTB": "use_xtb",
            "COSMO-RS": "use_cosmo",
            "UniPKa": "use_unipka",
            "GroupContrib": "use_contrib",
            "UMA": "use_uma",
            "scikit-FP": "use_skfp",
            "PaDEL": "use_padel",
            "DescriptaStorus": "use_ds",
            "Mol2Vec": "use_mol2vec",
            "Molfeat": "use_molfeat",
            "Chemprop": "use_chemprop",
            "MolAI": "use_molai",
        }
        return mapping.get(adapter_name)
    
    def calculate(
        self,
        smiles_list: Union[List[str], pd.Series],
        descriptor_set: DescriptorSet,
        charge_config: Optional[ChargeConfig] = None,
        progress_callback: Optional[callable] = None,
    ) -> FeatureCalculationResult:
        """
        SMILESリストから特徴量を計算
        """
        if isinstance(smiles_list, pd.Series):
            smiles_list = smiles_list.dropna().astype(str).tolist()
        
        # ユニークなSMILESのみを処理
        unique_smiles = list(dict.fromkeys(smiles_list))
        
        # キャッシュチェック
        cache_key = self._make_cache_key(unique_smiles, descriptor_set, charge_config)
        cached = self._load_from_cache(cache_key)
        if cached:
            # 元の順序に復元 (reindex は DataFrame に対して行う)
            lookup = cached.features
            # unique_smiles 順に並んでいるので、smiles_list 順に並べ替え
            # ただし、lookup のインデックスが unique_smiles である必要がある
            # adapter.compute の結果は smiles_list と同じ長さ・順序
            # ここでは unique_smiles に対して計算してキャッシュしているので問題ない
            features_mapped = lookup.reindex(range(len(unique_smiles))).iloc[[unique_smiles.index(s) for s in smiles_list]]
            features_mapped.index = range(len(smiles_list))

            return FeatureCalculationResult(
                features=features_mapped,
                smiles_list=smiles_list,
                metadata=cached.metadata,
                failed_smiles=cached.failed_smiles,
            )
        
        # 特徴量計算
        all_features: List[pd.DataFrame] = []
        failed_smiles: List[str] = []
        metadata: Dict[str, Any] = {"engines_used": [], "n_total": len(unique_smiles)}
        
        # RDKitは常に計算
        if "RDKit" in self.available_adapters:
            try:
                adapter = self.available_adapters["RDKit"]
                result = adapter.compute(unique_smiles, charge_config=charge_config)
                df = result.descriptors.add_prefix("rdkit_")
                all_features.append(df)
                metadata["engines_used"].append("RDKit")
                metadata["rdkit_success"] = result.success_rate
            except Exception as e:
                logger.warning(f"RDKit calculation failed: {e}")
        
        # 他のエンジン
        for adapter_name, adapter in self.available_adapters.items():
            if adapter_name == "RDKit":
                continue
            
            flag_key = self._adapter_name_to_flag(adapter_name)
            if not flag_key or not descriptor_set.engine_flags.get(flag_key, False):
                continue
            
            try:
                kwargs = {}
                if charge_config and adapter_name in ["xTB", "UniPKa", "COSMO-RS"]:
                    kwargs["charge_config"] = charge_config
                if adapter_name == "MolAI":
                    kwargs["n_components"] = descriptor_set.molai_n_components
                
                result = adapter.compute(unique_smiles, **kwargs)
                
                if result.n_descriptors > 0:
                    prefix = adapter_name.lower().replace("-", "_")
                    df = result.descriptors.add_prefix(f"{prefix}_")
                    all_features.append(df)
                    metadata["engines_used"].append(adapter_name)
                    metadata[f"{prefix}_success"] = result.success_rate
                    
                    if result.failed_indices:
                        failed_smiles.extend([unique_smiles[i] for i in result.failed_indices])
            except Exception as e:
                logger.warning(f"{adapter_name} calculation failed: {e}")
                metadata[f"{adapter_name}_error"] = str(e)
        
        if not all_features:
            raise RuntimeError("No features could be calculated with the specified set")
        
        combined = pd.concat(all_features, axis=1)
        
        # 後処理
        # 定数列削除
        combined = combined.loc[:, combined.std() > 1e-10]
        # 重複列削除
        combined = combined.T.drop_duplicates().T
        
        # スケーリング
        for col in combined.columns:
            min_val, max_val = combined[col].min(), combined[col].max()
            if max_val > min_val and not np.isnan(min_val):
                combined[col] = (combined[col] - min_val) / (max_val - min_val)
        
        # 結果オブジェクト作成 (unique_smiles に対して保存)
        result_obj = FeatureCalculationResult(
            features=combined,
            smiles_list=unique_smiles,
            metadata=metadata,
            failed_smiles=list(set(failed_smiles)),
        )
        
        # キャッシュ保存 (unique_smiles 単位)
        self._save_to_cache(cache_key, result_obj)
        
        # 元の順序 (smiles_list) にマッピングして返す
        features_mapped = combined.iloc[[unique_smiles.index(s) for s in smiles_list]]
        features_mapped.index = range(len(smiles_list))
        
        if progress_callback:
            progress_callback(1, 1)
            
        return FeatureCalculationResult(
            features=features_mapped,
            smiles_list=smiles_list,
            metadata=metadata,
            failed_smiles=list(set(failed_smiles)),
        )
