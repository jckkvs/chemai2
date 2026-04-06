from typing import Dict, List, Optional

class FeatureMetadataRegistry:
    """SMILES記述子エンジンの特徴量メタデータを管理"""
    
    def __init__(self):
        self._registry: Dict[str, Dict] = {}
    
    def register_engine(self, engine_name: str, feature_prefixes: List[str], 
                       feature_descriptions: Optional[Dict[str, str]] = None):
        """エンジンの特徴量プレフィックス・説明を登録"""
        self._registry[engine_name] = {
            "prefixes": feature_prefixes,
            "descriptions": feature_descriptions or {},
            "default_constraints": {
                "direction": "unknown",  # 默认: 方向不明
                "linearity": False,
                "strength": 0.5,  # 中等度の制約
                "sigma_range": 3.0
            }
        }
    
    def get_default_constraints(self, feature_name: str) -> Dict:
        """特徴量名からデフォルト制約設定を取得"""
        for engine, config in self._registry.items():
            if any(feature_name.lower().startswith(p.lower()) for p in config["prefixes"]):
                return config["default_constraints"].copy()
        # 未知の特徴量用のデフォルト
        return {"direction": "none", "linearity": False, "strength": 0.0, "sigma_range": 3.0}
    
    def export_for_frontend(self) -> Dict[str, List[str]]:
        """フロントエンド用のプレフィックスマップを出力"""
        return {engine: config["prefixes"] for engine, config in self._registry.items()}

# 初期化と登録例
feature_metadata = FeatureMetadataRegistry()
feature_metadata.register_engine(
    "rdkit", 
    ["rdkit_", "molwt", "logp", "tpsa", "num_hbd", "num_hba", "fp_", "balaban", "bertz"],
    feature_descriptions={"molwt": "分子量", "logp": "オクタノール/水分配係数", "tpsa": "極性表面積"}
)
feature_metadata.register_engine(
    "mordred",
    ["mordred_", "ABC", "ATS", "BCUT", "GETAWAY", "MOR", "WHIM"],
)
# TODO: add others as needed
