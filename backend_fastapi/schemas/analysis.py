from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class AnalysisConfig(BaseModel):
    """NiceGUI state と完全互換の解析設定スキーマ"""
    target_col: str
    task_type: Literal["regression", "classification", "auto"] = "auto"
    smiles_col: Optional[str] = None
    exclude_cols: List[str] = Field(default_factory=list)
    cv_folds: int = Field(5, ge=2, le=10)
    num_scaler: str = "standard"
    num_imputer: str = "median"
    cat_encoder: str = "onehot"
    cat_imputer: str = "most_frequent"
    feature_selector: str = "none"
    n_features_to_select: int = 20
    selected_models: List[str] = Field(default_factory=list)
    model_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    monotonic_constraints: Dict[str, int] = Field(default_factory=dict)
    do_eda: bool = True
    do_shap: bool = True
    do_pca: bool = True
    do_polynomial: bool = False
    poly_degree: int = 2

    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [{
        "target_col": "target", "cv_folds": 5, "selected_models": ["rf", "xgb"]
    }]}}
