import logging
import re
from typing import Dict, Any, Union, List, Optional

logger = logging.getLogger(__name__)

def apply_monotonicity_constraints(
    estimator: Any,
    pipeline: Any,
    constraints_dict: Dict[str, Union[int, Dict[str, Any]]]
) -> Any:
    """
    変数単位の制約辞書を、選択モデルのネイティブ形式へ自動マッピングして適用する。
    
    拡張機能:
    - __set__:<set_id> : 特定のSMILES特徴量セット全体に制約を適用
    - __group__:<group_name> : 自由定義されたグループ全体に制約を適用
    """
    if not constraints_dict:
        return estimator

    try:
        feature_order = list(pipeline.get_feature_names_out())
    except AttributeError:
        logger.warning("Pipelineがget_feature_names_out()をサポートしていません。制約適用をスキップします。")
        return estimator

    def _get_clean_name(s4_name: str) -> str:
        # preprocess__num__set1_RDKit_MW -> set1_RDKit_MW
        # 最後の __ より後を取得
        return s4_name.split("__")[-1]

    # 方向性のマッピング: 1/-1/0
    def _resolve_val(cfg: Any) -> int:
        if isinstance(cfg, int):
            return cfg
        if isinstance(cfg, dict):
            direction = cfg.get("direction", "none")
            if direction == "increasing": return 1
            if direction == "decreasing": return -1
        return 0

    # 事前計算: グループとセットの制約を抽出
    set_constraints = {}
    group_constraints = {}
    for k, v in constraints_dict.items():
        if k.startswith("__set__:"):
            set_constraints[k.replace("__set__:", "")] = _resolve_val(v)
        elif k.startswith("__group__:"):
            group_constraints[k.replace("__group__:", "")] = _resolve_val(v)

    # 制約配列の構築
    constraint_values: List[int] = []
    applied_count: int = 0
    
    for feat in feature_order:
        clean_feat = _get_clean_name(feat)
        val = 0
        
        # 1. 直接指定（最優先）
        if clean_feat in constraints_dict:
            val = _resolve_val(constraints_dict[clean_feat])
        
        # 2. セット指定 (set1_engine_...)
        if val == 0:
            for set_id, s_val in set_constraints.items():
                if clean_feat.startswith(f"{set_id}_"):
                    val = s_val
                    break
        
        # 3. グループ指定 (現状は clean_feat が group に含まれているかリスト等で持つ必要があるが、
        #    シンプルにキーマッチングを検討。将来的にグループメンバシップ情報を state から引き継ぐ設計にする)
        # TODO: グループメンバシップ情報を constraints_dict に含める

        # Fallback: one-hot encoded categories (col_val)
        if val == 0:
            base_col = clean_feat.split("_")[0]
            if base_col in constraints_dict:
                val = _resolve_val(constraints_dict[base_col])

        constraint_values.append(val)
        if val != 0:
            applied_count += 1

    if applied_count == 0:
        return estimator

    # モデル別適用
    model_name = estimator.__class__.__name__.lower()

    if any(m in model_name for m in ["lgbm", "lightgbm"]):
        estimator.set_params(monotone_constraints=constraint_values)
        logger.info(f"LightGBM: {applied_count}変数に単調性制約を適用しました")
        
    elif any(m in model_name for m in ["xgb", "xgboost"]):
        estimator.set_params(monotone_constraints=str(tuple(constraint_values)))
        logger.info(f"XGBoost: {applied_count}変数に単調性制約を適用しました")
        
    elif "catboost" in model_name:
        active_dict = {f: v for f, v in zip(feature_order, constraint_values) if v != 0}
        estimator.set_params(monotone_constraints=active_dict)
        logger.info(f"CatBoost: {len(active_dict)}変数に単調性制約を適用しました")
        
    elif any(m in model_name for m in ["histgradient", "hist"]):
        estimator.set_params(monotonic_cst=constraint_values)
        logger.info(f"HistGradientBoosting: {applied_count}変数に単調性制約を適用しました")
        
    else:
        logger.info(f"{model_name} は単調性制約にネイティブ対応していません。設定は保持されますが適用されません。")

    # 特徴量ごとの制約結果をインスタンスに保持（後でUI等で確認用）
    estimator.resolved_constraints_ = tuple(constraint_values)

    return estimator
