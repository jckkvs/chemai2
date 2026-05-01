"""
backend/pipeline/executor.py
High-level execution wrapper for ChemAI ML Pipeline - Bridging FastAPI and AutoMLEngine
"""
import logging
import pandas as pd
from typing import Any, Dict, List, Optional
from backend.models.automl import AutoMLEngine, AutoMLResult

logger = logging.getLogger(__name__)

async def run_automl_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute the full AutoML pipeline using AutoMLEngine
    """
    logger.info(f"Starting AutoML pipeline for target: {target_col} ({task_type})")
    
    # Initialize engine with provided config
    engine = AutoMLEngine(
        task=task_type,
        cv_folds=config.get("cv_folds", 5),
        model_keys=config.get("selected_models"),
        monotonic_constraints_dict=config.get("monotonic_constraints", {}),
        # Map frontend config to engine params
        preprocess_params={
            "numeric_scaler": config.get("num_scaler", "standard"),
            "numeric_imputer": config.get("num_imputer", "median"),
            "categorical_encoder": config.get("cat_encoder", "onehot"),
        }
    )
    
    try:
        # Run the engine
        # Note: AutoMLEngine.run is synchronous, we run it directly here
        # (In a real high-load scenario, use run_in_executor)
        result: AutoMLResult = engine.run(df, target_col)
        
        # Format result for API response
        return {
            "status": "completed",
            "best_model": result.best_model_key,
            "score": float(result.best_score),
            "cv_scores": result.model_details.get(result.best_model_key, {}).get("fold_scores", []),
            "feature_importances": _get_importances(result),
            "message": "Analysis completed successfully"
        }
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise e

def _get_importances(result: AutoMLResult) -> List[Dict[str, Any]]:
    """Extract feature importances from the best pipeline"""
    try:
        # Simplified importance extraction
        # In a real scenario, this would use SHAP or model-specific importances
        model = result.best_pipeline.steps[-1][1]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Get feature names from preprocessor
            if hasattr(result.best_pipeline, "named_steps") and "preprocess" in result.best_pipeline.named_steps:
                try:
                    feature_names = result.best_pipeline.named_steps["preprocess"].get_feature_names_out()
                except:
                    feature_names = [f"Feature {i}" for i in range(len(importances))]
            else:
                feature_names = [f"Feature {i}" for i in range(len(importances))]
                
            return sorted([
                {"name": name, "value": float(imp)}
                for name, imp in zip(feature_names, importances)
            ], key=lambda x: x["value"], reverse=True)[:20]
    except Exception as e:
        logger.warning(f"Failed to extract feature importances: {e}")
    return []
