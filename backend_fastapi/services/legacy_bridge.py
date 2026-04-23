"""
backend_fastapi/services/legacy_bridge.py
既存 backend/ パッケージを直接呼び出し。ロジックの書き換え・複製は厳禁。
"""
import sys
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

class LegacyBridge:
    @staticmethod
    def execute_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """NiceGUI の _run_analysis と同一ロジックをスレッドプールで実行"""
        try:
            from backend.utils.preprocessing import build_preprocessor
            from backend.pipelines.auto_ml import run_automl_pipeline
            from backend.interpretation.shap_analyzer import analyze_shap

            # 前処理（既存関数と同一シグネチャ）
            preprocessor = build_preprocessor(df, config)
            X, y = preprocessor.transform(df, config["target_col"])
            
            # AutoML 実行
            automl_result = run_automl_pipeline(X, y, config)
            
            # SHAP/XAI（オプション）
            shap_data = None
            if config.get("do_shap"):
                shap_data = analyze_shap(automl_result, X, preprocessor)
                
            return {
                "best_model": getattr(automl_result, "best_model_name", "Unknown"),
                "best_score": float(getattr(automl_result, "best_score", 0.0)),
                "metrics": getattr(automl_result, "metrics", {}),
                "feature_importance": getattr(automl_result, "feature_importance", []),
                "shap_summary": shap_data,
                "predictions": getattr(automl_result, "predictions", []).tolist() if hasattr(getattr(automl_result, "predictions", None), "tolist") else [],
                "n_features": len(X.columns) if hasattr(X, "columns") else X.shape[1],
                "n_samples": len(y)
            }
        except ImportError as e:
            logger.error(f"Legacy backend import failed: {e}")
            raise RuntimeError("既存 backend モジュールが見つかりません。PYTHONPATH を確認してください。")
        except Exception as e:
            logger.error(f"Legacy pipeline execution failed: {e}", exc_info=True)
            raise
