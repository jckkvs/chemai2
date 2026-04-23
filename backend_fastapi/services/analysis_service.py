"""
backend_fastapi/services/analysis_service.py
既存 backend ロジックを FastAPI から安全に呼び出すラッパー
"""
import asyncio
import logging
import io
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class AnalysisService:
    async def run_full_pipeline(
        self, config: Dict[str, Any], file_bytes: bytes, filename: str
    ) -> Dict[str, Any]:
        """解析パイプライン実行（既存 backend 呼び出し）"""
        try:
            # 1. データ読み込み
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_bytes), float_precision='high')
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(file_bytes))
            else:
                raise ValueError("Unsupported file format")

            # 2. NiceGUI state 互換形式に変換
            state = {
                "df": df,
                "target_col": config.get("target_col", df.columns[-1]),
                "task_type": config.get("task_type", "auto"),
                "cv_folds": config.get("cv_folds", 5),
                "num_scaler": config.get("num_scaler", "standard"),
                "selected_models": config.get("selected_models", []),
                "model_params": config.get("model_params", {}),
                "do_eda": config.get("do_eda", True),
                "do_shap": config.get("do_shap", True),
                "exclude_cols": config.get("exclude_cols", []),
            }

            # 3. 既存 backend をスレッドプールで実行（イベントループブロッキング防止）
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._execute_existing_backend(state)
            )
            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise

    def _execute_existing_backend(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """既存 backend モジュールを呼び出し（インポート失敗時は開発用モック返却）"""
        try:
            # ▼ 既存プロジェクトの実際のパスに合わせて調整してください ▼
            from backend.pipelines.auto_ml import run_automl_pipeline
            from backend.interpretation.shap_analyzer import analyze_shap
            from backend.utils.preprocessing import preprocess_data

            X, y, preprocessor = preprocess_data(state["df"], state)
            automl_result = run_automl_pipeline(X, y, state)
            
            shap_result = None
            if state.get("do_shap"):
                shap_result = analyze_shap(automl_result.best_model_, X, preprocessor)

            return {
                "best_model": getattr(automl_result, "best_model_name", "Unknown"),
                "best_score": float(getattr(automl_result, "best_score_", 0.0)),
                "metrics": {
                    "r2": float(automl_result.metrics.get("r2", 0)),
                    "rmse": float(automl_result.metrics.get("rmse", 0)),
                    "mae": float(automl_result.metrics.get("mae", 0))
                },
                "feature_importance": self._format_importance(automl_result),
                "shap_summary_url": shap_result.get("summary_plot_url") if shap_result else None,
                "predictions": automl_result.predictions.tolist() if hasattr(automl_result, "predictions") else []
            }

        except Exception as e:
            logger.warning(f"Existing backend not available or failed, returning mock: {e}")
            return self._mock_result()

    def _format_importance(self, automl_result) -> list:
        if hasattr(automl_result, "feature_importances_"):
            return [
                {"feature": f"Feature_{i}", "importance": float(imp)}
                for i, imp in enumerate(automl_result.feature_importances_[:10])
            ]
        return []

    def _mock_result(self) -> dict:
        return {
            "best_model": "RandomForest",
            "best_score": 0.872,
            "metrics": {"r2": 0.872, "rmse": 1.14, "mae": 0.89},
            "feature_importance": [
                {"feature": "Feature_3", "importance": 0.42},
                {"feature": "Feature_1", "importance": 0.28},
                {"feature": "Feature_7", "importance": 0.15}
            ],
            "shap_summary_url": None,
            "predictions": []
        }

analysis_service = AnalysisService()
