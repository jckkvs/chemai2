"""
backend_fastapi/services/analysis_service.py
既存 backend ロジックを FastAPI から呼び出すためのサービスクラス
"""
import asyncio
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AnalysisService:
    """
    既存の backend パッケージを呼び出して解析を実行する。
    NiceGUI 版と同様の state 辞書を受け取り、処理する。
    """
    
    async def run_full_pipeline(self, config: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        完全なパイプライン実行。
        config には UI から送られた設定（cv_folds, models, scalers等）が含まれる。
        """
        try:
            # 1. 既存 backend の state 形式に整形
            state = self._prepare_state(config, df)
            
            # 2. 既存 backend の解析関数を呼び出す（仮のインポートパス）
            # 実際のプロジェクトに合わせてパスを調整してください
            # from backend.pipelines.auto_ml import run_automl_pipeline
            
            # 例: 既存のロジックを非同期で実行
            # loop = asyncio.get_event_loop()
            # result = await loop.run_in_executor(None, lambda: run_automl_pipeline(state))
            
            # 仮の結果返却（実装時に既存関数と置き換え）
            await asyncio.sleep(2) 
            
            result = {
                "status": "success",
                "best_model": "RandomForest",
                "metrics": {
                    "r2": 0.85,
                    "rmse": 1.23,
                    "mae": 0.95
                },
                "feature_importance": [
                    {"feature": "Feature_1", "importance": 0.45},
                    {"feature": "Feature_2", "importance": 0.30},
                    {"feature": "Feature_3", "importance": 0.15},
                ]
            }
            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise

    def _prepare_state(self, config: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Next.js/FastAPI から来た設定を NiceGUI の state 形式に変換する。
        """
        return {
            "df": df,
            "target_col": config.get("target_col"),
            "cv_folds": config.get("cv_folds", 5),
            "models": config.get("selected_models", []),
            "scaler": config.get("num_scaler", "standard"),
            # ... その他の設定マッピング ...
        }

analysis_service = AnalysisService()
