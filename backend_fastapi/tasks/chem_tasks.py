"""
backend_fastapi/tasks/chem_tasks.py
化学記述子計算の Celery タスク
"""
import logging
from backend_fastapi.celery_app import celery_app
from backend_fastapi.services.chem_service import chem_service
import pandas as pd

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="tasks.compute_descriptors")
def compute_descriptors_task(self, smiles_list, engines, options):
    """
    SMILES記述子計算をバックグラウンドで実行
    """
    total = len(engines)
    
    def on_progress(idx, total_eng, message):
        self.update_state(
            state="PROGRESS",
            meta={
                "current": idx,
                "total": total_eng,
                "percent": round(idx / total_eng * 100, 2),
                "message": message
            }
        )

    try:
        # サービス呼び出し（同期的に実行）
        result_df = chem_service.compute_descriptors(
            smiles_list, 
            engines, 
            options, 
            on_progress=on_progress
        )
        
        # 結果を辞書形式で返す（JSONシリアライズ可能にする）
        return {
            "status": "completed",
            "columns": result_df.columns.tolist(),
            "shape": result_df.shape,
            "data": result_df.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Descriptor task failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}
