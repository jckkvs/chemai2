"""
backend_fastapi/tasks/analysis_tasks.py
ML解析パイプラインの Celery タスク
"""
import logging
from backend_fastapi.celery_app import celery_app
from backend_fastapi.services.analysis_service import AnalysisService
import pandas as pd
import io
import asyncio

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="tasks.run_analysis")
def run_analysis_task(self, config, file_bytes_base64, filename):
    """
    ML解析をバックグラウンドで実行
    """
    import base64
    file_bytes = base64.b64decode(file_bytes_base64)
    
    # 進行状況の更新用ヘルパー
    def update_progress(current, total, message):
        self.update_state(
            state="PROGRESS",
            meta={
                "current": current,
                "total": total,
                "percent": round(current / total * 100, 2),
                "message": message
            }
        )

    try:
        service = AnalysisService()
        # 同期的に実行（Worker スレッド内）
        # AnalysisService.run_full_pipeline が async の場合は、event loop を作成して実行
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            service.run_full_pipeline(config, file_bytes, filename)
        )
        return result
    except Exception as e:
        logger.error(f"Analysis task failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}
