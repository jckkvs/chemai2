"""
非同期ジョブマネージャー
ML解析タスクの管理と進捗追跡
"""
import asyncio
import uuid
from typing import Any, Callable, Coroutine, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class JobManager:
    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
        **kwargs
    ) -> str:
        """ジョブを非同期で実行"""
        job_id = str(uuid.uuid4())
        async with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "status": "pending",
                "progress": 0.0,
                "message": "初期化中...",
                "result": None,
                "error": None,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "_on_progress": on_progress,
            }

        async def _runner():
            try:
                async with self._lock:
                    self._jobs[job_id]["status"] = "running"
                    self._jobs[job_id]["progress"] = 0.05
                    self._jobs[job_id]["message"] = "データ検証中..."
                    self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

                # 既存backendロジックの実行
                result = await func(*args, **kwargs)

                async with self._lock:
                    self._jobs[job_id]["status"] = "completed"
                    self._jobs[job_id]["progress"] = 1.0
                    self._jobs[job_id]["message"] = "完了"
                    self._jobs[job_id]["result"] = result
                    self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
            except Exception as e:
                logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
                async with self._lock:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["error"] = str(e)
                    self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
            finally:
                # コールバック通知
                if self._jobs[job_id].get("_on_progress"):
                    try:
                        self._jobs[job_id]["_on_progress"](
                            job_id,
                            self._jobs[job_id]["progress"],
                            self._jobs[job_id]["message"]
                        )
                    except Exception as cb_err:
                        logger.warning(f"Callback error: {cb_err}")

        asyncio.create_task(_runner())
        return job_id

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """ジョブ状態を取得"""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return {"error": "Job not found", "status": "not_found"}
            
            # 内部コールバック参照は外部に漏らさない
            safe_job = {k: v for k, v in job.items() if not k.startswith("_")}
            return safe_job

    async def cancel_job(self, job_id: str) -> bool:
        """ジョブをキャンセル"""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            if job["status"] in ["pending", "running"]:
                job["status"] = "cancelled"
                job["message"] = "キャンセルされました"
                job["updated_at"] = datetime.utcnow().isoformat()
                return True
            return False

# グローバルインスタンス
job_manager = JobManager()
