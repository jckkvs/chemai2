"""
解析関連APIエンドポイント
"""
from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
import json
import asyncio
import logging
from backend_fastapi.services.job_manager import job_manager
from backend_fastapi.services.analysis_service import analysis_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.post("/run")
async def run_analysis(
    file: UploadFile = File(...),
    config_json: str = Form(...)
):
    """Run analysis pipeline with uploaded file and config JSON"""
    try:
        config = json.loads(config_json)
        file_bytes = await file.read()
        
        async def _execute():
            return await analysis_service.run_full_pipeline(config, file_bytes, file.filename)
        
        job_id = await job_manager.submit(_execute)
        return {"job_id": job_id, "status": "submitted"}
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def get_status(job_id: str):
    status = await job_manager.get_status(job_id)
    if status.get("error") == "Job not found":
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@router.websocket("/ws/progress/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            status = await job_manager.get_status(job_id)
            await websocket.send_json(status)
            if status.get("status") in ["completed", "failed", "cancelled"]:
                break
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        logger.info(f"WebSocket closed for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import asyncio
import logging

from backend_fastapi.services.job_manager import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

class AnalysisRequest(BaseModel):
    """解析リクエスト"""
    config: Dict[str, Any] = Field(..., description="CV、スケーラー、モデル選択等の設定")
    data: Optional[Dict[str, Any]] = Field(None, description="データ情報")

@router.post("/run")
async def run_analysis(req: AnalysisRequest):
    """解析を実行"""
    try:
        # 既存のbackendロジックを呼び出すラッパー関数
        async def _execute_analysis():
            # TODO: 既存のbackend.pipelines.auto_mlをインポートして実行
            # from backend.pipelines.auto_ml import run_full_pipeline
            # return await run_full_pipeline(config=req.config)
            
            # 仮の実装
            await asyncio.sleep(2)
            return {
                "status": "completed",
                "best_model": "random_forest",
                "best_score": 0.85,
                "metrics": {
                    "r2": 0.85,
                    "rmse": 1.23
                }
            }

        # 進捗コールバック
        def on_progress(job_id: str, progress: float, message: str):
            logger.info(f"Job {job_id}: {progress*100:.1f}% - {message}")

        job_id = await job_manager.submit(_execute_analysis, on_progress=on_progress)
        
        return {
            "job_id": job_id,
            "status": "submitted",
            "message": "解析ジョブを開始しました"
        }
    except Exception as e:
        logger.error(f"Analysis start failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def get_analysis_status(job_id: str):
    """解析ジョブのステータスを取得"""
    status = await job_manager.get_status(job_id)
    
    if "error" in status and status.get("error") == "Job not found":
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status

@router.delete("/cancel/{job_id}")
async def cancel_analysis(job_id: str):
    """解析ジョブをキャンセル"""
    success = await job_manager.cancel_job(job_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job (already finished or not found)")
    
    return {"status": "cancelled", "message": "ジョブをキャンセルしました"}

@router.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocketで進捗を配信"""
    await websocket.accept()
    
    try:
        while True:
            status = await job_manager.get_status(job_id)
            
            if "error" in status:
                await websocket.send_json({"error": "Job not found"})
                break
            
            await websocket.send_json(status)
            
            # 完了または失敗で終了
            if status["status"] in ["completed", "failed", "cancelled"]:
                break
            

from fastapi.responses import StreamingResponse
import json

@router.get("/analysis/stream/{job_id}")
async def stream_progress(job_id: str):
    """SSEによる進捗配信（WebSocket互換/フォールバック用）"""
    async def event_generator():
        while True:
            status = await job_manager.get_status(job_id)
            yield f"event: progress\ndata: {json.dumps(status)}\n\n"
            if status.get("status") in ("completed", "failed", "cancelled", "not_found"):
                yield f"event: complete\ndata: {json.dumps(status)}\n\n"
                break
            await asyncio.sleep(0.8)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
