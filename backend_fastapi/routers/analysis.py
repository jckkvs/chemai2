"""
backend_fastapi/routers/analysis.py
解析関連APIエンドポイント
"""
from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
import json
import asyncio
import logging
from typing import Optional, Dict, Any
from backend_fastapi.services.job_manager import job_manager
from backend_fastapi.services.analysis_service import analysis_service
from backend_fastapi.schemas.analysis import AnalysisConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.post("/run")
async def run_analysis(
    file: UploadFile = File(...),
    config_json: str = Form(...)
):
    """Run analysis pipeline with uploaded file and config JSON"""
    try:
        config_dict = json.loads(config_json)
        # Validate with Pydantic
        config = AnalysisConfig(**config_dict)
        file_bytes = await file.read()
        
        async def _execute():
            return await analysis_service.run_full_pipeline(config.model_dump(), file_bytes, file.filename)
        
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

@router.delete("/cancel/{job_id}")
async def cancel_analysis(job_id: str):
    """解析ジョブをキャンセル"""
    success = await job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job (already finished or not found)")
    return {"status": "cancelled", "message": "ジョブをキャンセルしました"}

@router.websocket("/ws/progress/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    """NiceGUI の通知形式と同一フォーマットで進捗配信"""
    await websocket.accept()
    try:
        while True:
            status = await job_manager.get_status(job_id)
            if status.get("status") == "not_found":
                 await websocket.send_json({"error": "Job not found"})
                 break
            
            # NiceGUI ui.notify 互換形式
            payload = {
                "type": "notification",
                "message": status.get("message", ""),
                "progress": status.get("progress", 0.0),
                "status": status.get("status", "unknown"),
                "timeout": 5000 if status.get("status") in ("completed", "failed") else None
            }
            await websocket.send_json(payload)
            if status.get("status") in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)

@router.get("/stream/{job_id}")
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
