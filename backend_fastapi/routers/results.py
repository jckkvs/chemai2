"""
結果関連APIエンドポイント
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/results", tags=["results"])

# 仮のストレージ（実際はデータベースまたはファイルシステムを使用）
_results_store: Dict[str, Dict[str, Any]] = {}

@router.get("/{job_id}")
async def get_results(job_id: str):
    """解析結果を取得"""
    result = _results_store.get(job_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return result

@router.get("/{job_id}/metrics")
async def get_metrics(job_id: str):
    """評価指標を取得"""
    result = _results_store.get(job_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return result.get("metrics", {})

@router.get("/{job_id}/feature_importance")
async def get_feature_importance(job_id: str):
    """特徴量重要度を取得"""
    result = _results_store.get(job_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return result.get("feature_importance", [])

@router.get("/{job_id}/predictions")
async def get_predictions(job_id: str):
    """予測結果を取得"""
    result = _results_store.get(job_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return result.get("predictions", [])
