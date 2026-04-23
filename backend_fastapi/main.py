"""
ChemAI Nexus - FastAPI Backend
既存のbackendロジックを活用したAPIサーバー
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import asyncio
import hashlib
import logging
from datetime import datetime
import sys
from pathlib import Path

# backendへのパスを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend_fastapi.services.job_manager import job_manager
from backend_fastapi.routers import analysis, data, results, eda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChemAI Nexus API",
    description="ChemAI Nexus Backend API",
    version="2.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8085"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーター登録
app.include_router(analysis.router)
app.include_router(data.router)
app.include_router(results.router)
app.include_router(eda.router)

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "backend": "fastapi",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "ChemAI Nexus API",
        "docs": "/docs",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
