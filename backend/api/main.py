"""
backend/api/main.py
FastAPI ベースの ChemAI Nexus バックエンド
NiceGUI の state 管理・ファイル読み込みロジックを API 化
"""
import io
import uuid
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChemAI Nexus API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── セッション管理（インメモリ）─
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "df": None,
            "filename": None,
            "target_col": None,
            "task_type": "regression",
            "smiles_col": None,
            "config": {},
            "metrics": {}
        }
    return SESSIONS[session_id]

# ── リクエスト/レスポンスモデル ─
class UploadResponse(BaseModel):
    success: bool
    filename: str
    rows: int
    cols: int
    target_col: str
    task_type: str
    preview: List[dict]
    metrics: dict

class ColumnConfig(BaseModel):
    target_col: str
    task_type: Optional[str] = None

class PipelineConfig(BaseModel):
    cv_folds: int = 5
    num_scaler: str = "standard"
    feature_selector: str = "none"
    selected_models: List[str] = []

# ── エンドポイント ─

@app.post("/api/session/init")
async def init_session():
    """セッション初期化"""
    session_id = str(uuid.uuid4())
    get_session(session_id)
    logger.info(f"Initialized session: {session_id}")
    return {"session_id": session_id}

@app.post("/api/upload", response_model=UploadResponse)
async def upload_data(session_id: str = Body(...), file: UploadFile = File(...)):
    """ファイルアップロード・パース・状態更新（data_tab.py handle_upload 移植）"""
    session = get_session(session_id)
    logger.info(f"Upload start: {file.filename} (session: {session_id})")
    
    try:
        contents = await file.read()
        
        # パース処理
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents), float_precision="high")
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
            
        # 数値精度保証（_render_data_load 内のキャスト処理移植）
        for col in df.select_dtypes(include=['float16', 'float32', 'int8', 'int16', 'int32', 'int64']).columns:
            df[col] = df[col].astype('float64')
            
        # 自動カラム検出（_auto_detect_columns 移植）
        target_col = df.columns[-1]
        smiles_col = None
        for col in df.columns:
            if col.lower() == "smiles":
                smiles_col = col
                break
                
        task_type = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
        
        # セッション状態更新
        session["df"] = df
        session["filename"] = file.filename
        session["target_col"] = target_col
        session["task_type"] = task_type
        session["smiles_col"] = smiles_col
        session["metrics"] = {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_rate": float(df.isna().mean().mean()),
            "numeric_cols": int(df.select_dtypes(include='number').shape[1])
        }
        
        preview = df.head(8).to_dict(orient="records")
        for row in preview:
            for k, v in row.items():
                if pd.isna(v):
                    row[k] = None
                elif isinstance(v, (float, np.float64)):
                    row[k] = round(float(v), 4)
                    
        logger.info(f"✅ Upload successful for {file.filename}")
                    
        return UploadResponse(
            success=True,
            filename=file.filename,
            rows=len(df),
            cols=len(df.columns),
            target_col=target_col,
            task_type=task_type,
            preview=preview,
            metrics=session["metrics"]
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/info")
async def get_data_info(session_id: str):
    """データ情報取得"""
    session = get_session(session_id)
    if session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    return {
        "filename": session["filename"],
        "columns": list(session["df"].columns),
        "target_col": session["target_col"],
        "task_type": session["task_type"],
        "metrics": session["metrics"]
    }

@app.post("/api/config/columns")
async def update_columns(session_id: str = Body(...), config: ColumnConfig = Body(...)):
    """列設定更新"""
    session = get_session(session_id)
    if session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
        
    session["target_col"] = config.target_col
    if config.task_type:
        session["task_type"] = config.task_type
    return {"status": "updated", "target_col": session["target_col"], "task_type": session["task_type"]}

@app.post("/api/pipeline/run")
async def run_pipeline(session_id: str = Body(...), cfg: PipelineConfig = Body(...)):
    """解析実行（既存 backend.pipeline へ委譲するプレースホルダー）"""
    session = get_session(session_id)
    if session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
        
    # TODO: 既存の解析ロジック (backend.models.automl, backend.pipeline.executor) を統合
    logger.info(f"Pipeline run: target={session['target_col']}, models={cfg.selected_models}")
    return {
        "status": "completed",
        "best_model": "RandomForest",
        "score": 0.85,
        "cv_scores": [0.82, 0.84, 0.86]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
