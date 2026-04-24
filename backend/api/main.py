"""
backend/api/main.py
FastAPI based ChemAI Nexus backend - Full migration from NiceGUI state management
"""
import io
import uuid
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChemAI Nexus API", version="2.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session State Management (In-memory, replace with Redis for production) ──
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "df": None,
            "filename": None,
            "target_col": None,
            "task_type": "regression",
            "smiles_col": None,
            "config": {
                "num_scaler": "standard",
                "num_imputer": "median",
                "num_transform": "none",
                "cat_encoder": "onehot",
                "cat_imputer": "most_frequent",
                "do_polynomial": False,
                "poly_degree": 2,
                "poly_interaction_only": True,
                "feature_selector": "none",
                "n_features_to_select": 20,
                "selected_models": [],
                "monotonic_constraints": {},
                "do_eda": True,
                "do_prep": True,
                "do_eval": True,
                "do_pca": True,
                "do_shap": True
            },
            "metrics": {},
            "preview": [],
            "automl_result": None,
            "pipeline_result": None
        }
    return SESSIONS[session_id]

def clear_session(session_id: str):
    SESSIONS.pop(session_id, None)

# ── Request/Response Models ──
class UploadResponse(BaseModel):
    success: bool
    filename: str
    rows: int
    cols: int
    target_col: str
    task_type: str
    metrics: dict
    preview: List[dict]
    columns: List[str]

class ColumnConfig(BaseModel):
    target_col: str
    task_type: Optional[str] = None
    exclude_cols: List[str] = []

class PipelineConfig(BaseModel):
    cv_folds: int = 5
    num_scaler: str = "standard"
    num_imputer: str = "median"
    cat_encoder: str = "onehot"
    feature_selector: str = "none"
    selected_models: List[str] = []
    monotonic_constraints: Dict[str, int] = {}
    do_polynomial: bool = False
    poly_degree: int = 2
    do_eda: bool = True
    do_prep: bool = True
    do_eval: bool = True

class AnalysisResult(BaseModel):
    status: str
    best_model: Optional[str] = None
    score: Optional[float] = None
    cv_scores: Optional[List[float]] = None
    feature_importances: Optional[List[dict]] = None
    message: str

# ── Endpoints ──

@app.post("/api/session/init")
async def init_session():
    """Initialize new session"""
    session_id = str(uuid.uuid4())
    get_session(session_id)
    logger.info(f"Initialized session: {session_id}")
    return {"session_id": session_id}

@app.post("/api/upload", response_model=UploadResponse)
async def upload_data(session_id: str = Query(...), file: UploadFile = File(...)):
    """File upload and parsing - migrated from _render_data_load handle_upload"""
    # フロントエンドの axios.params と同期するため Query(...) を使用
    session = get_session(session_id)
    logger.info(f"Upload start: {file.filename} (session: {session_id})")
    
    try:
        contents = await file.read()
        
        # Parse based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents), float_precision='high')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        # Precision guarantee: cast all numeric columns to float64
        for col in df.select_dtypes(include=['float16', 'float32', 'int8', 'int16', 'int32', 'int64']).columns:
            df[col] = df[col].astype('float64')
        
        # Auto-detect target & SMILES columns (migrated from _auto_detect_columns)
        target_col = df.columns[-1]
        smiles_col = None
        for col in df.columns:
            if col.lower() == "smiles":
                smiles_col = col
                break
        
        task_type = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
        
        # Update session state
        session["df"] = df
        session["filename"] = file.filename
        session["target_col"] = target_col
        session["task_type"] = task_type
        session["smiles_col"] = smiles_col
        session["automl_result"] = None
        session["pipeline_result"] = None
        
        # Calculate metrics
        numeric_cols = df.select_dtypes(include='number').shape[1]
        missing_rate = float(df.isna().mean().mean())
        session["metrics"] = {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_rate": missing_rate,
            "numeric_cols": numeric_cols
        }
        
        # Generate preview (migrated from _show_preview)
        preview = df.head(8).to_dict(orient="records")
        for row in preview:
            for k, v in row.items():
                if pd.isna(v):
                    row[k] = None
                elif isinstance(v, float):
                    row[k] = round(v, 4)
        session["preview"] = preview
        
        logger.info(f"✅ Upload successful for {file.filename}")

        return UploadResponse(
            success=True,
            filename=file.filename,
            rows=len(df),
            cols=len(df.columns),
            target_col=target_col,
            task_type=task_type,
            metrics=session["metrics"],
            preview=preview,
            columns=list(df.columns)
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/info")
async def get_data_info(session_id: str):
    """Get current data information"""
    session = get_session(session_id)
    if session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    return {
        "filename": session["filename"],
        "columns": list(session["df"].columns),
        "target_col": session["target_col"],
        "task_type": session["task_type"],
        "metrics": session["metrics"],
        "preview": session["preview"]
    }

@app.post("/api/config/columns")
async def update_columns(session_id: str = Body(...), config: ColumnConfig = Body(...)):
    """Update column configuration"""
    session = get_session(session_id)
    if session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    session["target_col"] = config.target_col
    if config.task_type:
        session["task_type"] = config.task_type
    if config.exclude_cols is not None:
        session["config"]["exclude_cols"] = config.exclude_cols
        
    return {"status": "updated", "target_col": session["target_col"], "task_type": session["task_type"]}

@app.get("/api/pipeline/config")
async def get_pipeline_config(session_id: str):
    """Get current pipeline configuration"""
    session = get_session(session_id)
    return session["config"]

@app.post("/api/pipeline/config")
async def update_pipeline_config(session_id: str = Body(...), config: PipelineConfig = Body(...)):
    """Update pipeline configuration"""
    session = get_session(session_id)
    session["config"].update(config.model_dump())
    return {"status": "updated", "config": session["config"]}

@app.post("/api/pipeline/run", response_model=AnalysisResult)
async def run_pipeline(session_id: str = Body(...), cfg: PipelineConfig = Body(...)):
    """Execute ML pipeline - placeholder for existing backend.models integration"""
    session = get_session(session_id)
    df = session.get("df")
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    target_col = session["target_col"]
    task_type = session["task_type"]
    
    try:
        # TODO: Integrate existing backend.pipeline.executor or backend.models.automl
        # from backend.pipeline.executor import run_automl_pipeline
        # result = await run_automl_pipeline(df, target_col, task_type, cfg.model_dump())
        
        logger.info(f"Pipeline run: target={target_col}, models={cfg.selected_models}")
        
        # Stub response for frontend integration
        result = AnalysisResult(
            status="completed",
            best_model="RandomForest",
            score=0.85,
            cv_scores=[0.82, 0.84, 0.86, 0.85, 0.84],
            feature_importances=[{"name": c, "value": 0.1} for c in df.columns[:5] if c != target_col],
            message="Analysis completed successfully"
        )
        
        session["automl_result"] = result.model_dump()
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results")
async def get_results(session_id: str):
    """Get analysis results"""
    session = get_session(session_id)
    return session.get("automl_result", {"status": "pending", "message": "No results yet"})

@app.delete("/api/session/{session_id}")
async def close_session(session_id: str):
    """Close and cleanup session"""
    clear_session(session_id)
    return {"status": "closed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
