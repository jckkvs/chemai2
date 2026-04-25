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
from backend.pipeline.executor import run_automl_pipeline

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChemAI Nexus API", version="2.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

from backend.api.session import get_session, clear_session

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

class EDAStatsResponse(BaseModel):
    columns: List[str]
    stats: List[dict]

class EDACorrelationResponse(BaseModel):
    columns: List[str]
    matrix: List[List[float]]

class EDADimReductionResponse(BaseModel):
    pca: List[dict]
    tsne: List[dict]
    explained_variance: List[float]

# Import sub-routers
from backend.api.params import router as params_router
from backend.api.data import router as data_router
from backend.api.mixture import router as mixture_router

app.include_router(params_router)
app.include_router(data_router)
app.include_router(mixture_router)

# ── Endpoints ──

@app.post("/api/session/init")
async def init_session():
    """Initialize new session"""
    session_id = str(uuid.uuid4())
    get_session(session_id)
    return {"session_id": session_id}

@app.post("/api/upload", response_model=UploadResponse)
async def upload_data(session_id: Optional[str] = Query(None), file: UploadFile = File(...)):
    """File upload and parsing - migrated from _render_data_load handle_upload"""
    # 同期のため Query(...) を使用。セッションIDがない場合はデフォルトを使用。
    sid = session_id or "default_session"
    session = get_session(sid)
    logger.info(f"Upload start: {file.filename} (session: {sid})")
    
    try:
        contents = await file.read()
        
        # Parse based on extension with high precision and NA handling
        if file.filename.endswith('.csv'):
            df = pd.read_csv(
                io.BytesIO(contents), 
                float_precision='high',
                na_values=['', ' ', 'None', 'nan', 'NaN']
            )
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        
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
    """Execute ML pipeline using AutoMLEngine"""
    session = get_session(session_id)
    df = session.get("df")
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    target_col = session["target_col"]
    task_type = session["task_type"]
    
    try:
        # Run the actual pipeline
        result_dict = await run_automl_pipeline(df, target_col, task_type, cfg.model_dump())
        
        result = AnalysisResult(**result_dict)
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

# ── EDA Endpoints ──

@app.get("/api/eda/stats", response_model=EDAStatsResponse)
async def get_eda_stats(session_id: str):
    """Get detailed statistical summary for numerical columns"""
    session = get_session(session_id)
    df = session.get("df")
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        return EDAStatsResponse(columns=[], stats=[])
    
    stats_df = num_df.describe().T
    stats_df["missing_rate"] = (num_df.isna().mean() * 100).round(1)
    stats_df["skew"] = num_df.skew().round(3)
    stats_df["kurtosis"] = num_df.kurtosis().round(3)
    stats_df = stats_df.reset_index().rename(columns={"index": "column"})
    
    # Clean for JSON
    stats_df = stats_df.where(pd.notnull(stats_df), None)
    
    return EDAStatsResponse(
        columns=list(stats_df.columns),
        stats=stats_df.to_dict(orient="records")
    )

@app.get("/api/eda/correlation", response_model=EDACorrelationResponse)
async def get_eda_correlation(session_id: str, method: str = "pearson"):
    """Get correlation matrix"""
    session = get_session(session_id)
    df = session.get("df")
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return EDACorrelationResponse(columns=[], matrix=[])
    
    corr = num_df.corr(method=method)
    # Handle NaN in correlation
    corr = corr.where(pd.notnull(corr), 0.0)
    
    return EDACorrelationResponse(
        columns=list(corr.columns),
        matrix=corr.values.tolist()
    )

@app.get("/api/eda/dim_reduction", response_model=EDADimReductionResponse)
async def get_eda_dim_reduction(session_id: str):
    """Run PCA and t-SNE for dimensionality reduction"""
    session = get_session(session_id)
    df = session.get("df")
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    num_df = df.select_dtypes(include="number").dropna()
    if num_df.shape[0] < 3 or num_df.shape[1] < 2:
        return EDADimReductionResponse(pca=[], tsne=[], explained_variance=[])
    
    target_col = session.get("target_col")
    X = num_df.drop(columns=[target_col]) if target_col in num_df.columns else num_df
    y = df.loc[X.index, target_col] if target_col in df.columns else None
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=min(10, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    pca_res = [{"pc1": float(row[0]), "pc2": float(row[1]), "target": y.iloc[i] if y is not None else 0} 
               for i, row in enumerate(X_pca)]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_res = [{"v1": float(row[0]), "v2": float(row[1]), "target": y.iloc[i] if y is not None else 0} 
                for i, row in enumerate(X_tsne)]
    
    return EDADimReductionResponse(
        pca=pca_res,
        tsne=tsne_res,
        explained_variance=pca.explained_variance_ratio_.tolist()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
