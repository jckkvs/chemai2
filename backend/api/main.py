"""
backend/api/main.py
ChemAI Nexus FastAPI Backend - Production Ready Implementation
"""
from __future__ import annotations

import io
import uuid
import logging
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal, AsyncGenerator
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError

from backend.pipeline.executor import run_automl_pipeline

# ── 構造化ロギング ─────────────────────────────────
def setup_logging():
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("logs/chemai.log", encoding="utf-8", mode="a"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ── 環境設定 ─────────────────────────────────
class Settings(BaseModel):
    env: Literal["development", "staging", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    allowed_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]
    max_file_size: int = 52_428_800  # 50MB
    session_ttl: int = 3600  # 1 hour
    redis_url: Optional[str] = None
    
    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

settings = Settings()

# ── Pydantic v2 型定義 ─────────────────────────────────
class MetricsSchema(BaseModel):
    rows: int = Field(..., ge=1, le=10_000_000)
    cols: int = Field(..., ge=1, le=10_000)
    missing_rate: float = Field(..., ge=0.0, le=1.0)
    numeric_cols: int = Field(..., ge=0)
    
    model_config = ConfigDict(json_schema_extra={
        "example": {"rows": 1000, "cols": 10, "missing_rate": 0.05, "numeric_cols": 8}
    })

class UploadResponse(BaseModel):
    success: bool
    filename: str = Field(..., min_length=1, max_length=255)
    rows: int
    cols: int
    target_col: str
    task_type: Literal["regression", "classification"]
    metrics: MetricsSchema
    preview: List[Dict[str, Any]] = Field(..., max_length=100)
    columns: List[str]
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "filename": "data.csv",
            "rows": 1000,
            "cols": 10,
            "target_col": "Target",
            "task_type": "regression",
            "metrics": {"rows": 1000, "cols": 10, "missing_rate": 0.05, "numeric_cols": 8},
            "preview": [],
            "columns": ["Feature_1", "Feature_2", "Target"]
        }
    })

class ColumnConfig(BaseModel):
    target_col: str = Field(..., min_length=1)
    task_type: Optional[Literal["regression", "classification"]] = None
    exclude_cols: List[str] = Field(default_factory=list)

class PipelineConfig(BaseModel):
    cv_folds: int = Field(default=5, ge=2, le=10)
    num_scaler: Literal["standard", "robust", "minmax", "maxabs", "none"] = "standard"
    num_imputer: Literal["median", "mean", "knn", "iterative", "drop"] = "median"
    cat_encoder: Literal["onehot", "ordinal", "target", "binary"] = "onehot"
    feature_selector: Literal["none", "variance", "selectkbest_f", "selectkbest_mi", 
                              "select_from_model_lasso", "select_from_model_rf", "rfe", "boruta"] = "none"
    selected_models: List[str] = Field(default_factory=list)
    monotonic_constraints: Dict[str, Literal[-1, 0, 1]] = Field(default_factory=dict)
    do_polynomial: bool = False
    poly_degree: int = Field(default=2, ge=2, le=3)
    do_eda: bool = True
    do_prep: bool = True
    do_eval: bool = True

class AnalysisResult(BaseModel):
    status: Literal["pending", "running", "completed", "failed"]
    best_model: Optional[str] = None
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    cv_scores: Optional[List[float]] = None
    feature_importances: Optional[List[Dict[str, Any]]] = None
    message: str
    
    @field_validator("cv_scores")
    @classmethod
    def validate_cv_scores(cls, v):
        if v is not None:
            return [s for s in v if 0.0 <= s <= 1.0]
        return v

class APIError(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = None

# ── セッション管理（抽象化・本番対応）─────────────────────────
class SessionBackend:
    """セッション管理の抽象基底クラス"""
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
    def set(self, session_id: str, data: Dict[str, Any], ttl: int = settings.session_ttl) -> bool:
        raise NotImplementedError
    def delete(self, session_id: str) -> bool:
        raise NotImplementedError
    def exists(self, session_id: str) -> bool:
        raise NotImplementedError
    def cleanup_expired(self) -> int:
        raise NotImplementedError

class InMemorySessionBackend(SessionBackend):
    """開発環境用インメモリ実装"""
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
    
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id not in self._store:
            return None
        if datetime.now().timestamp() - self._timestamps.get(session_id, 0) > settings.session_ttl:
            self.delete(session_id)
            return None
        return self._store[session_id] # Direct reference to allow DF storage
    
    def set(self, session_id: str, data: Dict[str, Any], ttl: int = settings.session_ttl) -> bool:
        self._store[session_id] = data
        self._timestamps[session_id] = datetime.now().timestamp()
        return True
    
    def delete(self, session_id: str) -> bool:
        self._store.pop(session_id, None)
        self._timestamps.pop(session_id, None)
        return True
    
    def exists(self, session_id: str) -> bool:
        return session_id in self._store and \
               datetime.now().timestamp() - self._timestamps.get(session_id, 0) <= settings.session_ttl
    
    def cleanup_expired(self) -> int:
        now = datetime.now().timestamp()
        expired = [sid for sid, ts in self._timestamps.items() if now - ts > settings.session_ttl]
        for sid in expired:
            self.delete(sid)
        return len(expired)

# Global backend instance
_GLOBAL_BACKEND = InMemorySessionBackend()

# ── 依存性注入 ─────────────────────────────────
def get_session_backend() -> SessionBackend:
    return _GLOBAL_BACKEND

def get_default_session_config() -> Dict[str, Any]:
    """新規セッションのデフォルト設定"""
    return {
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
        "pipeline_result": None,
        "created_at": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat()
    }

async def get_request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))

# ── ユーティリティ関数 ─────────────────────────────────
def validate_file(file: UploadFile) -> tuple[bytes, str]:
    filename = file.filename or ""
    if not any(filename.lower().endswith(ext) for ext in [".csv", ".xlsx", ".xls"]):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > settings.max_file_size:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        return content, filename
    except Exception as e:
        logger.error(f"File read error: {e}")
        raise HTTPException(status_code=500, detail="Failed to read file")

def parse_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content), float_precision="high",
                             na_values=["", " ", "NA", "null"], keep_default_na=True,
                             encoding_errors="ignore")
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("Unsupported format")
        
        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = df[col].astype("float64")
        
        exclude_cols = ["Sample_ID", "Category", "ID", "id"]
        df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
        return df
    except Exception as e:
        logger.error(f"Parse error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

def auto_detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    target_col = "Target" if "Target" in df.columns else df.columns[-1]
    task_type = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
    return target_col, task_type

def serialize_preview(df: pd.DataFrame, max_rows: int = 8) -> List[Dict[str, Any]]:
    preview = df.head(max_rows)
    result = []
    for _, row in preview.iterrows():
        row_dict = {}
        for col in preview.columns:
            v = row[col]
            if pd.isna(v): row_dict[col] = None
            elif isinstance(v, (np.floating, float)): row_dict[col] = round(float(v), 4)
            elif isinstance(v, (np.integer, int)): row_dict[col] = int(v)
            else: row_dict[col] = str(v)
        result.append(row_dict)
    return result

# ── FastAPI lifespan ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"ChemAI Nexus API starting (env={settings.env})")
    os.makedirs("logs", exist_ok=True)
    yield
    logger.info("ChemAI Nexus API shutting down...")

# ── FastAPI アプリケーション ─────────────────────────────────
app = FastAPI(
    title="ChemAI Nexus API",
    description="ケモインフォマティクス機械学習プラットフォーム",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# ── Middleware ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ── Error Handlers ─────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=APIError(error=exc.__class__.__name__, message=exc.detail, details={"path": request.url.path}).model_dump()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=APIError(error=exc.__class__.__name__, message="Internal server error").model_dump()
    )

# ── API Endpoints ─────────────────────────────────

@app.post("/api/session/init", response_model=Dict[str, str], tags=["session"])
async def init_session(backend: SessionBackend = Depends(get_session_backend)):
    session_id = str(uuid.uuid4())
    backend.set(session_id, get_default_session_config())
    logger.info(f"Session initialized: {session_id}")
    return {"session_id": session_id}

@app.delete("/api/session/{session_id}", tags=["session"])
async def close_session(session_id: str, backend: SessionBackend = Depends(get_session_backend)):
    if not backend.exists(session_id): raise HTTPException(status_code=404, detail="Session not found")
    backend.delete(session_id)
    return {"status": "closed", "session_id": session_id}

@app.post("/api/upload", response_model=UploadResponse, tags=["data"])
async def upload_data(
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(None),
    backend: SessionBackend = Depends(get_session_backend)
):
    if session_id and not backend.exists(session_id): raise HTTPException(status_code=404, detail="Invalid session")
    if not session_id:
        session_id = str(uuid.uuid4())
        backend.set(session_id, get_default_session_config())
    
    session = backend.get(session_id)
    try:
        content, filename = validate_file(file)
        df = parse_dataframe(content, filename)
        target_col, task_type = auto_detect_columns(df)
        
        session.update({
            "df": df, "filename": filename, "target_col": target_col, "task_type": task_type,
            "metrics": {"rows": len(df), "cols": len(df.columns), "missing_rate": float(df.isna().mean().mean()), "numeric_cols": int(df.select_dtypes(include="number").shape[1])},
            "preview": serialize_preview(df), "last_accessed": datetime.now().isoformat()
        })
        backend.set(session_id, session)
        return UploadResponse(success=True, filename=filename, rows=len(df), cols=len(df.columns), target_col=target_col, task_type=task_type, metrics=session["metrics"], preview=session["preview"], columns=list(df.columns))
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/info", tags=["data"])
async def get_data_info(session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session or session.get("df") is None: raise HTTPException(status_code=404, detail="No data loaded")
    return {"filename": session["filename"], "columns": list(session["df"].columns), "target_col": session["target_col"], "task_type": session["task_type"], "metrics": session["metrics"], "preview": session["preview"]}

@app.post("/api/config/columns", tags=["config"])
async def update_columns(config: ColumnConfig, session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session: raise HTTPException(status_code=404, detail="Session not found")
    session["target_col"] = config.target_col
    if config.task_type: session["task_type"] = config.task_type
    if config.exclude_cols is not None: session["config"]["exclude_cols"] = config.exclude_cols
    backend.set(session_id, session)
    return {"status": "updated"}

@app.get("/api/pipeline/config", response_model=PipelineConfig, tags=["pipeline"])
async def get_pipeline_config(session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session: raise HTTPException(status_code=404, detail="Session not found")
    return PipelineConfig(**session["config"])

@app.post("/api/pipeline/config", tags=["pipeline"])
async def update_pipeline_config(config: PipelineConfig, session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session: raise HTTPException(status_code=404, detail="Session not found")
    session["config"].update(config.model_dump())
    backend.set(session_id, session)
    return {"status": "updated"}

@app.post("/api/pipeline/run", response_model=AnalysisResult, tags=["pipeline"])
async def run_pipeline(cfg: PipelineConfig, session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session or session.get("df") is None: raise HTTPException(status_code=404, detail="No data loaded")
    
    logger.info(f"Pipeline run: session={session_id}")
    session["automl_result"] = {"status": "running", "message": "Analysis started..."}
    backend.set(session_id, session)
    
    try:
        # Run the actual pipeline
        result_dict = await asyncio.to_thread(
            asyncio.run,
            run_automl_pipeline(session["df"], session["target_col"], session["task_type"], cfg.model_dump())
        )
        
        result = AnalysisResult(**result_dict)
        session["automl_result"] = result.model_dump()
        backend.set(session_id, session)
        return result
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        res = AnalysisResult(status="failed", message=str(e))
        session["automl_result"] = res.model_dump()
        backend.set(session_id, session)
        return res

@app.get("/api/results", response_model=AnalysisResult, tags=["results"])
async def get_results(session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session: raise HTTPException(status_code=404, detail="Session not found")
    return AnalysisResult(**session.get("automl_result", {"status": "pending", "message": "No results"}))

@app.get("/api/eda/stats", tags=["eda"])
async def get_eda_stats(session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session or session.get("df") is None: raise HTTPException(status_code=404, detail="No data")
    numeric_df = session["df"].select_dtypes(include="number")
    stats = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        stats.append({"column": col, "count": len(s), "mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())})
    return {"stats": stats}

@app.get("/api/eda/correlation", tags=["eda"])
async def get_eda_correlation(session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session or session.get("df") is None: raise HTTPException(status_code=404, detail="No data")
    corr = session["df"].select_dtypes(include="number").corr().fillna(0)
    return {"columns": list(corr.columns), "matrix": corr.values.tolist()}

@app.get("/api/eda/dim_reduction", tags=["eda"])
async def get_eda_dim_reduction(session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session or session.get("df") is None: raise HTTPException(status_code=404, detail="No data")
    df = session["df"].select_dtypes(include="number").dropna()
    if len(df) < 10: return {"message": "Too little data"}
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(df.values)
    pca = PCA(n_components=2)
    res = pca.fit_transform(X)
    return {"pca": res.tolist(), "explained_variance": pca.explained_variance_ratio_.tolist()}

@app.get("/api/params/models", tags=["params"])
async def get_models(task: str = "regression"):
    try:
        from backend.models.factory import list_models
        return list_models(task=task)
    except: return []

@app.get("/api/params/models/{model_key}/schema", tags=["params"])
async def get_model_schema(model_key: str, task: str = "regression"):
    try:
        from backend.ui.param_schema import introspect_params
        from backend.models.factory import get_model_class
        return introspect_params(get_model_class(model_key, task=task))
    except: raise HTTPException(status_code=404)

@app.get("/api/data/benchmarks", tags=["data"])
async def get_benchmarks():
    return [{"id": "esol", "name": "ESOL"}]

@app.get("/health", tags=["health"])
async def health_check(backend: SessionBackend = Depends(get_session_backend)):
    return {"status": "healthy", "sessions": len(getattr(backend, "_store", {})), "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
