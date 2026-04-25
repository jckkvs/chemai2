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
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal, AsyncGenerator
from contextlib import asynccontextmanager

import re
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
from sklearn.impute import SimpleImputer

# ── 構造化ロギング設定 ─────────────────────────────────
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
    allowed_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
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
    feature_importances: Optional[List[Dict[str, Any]]] = None # Changed to Any for real data support
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
        return self._store[session_id].copy()
    
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

_GLOBAL_BACKEND = InMemorySessionBackend()

# ── 依存性注入 ─────────────────────────────────
def get_session_backend() -> SessionBackend:
    """環境に応じたセッションバックエンドを返す"""
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
    """リクエスト固有 ID 生成"""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))

# ── ユーティリティ関数 ─────────────────────────────────
def validate_file(file: UploadFile) -> tuple[bytes, str]:
    """ファイルの検証と読み込み"""
    filename = file.filename or ""
    
    # 拡張子検証
    if not any(filename.lower().endswith(ext) for ext in [".csv", ".xlsx", ".xls"]):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use CSV or Excel (.xlsx, .xls)"
        )
    
    # ファイルサイズ検証
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_file_size / 1024 / 1024:.1f}MB"
        )
    
    # 内容読み込み
    try:
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        return content, filename
    except Exception as e:
        logger.error(f"File read error: {e}")
        raise HTTPException(status_code=500, detail="Failed to read file")

def parse_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    """CSV/Excel のパースと前処理"""
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(
                io.BytesIO(content),
                float_precision="high",
                na_values=["", " ", "NA", "N/A", "null", "None"],
                keep_default_na=True,
                encoding="utf-8",
                encoding_errors="ignore"
            )
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("Unsupported format")
        
        # --- データクリーニング (自動除外ロジック) ---
        # 解析対象外となる列名を正規表現やリストで除外
        drop_patterns = ["sample_id", "category", "^id$", "index", "unnamed"]
        cols_to_drop = []
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(re.match(p, col_lower) for p in drop_patterns):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            logger.info(f"Dropping columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # 数値列の精度保証
        for col in df.select_dtypes(include=["float16", "float32", "int8", "int16", "int32", "int64"]).columns:
            df[col] = df[col].astype("float64")
        
        return df
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV file")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {str(e)}")
    except Exception as e:
        logger.error(f"Parse error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

def auto_detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """目的変数・タスクタイプの自動検出"""
    target_col = "Target" if "Target" in df.columns else df.columns[-1]
    task_type = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
    return target_col, task_type

def serialize_preview(df: pd.DataFrame, max_rows: int = 8) -> List[Dict[str, Any]]:
    """DataFrame プレビューの JSON 直列化"""
    preview = df.head(max_rows)
    result = []
    for _, row in preview.iterrows():
        row_dict = {}
        for col in preview.columns:
            v = row[col]
            if pd.isna(v):
                row_dict[col] = None
            elif isinstance(v, (np.floating, float)):
                row_dict[col] = round(float(v), 4)
            elif isinstance(v, (np.integer, int)):
                row_dict[col] = int(v)
            else:
                row_dict[col] = str(v)
        result.append(row_dict)
    return result

# ── FastAPI lifespan 管理 ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """lifespan 管理: 起動・終了時のリソース管理"""
    logger.info(f"ChemAI Nexus API starting (env={settings.env})")
    os.makedirs("logs", exist_ok=True)
    yield
    logger.info("ChemAI Nexus API shutting down...")
    backend = get_session_backend()
    if hasattr(backend, "cleanup_expired"):
        cleaned = backend.cleanup_expired()
        logger.info(f"Cleaned up {cleaned} expired sessions")

# ── FastAPI アプリケーション ─────────────────────────────────
app = FastAPI(
    title="ChemAI Nexus API",
    description="ケモインフォマティクス機械学習プラットフォーム - FastAPI Backend",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

# ── CORS 設定 ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ── Request ID middleware ─────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ── Global Exception Handlers ─────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url.path} [req:{request_id}]")
    return JSONResponse(
        status_code=exc.status_code,
        content=APIError(
            error=exc.__class__.__name__,
            message=exc.detail,
            details={"path": request.url.path, "method": request.method},
            request_id=request_id
        ).model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.error(f"Validation error: {exc.errors()} - {request.url.path} [req:{request_id}]")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=APIError(
            error="ValidationError",
            message="Request validation failed",
            details={"errors": exc.errors(), "body": exc.body},
            request_id=request_id
        ).model_dump()
    )

@app.exception_handler(ValidationError)
async def pydantic_validation_handler(request: Request, exc: ValidationError):
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.error(f"Pydantic validation error: {exc} [req:{request_id}]")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=APIError(
            error="PydanticValidationError",
            message="Response validation failed",
            details={"errors": exc.errors()},
            request_id=request_id
        ).model_dump()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.error(f"Unhandled exception: {exc}", exc_info=True, extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=APIError(
            error=exc.__class__.__name__,
            message="Internal server error",
            details={"path": request.url.path, "method": request.method},
            request_id=request_id
        ).model_dump()
    )

# ── API Endpoints ─────────────────────────────────

@app.post("/api/session/init", response_model=Dict[str, str], tags=["session"])
async def init_session(backend: SessionBackend = Depends(get_session_backend)):
    """Initialize new session"""
    session_id = str(uuid.uuid4())
    backend.set(session_id, get_default_session_config())
    logger.info(f"Session initialized: {session_id}")
    return {"session_id": session_id}

@app.delete("/api/session/{session_id}", response_model=Dict[str, str], tags=["session"])
async def close_session(
    session_id: str,
    backend: SessionBackend = Depends(get_session_backend)
):
    """Close and cleanup session"""
    if not backend.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    backend.delete(session_id)
    logger.info(f"Session closed: {session_id}")
    return {"status": "closed", "session_id": session_id}

@app.post("/api/upload", response_model=UploadResponse, tags=["data"])
async def upload_data(
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(None, alias="session_id"),
    backend: SessionBackend = Depends(get_session_backend),
    request_id: str = Depends(get_request_id)
):
    """File upload and parsing - migrated from _render_data_load handle_upload"""
    if session_id and not backend.exists(session_id):
        raise HTTPException(status_code=404, detail="Invalid session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        backend.set(session_id, get_default_session_config())
    
    session = backend.get(session_id)
    logger.info(f"Upload start: {file.filename} (session: {session_id}) [req:{request_id}]")
    
    try:
        content, filename = validate_file(file)
        df = parse_dataframe(content, filename)
        target_col, task_type = auto_detect_columns(df)
        
        session.update({
            "df": df,
            "filename": filename,
            "target_col": target_col,
            "task_type": task_type,
            "automl_result": None,
            "pipeline_result": None,
            "last_accessed": datetime.now().isoformat()
        })
        
        session["metrics"] = {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_rate": float(df.isna().mean().mean()),
            "numeric_cols": int(df.select_dtypes(include="number").shape[1])
        }
        session["preview"] = serialize_preview(df)
        backend.set(session_id, session)
        
        logger.info(f"Upload completed: {filename} ({len(df)} rows × {len(df.columns)} cols) [req:{request_id}]")
        
        return UploadResponse(
            success=True,
            filename=filename,
            rows=len(df),
            cols=len(df.columns),
            target_col=target_col,
            task_type=task_type,
            metrics=session["metrics"],
            preview=session["preview"],
            columns=list(df.columns)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True, extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/info", response_model=Dict[str, Any], tags=["data"])
async def get_data_info(
    session_id: str = Query(...),
    backend: SessionBackend = Depends(get_session_backend),
    request_id: str = Depends(get_request_id)
):
    """Get current data information"""
    session = backend.get(session_id)
    if not session or session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    return {
        "filename": session["filename"],
        "columns": list(session["df"].columns),
        "target_col": session["target_col"],
        "task_type": session["task_type"],
        "metrics": session["metrics"],
        "preview": session["preview"]
    }

@app.post("/api/config/columns", response_model=Dict[str, Any], tags=["config"])
async def update_columns(
    config: ColumnConfig,
    session_id: str = Query(...),
    backend: SessionBackend = Depends(get_session_backend),
    request_id: str = Depends(get_request_id)
):
    """Update column configuration"""
    session = backend.get(session_id)
    if not session or session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    session["target_col"] = config.target_col
    if config.task_type:
        session["task_type"] = config.task_type
    if config.exclude_cols is not None:
        session["config"]["exclude_cols"] = config.exclude_cols
    session["last_accessed"] = datetime.now().isoformat()
    backend.set(session_id, session)
    
    return {"status": "updated", "target_col": session["target_col"], "task_type": session["task_type"]}

@app.get("/api/pipeline/config", response_model=PipelineConfig, tags=["pipeline"])
async def get_pipeline_config(
    session_id: str = Query(...),
    backend: SessionBackend = Depends(get_session_backend)
):
    """Get current pipeline configuration"""
    session = backend.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return PipelineConfig(**session["config"])

@app.post("/api/pipeline/config", response_model=Dict[str, Any], tags=["pipeline"])
async def update_pipeline_config(
    config: PipelineConfig,
    session_id: str = Query(...),
    backend: SessionBackend = Depends(get_session_backend),
    request_id: str = Depends(get_request_id)
):
    """Update pipeline configuration"""
    session = backend.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session["config"].update(config.model_dump())
    session["last_accessed"] = datetime.now().isoformat()
    backend.set(session_id, session)
    
    return {"status": "updated", "config": session["config"]}

@app.post("/api/pipeline/run", response_model=AnalysisResult, tags=["pipeline"])
async def run_pipeline(
    cfg: PipelineConfig,
    session_id: str = Query(...),
    backend: SessionBackend = Depends(get_session_backend),
    request_id: str = Depends(get_request_id)
):
    """
    実際の ML パイプラインを実行する関数 (Scikit-learn 統合)
    """
    session = backend.get(session_id)
    if not session or session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    df = session["df"]
    target_col = session["target_col"]
    task_type = session["task_type"]
    
    # 目的変数と説明変数に分割
    # 除外列の削除 (Sample_ID, Category などが含まれている場合を想定)
    exclude_cols = [target_col, "Sample_ID", "Category", "id", "ID"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if not feature_cols:
        return AnalysisResult(status="failed", message="有効な特徴量が 0 です")
        
    X = df[feature_cols]
    y = df[target_col]
    
    # 欠損値の補完 (SimpleImputer)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # 前処理: スケーリング
    if cfg.num_scaler == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_imputed)
    elif cfg.num_scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_processed = scaler.fit_transform(X_imputed)
    else:
        X_processed = X_imputed
        
    # モデル選択と学習
    model = None
    score = 0.0
    cv_scores = []
    importances = []
    
    logger.info(f"Training model: {cfg.selected_models or ['default']} [req:{request_id}]")
    
    try:
        if task_type == "regression":
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_processed, y)
            
            cv_scores = cross_val_score(model, X_processed, y, cv=min(5, len(df)), scoring="r2")
            score = float(cv_scores.mean())
            
            importances = [
                {"name": name, "value": float(val)}
                for name, val in zip(feature_cols, model.feature_importances_)
            ]
            
        else: # Classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_processed, y)
            
            cv_scores = cross_val_score(model, X_processed, y, cv=min(5, len(df)), scoring="accuracy")
            score = float(cv_scores.mean())
            
            importances = [
                {"name": name, "value": float(val)}
                for name, val in zip(feature_cols, model.feature_importances_)
            ]
        
        # 重要度でソート
        importances.sort(key=lambda x: x["value"], reverse=True)
        
        result = AnalysisResult(
            status="completed",
            best_model="RandomForest",
            score=round(score, 4),
            cv_scores=[round(s, 4) for s in cv_scores.tolist() if isinstance(cv_scores, np.ndarray)] if isinstance(cv_scores, np.ndarray) else [round(s, 4) for s in cv_scores],
            feature_importances=importances[:10],
            message=f"完了: {task_type} | CV Score: {score:.4f}"
        )
        
        session["automl_result"] = result.model_dump()
        session["last_accessed"] = datetime.now().isoformat()
        backend.set(session_id, session)
        
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return AnalysisResult(status="failed", message=f"Pipeline execution error: {str(e)}")

@app.get("/api/results", response_model=AnalysisResult, tags=["results"])
async def get_results(
    session_id: str = Query(...),
    backend: SessionBackend = Depends(get_session_backend)
):
    """Get analysis results"""
    session = backend.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = session.get("automl_result")
    if result:
        return AnalysisResult(**result)
    return AnalysisResult(status="pending", message="No results yet")

@app.get("/api/eda/stats", response_model=Dict[str, Any], tags=["eda"])
async def get_eda_stats(
    session_id: str = Query(...),
    backend: SessionBackend = Depends(get_session_backend)
):
    """Get detailed statistical summary for numerical columns"""
    session = backend.get(session_id)
    if not session or session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    df = session["df"]
    numeric_df = df.select_dtypes(include="number")
    
    stats = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        stats.append({
            "column": col,
            "count": len(series),
            "mean": float(series.mean()) if len(series) > 0 else None,
            "std": float(series.std()) if len(series) > 1 else None,
            "min": float(series.min()) if len(series) > 0 else None,
            "max": float(series.max()) if len(series) > 0 else None,
            "q25": float(series.quantile(0.25)) if len(series) > 0 else None,
            "q50": float(series.quantile(0.50)) if len(series) > 0 else None,
            "q75": float(series.quantile(0.75)) if len(series) > 0 else None,
        })
    
    return {"columns": list(numeric_df.columns), "stats": stats}

@app.get("/api/eda/correlation", response_model=Dict[str, Any], tags=["eda"])
async def get_eda_correlation(
    session_id: str = Query(...),
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    backend: SessionBackend = Depends(get_session_backend)
):
    """Get correlation matrix"""
    session = backend.get(session_id)
    if not session or session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    df = session["df"].select_dtypes(include="number")
    if df.empty:
        return {"columns": [], "matrix": []}
    
    corr = df.corr(method=method).fillna(0)
    return {
        "columns": list(corr.columns),
        "matrix": corr.round(4).values.tolist()
    }

@app.get("/api/eda/dim_reduction", response_model=Dict[str, Any], tags=["eda"])
async def get_eda_dim_reduction(
    session_id: str = Query(...),
    backend: SessionBackend = Depends(get_session_backend)
):
    """Run PCA and t-SNE for dimensionality reduction"""
    session = backend.get(session_id)
    if not session or session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    df = session["df"].select_dtypes(include="number").dropna()
    if df.shape[1] < 2 or df.shape[0] < 10:
        return {"pca": [], "tsne": [], "explained_variance": [], "message": "Insufficient data"}
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        
        X = StandardScaler().fit_transform(df.values)
        pca = PCA(n_components=min(2, X.shape[1]))
        pca_result = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_.tolist()
        
        sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
        tsne = TSNE(n_components=2, perplexity=min(30, len(sample_idx)-1), random_state=42, n_iter=250)
        tsne_result = tsne.fit_transform(X[sample_idx])
        
        return {
            "pca": pca_result[:100].tolist(),
            "tsne": tsne_result.tolist(),
            "explained_variance": explained_var
        }
    except Exception as e:
        logger.error(f"Dimensionality reduction error: {e}")
        return {"pca": [], "tsne": [], "explained_variance": [], "error": str(e)}

# ── Parameter Schema Endpoints ─────────────────────────────────
@app.get("/api/params/models", response_model=List[Dict[str, Any]], tags=["params"])
async def get_models(task: Literal["regression", "classification"] = "regression"):
    """List available models for a given task"""
    try:
        from backend.models.factory import list_models
        return list_models(task=task, available_only=True)
    except ImportError:
        return []

@app.get("/api/params/models/{model_key}/schema", response_model=Dict[str, Any], tags=["params"])
async def get_model_schema(model_key: str, task: Literal["regression", "classification"] = "regression"):
    """Get dynamic parameter schema for a specific model"""
    try:
        from backend.ui.param_schema import introspect_params
        from backend.models.factory import get_model_class
        model_cls = get_model_class(model_key, task=task)
        return introspect_params(model_cls)
    except Exception as e:
        logger.error(f"Schema fetch error: {e}")
        raise HTTPException(status_code=404, detail=f"Model or schema not found: {model_key}")

@app.get("/api/params/adapters", response_model=List[Dict[str, Any]], tags=["params"])
async def get_adapters():
    """List available SMILES feature adapters"""
    return [
        {"key": name.lower().replace(" ", "_"), "name": name, "module": module}
        for name, module, _, _ in [
            ("RDKit", "backend.chem.rdkit_adapter"),
            ("Mordred", "backend.chem.mordred_adapter"),
            ("GroupContrib", "backend.chem.group_contrib_adapter"),
        ]
    ]

@app.get("/api/params/adapters/{adapter_key}/schema", response_model=Dict[str, Any], tags=["params"])
async def get_adapter_schema(adapter_key: str):
    """Get dynamic parameter schema for a specific SMILES adapter"""
    try:
        from backend.ui.param_schema import introspect_params
        adapter_map = {
            "rdkit": ("backend.chem.rdkit_adapter", "RDKitAdapter"),
            "mordred": ("backend.chem.mordred_adapter", "MordredAdapter"),
        }
        if adapter_key not in adapter_map:
            raise KeyError
        module_path, class_name = adapter_map[adapter_key]
        import importlib
        mod = importlib.import_module(module_path)
        adapter_cls = getattr(mod, class_name)
        return introspect_params(adapter_cls)
    except Exception as e:
        logger.error(f"Adapter schema error: {e}")
        raise HTTPException(status_code=404, detail=f"Adapter or schema not found: {adapter_key}")

# ── Benchmark Data Endpoints ─────────────────────────────────
@app.get("/api/data/benchmarks", response_model=List[Dict[str, str]], tags=["data"])
async def get_benchmarks():
    """List available benchmark datasets"""
    return [
        {"id": "esol", "name": "ESOL", "description": "Water solubility dataset (1,128 compounds)", "target": "measured log solubility"},
        {"id": "freesolv", "name": "FreeSolv", "description": "Hydration free energy dataset (642 compounds)", "target": "expt"},
        {"id": "lipophilicity", "name": "Lipophilicity", "description": "Octanol/water distribution coefficient (4,200 compounds)", "target": "exp"},
    ]

@app.post("/api/data/benchmarks/load", response_model=UploadResponse, tags=["data"])
async def load_benchmark_data(
    dataset_id: str = Query(...),
    session_id: Optional[str] = Query(None),
    backend: SessionBackend = Depends(get_session_backend)
):
    """Load a benchmark dataset into the current session"""
    if session_id and not backend.exists(session_id):
        raise HTTPException(status_code=404, detail="Invalid session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        backend.set(session_id, get_default_session_config())
    
    session = backend.get(session_id)
    
    try:
        from backend.data.benchmark_datasets import load_benchmark
        df = load_benchmark(dataset_id)
        target_col, task_type = auto_detect_columns(df)
        filename = f"benchmark_{dataset_id}.csv"
        
        session.update({
            "df": df, "filename": filename, "target_col": target_col, "task_type": task_type,
            "automl_result": None, "pipeline_result": None,
            "last_accessed": datetime.now().isoformat()
        })
        session["metrics"] = {
            "rows": len(df), "cols": len(df.columns),
            "missing_rate": float(df.isna().mean().mean()),
            "numeric_cols": int(df.select_dtypes(include="number").shape[1])
        }
        session["preview"] = serialize_preview(df)
        backend.set(session_id, session)
        
        return UploadResponse(
            success=True, filename=filename, rows=len(df), cols=len(df.columns),
            target_col=target_col, task_type=task_type,
            metrics=session["metrics"], preview=session["preview"], columns=list(df.columns)
        )
    except Exception as e:
        logger.error(f"Benchmark load error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load benchmark: {str(e)}")

# ── Mixture Feature Calculation ─────────────────────────────────
@app.post("/api/mixture/calculate", response_model=Dict[str, Any], tags=["mixture"])
async def calculate_mixture_features(
    components: List[Dict[str, Any]] = Body(...),
):
    """Calculate weighted average descriptors for a mixture of compounds"""
    return {
        "success": True,
        "features": {},
        "message": "Mixture feature calculation is under development"
    }

# ── Health Check ─────────────────────────────────
@app.get("/health", response_model=Dict[str, Any], tags=["health"])
async def health_check(backend: SessionBackend = Depends(get_session_backend)):
    """Health check endpoint"""
    active_sessions = len(getattr(backend, "_store", {})) if isinstance(backend, InMemorySessionBackend) else 0
    return {
        "status": "healthy",
        "version": "2.0.0",
        "environment": settings.env,
        "sessions_active": active_sessions,
        "timestamp": datetime.now().isoformat()
    }

# ── Application Entry Point ─────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.env != "production",
        log_level=settings.log_level.lower(),
        access_log=True,
        workers=1 if settings.env == "development" else 4
    )
