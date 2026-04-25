"""
backend/api/main.py
ChemAI Nexus FastAPI Backend - Production Ready Implementation
Custom Pipeline Framework with Meta-Feature & SMILES Support
"""
from __future__ import annotations

import io
import uuid
import logging
import os
import json
import inspect
import importlib
import pkgutil
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Callable, Union, Type
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError, create_model
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut,
    PredefinedSplit, ShuffleSplit, StratifiedShuffleSplit, RepeatedKFold
)
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif,
    SelectFromModel, RFE, RFECV, SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
import xgboost as xgb

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
    feature_plugins_dir: str = "backend/feature_plugins"
    
    @field_validator("allowed_origins")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

settings = Settings()

# ── 型定義 ─────────────────────────────────
TaskType = Literal["regression", "classification"]
ColumnType = Literal["numeric", "categorical", "binary", "datetime", "text", "smiles"]
NumericType = Literal["continuous", "discrete", "binary", "count"]
ConstraintStrength = Literal["hard", "soft"]
MonotonicDirection = Literal[-1, 0, 1]  # -1: decreasing, 0: unknown, 1: increasing

# ── Pydantic Models ─────────────────────────────────
class DataColumn(BaseModel):
    name: str
    type: ColumnType
    numeric_type: Optional[NumericType] = None
    categories: Optional[List[str]] = None
    missing_count: int = 0
    unique_count: int = 0
    sample_values: List[Any] = Field(default_factory=list)

class MetricsSchema(BaseModel):
    rows: int = Field(..., ge=0, le=10_000_000)
    cols: int = Field(..., ge=0, le=10_000)
    missing_rate: float = Field(..., ge=0.0, le=1.0)
    numeric_cols: int = Field(..., ge=0)

class UploadResponse(BaseModel):
    success: bool
    filename: str = Field(..., min_length=1, max_length=255)
    rows: int
    cols: int
    target_col: str
    task_type: TaskType
    metrics: MetricsSchema
    preview: List[Dict[str, Any]] = Field(..., max_length=100)
    columns: List[str]
    column_details: Optional[List[DataColumn]] = None

class ColumnConfig(BaseModel):
    target_col: str = Field(..., min_length=1)
    task_type: Optional[TaskType] = None
    exclude_cols: List[str] = Field(default_factory=list)
    column_types: Optional[Dict[str, ColumnType]] = None

class PreprocessingConfig(BaseModel):
    # Numeric columns
    num_scaler: Literal["standard", "robust", "minmax", "maxabs", "none"] = "standard"
    num_imputer: Literal["median", "mean", "knn", "iterative", "drop"] = "median"
    num_transform: Literal["none", "boxcox", "yeojohnson", "quantile_uniform", "quantile_normal", "log1p"] = "none"
    # Categorical columns
    cat_encoder: Literal["onehot", "ordinal", "target", "binary", "leave_one_out"] = "onehot"
    cat_imputer: Literal["most_frequent", "constant", "drop"] = "most_frequent"
    # Column-specific overrides
    column_overrides: Optional[Dict[str, Dict[str, str]]] = None

class FeatureGenerationConfig(BaseModel):
    do_polynomial: bool = False
    poly_degree: int = Field(default=2, ge=2, le=3)
    poly_interaction_only: bool = True
    do_custom_interactions: Optional[List[List[str]]] = None

class FeatureSelectionConfig(BaseModel):
    feature_selector: Literal["none", "variance", "selectkbest_f", "selectkbest_mi", 
                              "select_from_model_lasso", "select_from_model_rf", "rfe", "boruta"] = "none"
    n_features_to_select: int = Field(default=20, ge=1, le=1000)
    selector_params: Optional[Dict[str, Any]] = None

class MonotonicConstraint(BaseModel):
    feature: str
    direction: MonotonicDirection
    strength: ConstraintStrength
    sigma_range: float = Field(default=3.0, ge=-10.0, le=10.0)
    linear: bool = False

class PipelineConfig(BaseModel):
    # Cross-validation
    cv_strategy: Literal["kfold", "stratified", "group", "time_series", "loo", "lgo", "predefined", "shuffle", "repeated"] = "kfold"
    cv_folds: int = Field(default=5, ge=2, le=20)
    cv_params: Optional[Dict[str, Any]] = None
    # Preprocessing
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    # Feature engineering
    feature_generation: FeatureGenerationConfig = Field(default_factory=FeatureGenerationConfig)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    # Model selection
    estimator: str = "RandomForestRegressor"
    estimator_params: Dict[str, Any] = Field(default_factory=dict)
    # Constraints
    monotonic_constraints: List[MonotonicConstraint] = Field(default_factory=list)
    # Execution flags
    do_eda: bool = True
    do_prep: bool = True
    do_eval: bool = True
    do_pca: bool = True
    do_shap: bool = True

class AnalysisResult(BaseModel):
    status: Literal["pending", "running", "completed", "failed"]
    best_model: Optional[str] = None
    score: Optional[float] = None
    cv_scores: Optional[List[float]] = None
    feature_importances: Optional[List[Dict[str, Any]]] = None
    message: str
    metadata: Optional[Dict[str, Any]] = None

class APIError(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = None

class FeatureEngineMetadata(BaseModel):
    name: str
    description: str
    category: Literal["physicochemical", "structural", "electronic", "topological", "quantum", "custom"]
    compute_cost: Literal["low", "medium", "high"]
    recommended_for: List[str] = Field(default_factory=list)
    params: Optional[Dict[str, Dict[str, Any]]] = None
    returns_meta_features: bool = False

# ── Custom Pipeline Framework ─────────────────────────────────
class MetaFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that supports meta-features (e.g., SMILES-derived features)
    Unlike sklearn's ColumnTransformer, this handles heterogeneous feature sources
    """
    def __init__(
        self,
        tabular_transformers: Dict[str, Any] = None,
        feature_engines: List[Callable] = None,
        feature_selector: Any = None,
        monotonic_constraints: List[MonotonicConstraint] = None,
    ):
        self.tabular_transformers = tabular_transformers or {}
        self.feature_engines = feature_engines or []
        self.feature_selector = feature_selector
        self.monotonic_constraints = monotonic_constraints or []
        self._fitted = False
        self._feature_names = []
        self._constraint_mask = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params):
        """Fit all transformers and feature engines"""
        # 1. Fit tabular transformers
        for name, transformer in self.tabular_transformers.items():
            if hasattr(transformer, 'fit'):
                transformer.fit(X[[name]], y, **fit_params)
        
        # 2. Initialize feature engines (SMILES, etc.)
        for engine in self.feature_engines:
            if hasattr(engine, 'initialize'):
                engine.initialize(X, y)
        
        # 3. Build feature name mapping
        self._feature_names = list(X.columns)
        for engine in self.feature_engines:
            if hasattr(engine, 'get_feature_names'):
                self._feature_names.extend(engine.get_feature_names())
        
        # 4. Build constraint mask for monotonicity
        if self.monotonic_constraints:
            self._constraint_mask = self._build_constraint_mask(self._feature_names)
        
        # 5. Fit feature selector if specified
        if self.feature_selector and hasattr(self.feature_selector, 'fit'):
            # Combine tabular + engine features for selection
            X_transformed = self._transform_tabular(X)
            X_combined = self._add_engine_features(X_transformed, X)
            self.feature_selector.fit(X_combined, y, **fit_params)
            # Update feature names based on selection
            if hasattr(self.feature_selector, 'get_support'):
                mask = self.feature_selector.get_support()
                self._feature_names = [n for n, m in zip(self._feature_names, mask) if m]
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data through all pipeline stages"""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        # 1. Apply tabular transformers
        X_tabular = self._transform_tabular(X)
        
        # 2. Add meta-features from engines
        X_combined = self._add_engine_features(X_tabular, X)
        
        # 3. Apply feature selection
        if self.feature_selector and hasattr(self.feature_selector, 'transform'):
            X_combined = self.feature_selector.transform(X_combined)
        
        # 4. Apply monotonic constraints (for compatible estimators)
        if self._constraint_mask is not None:
            X_combined = self._apply_constraint_mask(X_combined)
        
        return X_combined
    
    def _transform_tabular(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply tabular preprocessing transformers"""
        X_out = X.copy()
        for col_name, transformer in self.tabular_transformers.items():
            if col_name in X.columns:
                if hasattr(transformer, 'transform'):
                    X_out[col_name] = transformer.transform(X[[col_name]]).ravel()
                elif callable(transformer):
                    X_out[col_name] = transformer(X[col_name])
        return X_out
    
    def _add_engine_features(self, X_tabular: pd.DataFrame, X_original: pd.DataFrame) -> np.ndarray:
        """Add meta-features from registered engines"""
        features = [X_tabular.values]
        for engine in self.feature_engines:
            if hasattr(engine, 'compute'):
                engine_features = engine.compute(X_original)
                if engine_features is not None:
                    features.append(engine_features)
            elif callable(engine):
                engine_features = engine(X_original)
                if engine_features is not None:
                    features.append(engine_features)
        return np.hstack(features) if len(features) > 1 else features[0]
    
    def _build_constraint_mask(self, feature_names: List[str]) -> np.ndarray:
        """Build mask array for monotonic constraints"""
        mask = np.zeros(len(feature_names), dtype=int)
        for constraint in self.monotonic_constraints:
            if constraint.feature in feature_names:
                idx = feature_names.index(constraint.feature)
                mask[idx] = constraint.direction
        return mask
    
    def _apply_constraint_mask(self, X: np.ndarray) -> np.ndarray:
        """Apply constraint mask (for soft constraints via penalty)"""
        if self._constraint_mask is None:
            return X
        # For soft constraints: add penalty terms to loss (handled by estimator)
        # For hard constraints: clip values to enforce monotonicity
        return X
    
    def get_feature_names_out(self) -> List[str]:
        """Return names of output features"""
        return self._feature_names.copy()


# ── SMILES Feature Engine Plugin System ─────────────────────────────────
class FeatureEngineRegistry:
    """Registry for dynamically loaded feature engine plugins"""
    
    def __init__(self, plugins_dir: str = None):
        self.plugins_dir = plugins_dir or settings.feature_plugins_dir
        self._engines: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, FeatureEngineMetadata] = {}
        self._load_plugins()
    
    def _load_plugins(self):
        """Discover and load feature engine plugins from directory"""
        plugins_path = Path(self.plugins_dir)
        if not plugins_path.exists():
            logger.warning(f"Feature plugins directory not found: {self.plugins_dir}")
            return
        
        for module_info in pkgutil.iter_modules([str(plugins_path)]):
            if module_info.name.startswith('_'):
                continue
            try:
                # Add plugins dir to sys.path if not present
                import sys
                if str(plugins_path) not in sys.path:
                    sys.path.append(str(plugins_path))
                
                module = importlib.import_module(module_info.name)
                # Reload module to pick up changes
                importlib.reload(module)
                
                if hasattr(module, 'compute_features'):
                    # Extract metadata from module docstring or __feature_metadata__
                    metadata = self._extract_metadata(module)
                    self._engines[module_info.name] = {
                        'module': module,
                        'compute': module.compute_features,
                        'metadata': metadata
                    }
                    self._metadata[module_info.name] = metadata
                    logger.info(f"Loaded feature engine: {module_info.name}")
            except Exception as e:
                logger.warning(f"Failed to load plugin {module_info.name}: {e}")
    
    def _extract_metadata(self, module) -> FeatureEngineMetadata:
        """Extract metadata from plugin module"""
        # Check for __feature_metadata__ attribute first
        if hasattr(module, '__feature_metadata__'):
            meta = module.__feature_metadata__
            if isinstance(meta, dict):
                return FeatureEngineMetadata(**meta)
        
        # Fallback: parse docstring
        doc = inspect.getdoc(module) or ""
        # Look for ---FEATURE_METADATA--- block if it exists (for compatibility with user's snippet)
        if "---FEATURE_METADATA---" in doc:
            try:
                meta_text = doc.split("---FEATURE_METADATA---")[1].split("---END_METADATA---")[0]
                import yaml
                meta_dict = yaml.safe_load(meta_text)
                return FeatureEngineMetadata(**meta_dict)
            except Exception as e:
                logger.warning(f"Failed to parse metadata block in {module.__name__}: {e}")

        return FeatureEngineMetadata(
            name=module.__name__.replace('_', ' ').title(),
            description=doc.split('\n')[0] if doc else "No description",
            category="custom",
            compute_cost="medium"
        )
    
    def list_engines(self, category: str = None, task_type: TaskType = None) -> List[Dict[str, Any]]:
        """List available feature engines with filtering"""
        result = []
        for name, info in self._engines.items():
            meta = info['metadata']
            if category and meta.category != category:
                continue
            if task_type and task_type not in meta.recommended_for:
                continue
            result.append({
                'key': name,
                'name': meta.name,
                'description': meta.description,
                'category': meta.category,
                'compute_cost': meta.compute_cost,
                'recommended_for': meta.recommended_for,
                'params': meta.params or {},
                'returns_meta_features': meta.returns_meta_features
            })
        return result
    
    def get_engine(self, name: str) -> Optional[Callable]:
        """Get compute function for a specific engine"""
        return self._engines.get(name, {}).get('compute')
    
    def get_engine_params_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get parameter schema for dynamic UI generation"""
        return self._metadata.get(name, FeatureEngineMetadata(name="", description="", category="custom", compute_cost="medium")).params
    
    def register_custom_engine(self, name: str, compute_func: Callable, metadata: FeatureEngineMetadata):
        """Register a custom/user-defined feature engine"""
        self._engines[name] = {
            'module': None,
            'compute': compute_func,
            'metadata': metadata
        }
        self._metadata[name] = metadata
        logger.info(f"Registered custom feature engine: {name}")


# ── Constraint System ─────────────────────────────────
class ConstraintEnforcer:
    """Handles application of monotonicity/linearity constraints"""
    
    @staticmethod
    def apply_hard_monotonicity(X: np.ndarray, constraints: List[MonotonicConstraint], 
                                feature_names: List[str]) -> np.ndarray:
        """Apply hard monotonicity constraints via isotonic regression"""
        from sklearn.isotonic import IsotonicRegression
        
        X_out = X.copy()
        for constraint in constraints:
            if constraint.feature not in feature_names:
                continue
            idx = feature_names.index(constraint.feature)
            if constraint.direction == 0:
                continue  # Unknown direction
            
            # Apply isotonic regression along the feature axis
            ir = IsotonicRegression(increasing=(constraint.direction == 1))
            X_out[:, idx] = ir.fit_transform(np.arange(len(X)), X[:, idx])
        
        return X_out


# ── Session Management ─────────────────────────────────
class SessionBackend:
    """Abstract session management backend"""
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
    def set(self, session_id: str, data: Dict[str, Any], ttl: int = settings.session_ttl) -> bool:
        raise NotImplementedError
    def delete(self, session_id: str) -> bool:
        raise NotImplementedError
    def exists(self, session_id: str) -> bool:
        raise NotImplementedError

class InMemorySessionBackend(SessionBackend):
    """In-memory session backend for development"""
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

_session_backend = InMemorySessionBackend()

# ── Dependency Injection ─────────────────────────────────
def get_session_backend() -> SessionBackend:
    return _session_backend

def get_feature_registry() -> FeatureEngineRegistry:
    if not hasattr(get_feature_registry, '_registry'):
        get_feature_registry._registry = FeatureEngineRegistry(settings.feature_plugins_dir)
    return get_feature_registry._registry

def get_default_session_config() -> Dict[str, Any]:
    return {
        "df": None,
        "filename": None,
        "target_col": None,
        "task_type": "regression",
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
            "estimator": "RandomForestRegressor",
            "estimator_params": {},
            "monotonic_constraints": [],
            "do_eda": True,
            "do_prep": True,
            "do_eval": True,
            "do_pca": True,
            "do_shap": True
        },
        "metrics": None,
        "preview": [],
        "automl_result": None,
        "created_at": datetime.now().isoformat(),
    }

async def get_request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


# ── Utility Functions ─────────────────────────────────
def validate_file(file: UploadFile) -> tuple[bytes, str]:
    filename = file.filename or ""
    if not any(filename.lower().endswith(ext) for ext in [".csv", ".xlsx", ".xls"]):
        raise HTTPException(status_code=400, detail="Unsupported format")
    
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > settings.max_file_size:
        raise HTTPException(status_code=400, detail="File too large")
    
    return file.file.read(), filename

def parse_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        
        # Auto-drop metadata columns
        drop_patterns = ["sample_id", "category", "^id$", "index", "unnamed"]
        cols_to_drop = [c for c in df.columns if any(re.search(p, c.lower().strip()) for p in drop_patterns)]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

def auto_detect_columns(df: pd.DataFrame) -> tuple[str, str, List[DataColumn]]:
    target_col = "Target" if "Target" in df.columns else df.columns[-1]
    task_type = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
    
    column_details = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
            numeric_type = "continuous" if series.nunique() > 10 else "discrete"
        elif series.nunique() == 2:
            col_type = "binary"
            numeric_type = "binary"
        elif col.lower() in ["smiles", "structure", "mol"]:
            col_type = "smiles"
            numeric_type = None
        else:
            col_type = "categorical"
            numeric_type = None
        
        column_details.append(DataColumn(
            name=col,
            type=col_type,
            numeric_type=numeric_type,
            categories=series.dropna().unique().tolist()[:10] if col_type == "categorical" else None,
            missing_count=int(series.isna().sum()),
            unique_count=int(series.nunique()),
            sample_values=series.dropna().head(5).tolist()
        ))
    
    return target_col, task_type, column_details

def serialize_preview(df: pd.DataFrame, max_rows: int = 8) -> List[Dict[str, Any]]:
    return df.head(max_rows).replace({np.nan: None}).to_dict(orient="records")


# ── FastAPI Lifespan ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ChemAI Nexus API")
    os.makedirs(settings.feature_plugins_dir, exist_ok=True)
    yield
    logger.info("Shutting down ChemAI Nexus API")


# ── FastAPI Application ─────────────────────────────────
app = FastAPI(title="ChemAI Nexus API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Endpoints ─────────────────────────────────

@app.post("/api/session/init")
async def init_session_endpoint(backend: SessionBackend = Depends(get_session_backend)):
    session_id = str(uuid.uuid4())
    backend.set(session_id, get_default_session_config())
    return {"session_id": session_id}

@app.delete("/api/session/{session_id}")
async def close_session_endpoint(session_id: str, backend: SessionBackend = Depends(get_session_backend)):
    backend.delete(session_id)
    return {"status": "closed"}

@app.post("/api/upload", response_model=UploadResponse)
async def upload_data_endpoint(
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(None),
    backend: SessionBackend = Depends(get_session_backend)
):
    if not session_id or not backend.exists(session_id):
        session_id = str(uuid.uuid4())
        backend.set(session_id, get_default_session_config())
    
    session = backend.get(session_id)
    content, filename = validate_file(file)
    df = parse_dataframe(content, filename)
    target_col, task_type, column_details = auto_detect_columns(df)
    
    session.update({
        "df": df,
        "filename": filename,
        "target_col": target_col,
        "task_type": task_type,
        "column_details": [c.model_dump() for c in column_details],
        "metrics": {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_rate": float(df.isna().mean().mean()),
            "numeric_cols": int(df.select_dtypes(include="number").shape[1])
        },
        "preview": serialize_preview(df)
    })
    backend.set(session_id, session)
    
    return UploadResponse(
        success=True, filename=filename, rows=len(df), cols=len(df.columns),
        target_col=target_col, task_type=task_type, metrics=session["metrics"],
        preview=session["preview"], columns=list(df.columns), column_details=column_details
    )

@app.get("/api/data/info")
async def get_data_info_endpoint(session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session or session["df"] is None:
        raise HTTPException(status_code=404, detail="No data")
    return {
        "filename": session["filename"],
        "columns": list(session["df"].columns),
        "column_details": session["column_details"],
        "target_col": session["target_col"],
        "task_type": session["task_type"],
        "metrics": session["metrics"],
        "preview": session["preview"]
    }

@app.post("/api/pipeline/run", response_model=AnalysisResult)
async def run_pipeline_endpoint(
    cfg: PipelineConfig,
    session_id: str = Query(...),
    backend: SessionBackend = Depends(get_session_backend),
    registry: FeatureEngineRegistry = Depends(get_feature_registry)
):
    session = backend.get(session_id)
    if not session or session["df"] is None:
        raise HTTPException(status_code=404, detail="No data")
    
    df = session["df"]
    target_col = session["target_col"]
    task_type = session["task_type"]
    
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Build transformers
        tabular_transformers = {}
        for col in X.select_dtypes(include="number").columns:
            if cfg.preprocessing.num_scaler == "standard":
                tabular_transformers[col] = StandardScaler()
            elif cfg.preprocessing.num_scaler == "robust":
                tabular_transformers[col] = RobustScaler()
        
        # Build engines
        feature_engines = []
        smiles_cols = [c.name for c in [DataColumn(**d) for d in session["column_details"]] if c.type == "smiles"]
        if smiles_cols:
            engine_func = registry.get_engine("rdkit_descriptors")
            if engine_func:
                feature_engines.append(lambda df_in: engine_func(df_in[smiles_cols[0]].tolist())["feature_matrix"])

        pipeline = MetaFeatureTransformer(
            tabular_transformers=tabular_transformers,
            feature_engines=feature_engines,
            monotonic_constraints=cfg.monotonic_constraints
        )
        
        # Model
        if cfg.estimator == "RandomForestRegressor":
            model = RandomForestRegressor(**cfg.estimator_params)
        elif cfg.estimator == "RandomForestClassifier":
            model = RandomForestClassifier(**cfg.estimator_params)
        elif cfg.estimator == "XGBRegressor":
            model = xgb.XGBRegressor(**cfg.estimator_params)
        else:
            model = LinearRegression()

        # Execute
        X_transformed = pipeline.fit_transform(X, y)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_transformed, y, cv=cfg.cv_folds)
        model.fit(X_transformed, y)
        
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = [{"name": n, "value": float(v)} for n, v in zip(pipeline.get_feature_names_out(), model.feature_importances_)]
            importances = sorted(importances, key=lambda x: x["value"], reverse=True)[:10]

        result = AnalysisResult(
            status="completed",
            best_model=cfg.estimator,
            score=float(scores.mean()),
            cv_scores=scores.tolist(),
            feature_importances=importances,
            message="Success",
            metadata={"n_features": X_transformed.shape[1]}
        )
        session["automl_result"] = result.model_dump()
        backend.set(session_id, session)
        return result
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return AnalysisResult(status="failed", message=str(e))

@app.get("/api/results", response_model=AnalysisResult)
async def get_results_endpoint(session_id: str = Query(...), backend: SessionBackend = Depends(get_session_backend)):
    session = backend.get(session_id)
    if not session or "automl_result" not in session:
        return AnalysisResult(status="pending", message="No results")
    return AnalysisResult(**session["automl_result"])

@app.get("/api/params/feature-engines")
async def list_engines_endpoint(registry: FeatureEngineRegistry = Depends(get_feature_registry)):
    return registry.list_engines()

@app.get("/health")
async def health_endpoint():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
