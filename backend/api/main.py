"""
backend/api/main.py
ChemAI Nexus FastAPI Backend - Production Ready Modular Implementation
"""
from __future__ import annotations
import io, uuid, logging, os, json, inspect, importlib, pkgutil, re
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
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError, create_model

# ── ChemAI Modular System ─────────────────────────────────
from backend.core.config import settings
from backend.utils.logger import logger
from backend.data_manager import DataManager, DatasetMeta
from backend.feature_engine import FeaturePluginRegistry, FeatureConstraint
from backend.ml_pipeline import (
    build_pipeline, PreprocessingConfig, FeatureSelectionConfig, 
    generate_estimator_ui_metadata, ConstraintAwareEstimator
)

# Initialize Global Managers
data_manager = DataManager()
feature_registry = FeaturePluginRegistry()

# ── Dependency Injection ─────────────────────────────────
def get_data_manager(): return data_manager
def get_feature_registry(): return feature_registry

# ── Pydantic Models for API ─────────────────────────────────
class PipelineRunRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: Literal['regression', 'classification']
    preprocessing: Dict[str, PreprocessingConfig] = Field(default_factory=dict)
    feature_selection: Optional[FeatureSelectionConfig] = None
    estimator_name: str
    estimator_params: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[Dict[str, Any]] = Field(default_factory=list)

class AnalysisResult(BaseModel):
    status: Literal["pending", "running", "completed", "failed"]
    best_model: Optional[str] = None
    score: Optional[float] = None
    cv_scores: Optional[List[float]] = None
    feature_importances: Optional[List[Dict[str, Any]]] = None
    message: str
    metadata: Optional[Dict[str, Any]] = None

# ── Session Management (Legacy Support) ─────────────────────────────────
class SessionBackend:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(session_id)
    def set(self, session_id: str, data: Dict[str, Any]):
        self._store[session_id] = data
    def exists(self, session_id: str) -> bool:
        return session_id in self._store

_session_backend = SessionBackend()
def get_session_backend(): return _session_backend

# ── FastAPI Application ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ChemAI Nexus API (Modular)")
    yield
    logger.info("Shutting down ChemAI Nexus API")

app = FastAPI(title="ChemAI Nexus API", version="2.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Endpoints ─────────────────────────────────

@app.post("/api/data/upload", response_model=DatasetMeta)
async def upload_dataset(
    file: UploadFile = File(...),
    user_id: str = "default_user",
    dm: DataManager = Depends(get_data_manager)
):
    """Upload dataset and return metadata using DataManager"""
    return await dm.upload_dataset(file, user_id)

@app.get("/api/data/{dataset_id}/meta")
async def get_dataset_meta(dataset_id: str, dm: DataManager = Depends(get_data_manager)):
    return dm.get_metadata(dataset_id)

@app.get("/api/params/feature-engines")
async def list_feature_engines(reg: FeaturePluginRegistry = Depends(get_feature_registry)):
    return [
        {
            "name": p.name,
            "category": p.category,
            "description": p.description,
            "compute_cost": p.compute_cost,
            "params": p.default_params
        } for p in reg.list_plugins()
    ]

@app.get("/api/params/metadata/{estimator_name}")
async def get_estimator_metadata(estimator_name: str):
    """Dynamically discover estimator parameters and return UI metadata"""
    # Mapping names to classes
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.linear_model import Ridge, Lasso, LogisticRegression
    import xgboost as xgb
    
    mapping = {
        "RandomForestRegressor": RandomForestRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "XGBRegressor": xgb.XGBRegressor,
        "XGBClassifier": xgb.XGBClassifier,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "LogisticRegression": LogisticRegression
    }
    
    cls = mapping.get(estimator_name)
    if not cls:
        raise HTTPException(status_code=404, detail="Estimator not found")
    
    return generate_estimator_ui_metadata(cls)

@app.post("/api/pipeline/run", response_model=AnalysisResult)
async def run_pipeline(
    req: PipelineRunRequest,
    dm: DataManager = Depends(get_data_manager)
):
    """Execute 5-stage pipeline with modular architecture"""
    try:
        # 1. Load Data
        df = dm.get_dataset(req.dataset_id)
        X = df.drop(columns=[req.target_column])
        y = df[req.target_column]
        
        # 2. Instantiate Estimator
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        if req.estimator_name == "RandomForestRegressor":
            est_cls = RandomForestRegressor
        elif req.estimator_name == "XGBRegressor":
            est_cls = xgb.XGBRegressor
        else:
            est_cls = RandomForestRegressor
            
        estimator = est_cls(**req.estimator_params)
        
        # 3. Build Pipeline
        # Convert constraint dicts to FeatureConstraint objects
        constraints = {
            c['feature_name']: FeatureConstraint(**c) for c in req.constraints
        }
        
        pipeline = build_pipeline(
            column_config=req.preprocessing,
            feature_selection=req.feature_selection,
            estimator=estimator,
            constraints=constraints,
            task_type=req.task_type
        )
        
        # 4. Fit and Evaluate
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        pipeline.fit(X, y)
        
        # 5. Extract results
        final_est = pipeline.named_steps['estimator']
        importances = None
        if hasattr(final_est, 'feature_importances_'):
            # Note: This is simplified, real implementation needs feature name alignment
            importances = [{"name": f"Feature {i}", "value": float(v)} for i, v in enumerate(final_est.feature_importances_)]
            importances = sorted(importances, key=lambda x: x['value'], reverse=True)[:10]
            
        return AnalysisResult(
            status="completed",
            best_model=req.estimator_name,
            score=float(cv_scores.mean()),
            cv_scores=cv_scores.tolist(),
            feature_importances=importances,
            message="Success",
            metadata={"n_features": X.shape[1]}
        )
        
    except Exception as e:
        logger.error(f"Pipeline failure: {e}", exc_info=True)
        return AnalysisResult(status="failed", message=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
