"""
backend/api/data.py
Benchmark data loading and dataset management API
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from backend.data.benchmark_datasets import list_benchmark_datasets, load_benchmark
import pandas as pd
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/data", tags=["data"])

# We need access to SESSIONS from main.py, but for circular import reasons, 
# we'll use a getter/setter approach or just reference it if passed.
# In a real app, this would be a shared service or Redis.
from backend.api.main import get_session

@router.get("/benchmarks")
async def get_benchmarks():
    """List available benchmark datasets"""
    return list_benchmark_datasets()

@router.post("/benchmarks/load")
async def load_benchmark_data(session_id: str, dataset_id: str):
    """Load a benchmark dataset into the current session"""
    session = get_session(session_id)
    
    try:
        df = load_benchmark(dataset_id)
        
        # Sync with upload logic in main.py
        target_col = None
        benchmarks = list_benchmark_datasets()
        bench_info = next((b for b in benchmarks if b["id"] == dataset_id), None)
        if bench_info:
            target_col = bench_info["target"]
            
        # Fallback to last column if target not found
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]
            
        smiles_col = None
        for col in df.columns:
            if col.lower() == "smiles":
                smiles_col = col
                break
                
        task_type = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"
        
        # Update session
        session["df"] = df
        session["filename"] = f"benchmark_{dataset_id}.csv"
        session["target_col"] = target_col
        session["task_type"] = task_type
        session["smiles_col"] = smiles_col
        session["automl_result"] = None
        session["pipeline_result"] = None
        
        # Metrics
        numeric_cols = df.select_dtypes(include='number').shape[1]
        missing_rate = float(df.isna().mean().mean())
        session["metrics"] = {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_rate": missing_rate,
            "numeric_cols": numeric_cols
        }
        
        # Preview
        preview = df.head(8).to_dict(orient="records")
        for row in preview:
            for k, v in row.items():
                if pd.isna(v):
                    row[k] = None
                elif isinstance(v, float):
                    row[k] = round(v, 4)
        session["preview"] = preview
        
        return {
            "success": True,
            "filename": session["filename"],
            "rows": len(df),
            "target_col": target_col,
            "task_type": task_type,
            "metrics": session["metrics"],
            "preview": preview,
            "columns": list(df.columns)
        }
        
    except Exception as e:
        logger.error(f"Failed to load benchmark {dataset_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load benchmark: {str(e)}")
