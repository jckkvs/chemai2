"""backend_fastapi/routers/eda.py
探索的データ分析（EDA）API。既存 backend.eda ロジックを非同期ラップ。
"""
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/eda", tags=["eda"])

class EDAMetricsReq(BaseModel):
    target_col: str
    exclude_cols: List[str] = Field(default_factory=list)

class CorrelationReq(BaseModel):
    method: str = Field("pearson", pattern="^(pearson|spearman|kendall)$")
    min_abs: float = Field(0.3, ge=0, le=1)
    top_k: int = Field(20, ge=1, le=100)

def _load_df(file: UploadFile, contents: bytes) -> pd.DataFrame:
    if file.filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(contents), float_precision="high")
    elif file.filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(contents))
    raise ValueError("Unsupported format")

@router.post("/metrics")
async def get_metrics(file: UploadFile = File(...), req_json: str = Form(...)):
    try:
        req = EDAMetricsReq.model_validate_json(req_json)
        contents = await file.read()
        df = _load_df(file, contents)
        if req.exclude_cols:
            df = df.drop(columns=[c for c in req.exclude_cols if c in df.columns], errors="ignore")
            
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_stats = None
        if req.target_col in num_cols and len(df[req.target_col].dropna()) > 0:
            t = df[req.target_col].dropna()
            target_stats = {
                "mean": float(t.mean()), "std": float(t.std()),
                "min": float(t.min()), "max": float(t.max()),
                "median": float(t.median()), "q25": float(t.quantile(0.25)),
                "q75": float(t.quantile(0.75)), "skewness": float(t.skew()),
                "kurtosis": float(t.kurtosis())
            }
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "shape": df.shape,
            "dtypes": {c: str(d) for c, d in df.dtypes.items()},
            "missing_rate": (df.isna().mean() * 100).round(2).to_dict(),
            "numeric_columns": num_cols,
            "target_distribution": target_stats
        }
    except Exception as e:
        logger.error(f"EDA metrics failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/correlation")
async def get_correlation(file: UploadFile = File(...), req_json: str = Form(...), target: Optional[str] = Query(None)):
    try:
        req = CorrelationReq.model_validate_json(req_json)
        contents = await file.read()
        df_num = _load_df(file, contents).select_dtypes(include=[np.number])
        if df_num.shape[1] < 2:
            return {"top_pairs": [], "target_corr": None, "warning": "Insufficient numeric columns"}
            
        corr = df_num.corr(method=req.method).round(4)
        pairs = []
        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                v = corr.loc[c1, c2]
                if not np.isnan(v) and abs(v) >= req.min_abs:
                    pairs.append({"feature1": c1, "feature2": c2, "corr": float(v)})
    except Exception as e:
        logger.error(f"EDA correlation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
    
    target_corr = None
    if target and target in cols:
        target_corr = corr[target].drop(target).abs().sort_values(ascending=False).head(10).to_dict()
    return {"method": req.method, "matrix_cols": cols, "matrix_vals": corr.values.tolist(), "top_pairs": pairs[:req.top_k], "target_corr": target_corr}
