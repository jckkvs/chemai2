"""
backend_fastapi/routers/eda.py
探索的データ分析（EDA）APIエンドポイント
既存 backend ロジックをラップし、Next.js から非同期呼び出し可能にする
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

class EDAMetricsRequest(BaseModel):
    target_col: str
    exclude_cols: List[str] = Field(default_factory=list)
    numeric_only: bool = False

class CorrelationRequest(BaseModel):
    method: str = Field("pearson", pattern="^(pearson|spearman|kendall)$")
    min_abs_corr: float = Field(0.3, ge=0, le=1)
    top_k: int = Field(20, ge=1, le=100)

@router.post("/metrics")
async def get_eda_metrics(
    file: UploadFile = File(...),
    request_json: str = Form(...)
):
    """基本統計量・欠損値・型情報を計算"""
    import json
    try:
        req = EDAMetricsRequest(**json.loads(request_json))
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents), float_precision='high')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise ValueError("Unsupported file format")
            
        if req.exclude_cols:
            df = df.drop(columns=[c for c in req.exclude_cols if c in df.columns], errors='ignore')
        if req.numeric_only:
            df = df.select_dtypes(include=[np.number])
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target = df[req.target_col].dropna() if req.target_col in df.columns else None
        
        stats = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": {c: str(d) for c, d in df.dtypes.items()},
            "missing": df.isna().sum().to_dict(),
            "missing_rate": (df.isna().mean() * 100).round(2).to_dict(),
            "unique": df.nunique().to_dict(),
        }
        
        if target is not None and len(target) > 0:
            stats["target_distribution"] = {
                "mean": float(target.mean()), "std": float(target.std()),
                "min": float(target.min()), "max": float(target.max()),
                "median": float(target.median()), "q25": float(target.quantile(0.25)),
                "q75": float(target.quantile(0.75)),
                "skewness": float(target.skew()) if len(target) > 2 else 0,
                "kurtosis": float(target.kurtosis()) if len(target) > 4 else 0,
            }
            
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": stats,
            "numeric_columns": numeric_cols,
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        }
    except Exception as e:
        logger.error(f"EDA metrics failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/correlation")
async def get_correlation(
    file: UploadFile = File(...),
    request_json: str = Form(...),
    target_col: Optional[str] = Query(None)
):
    """相関行列計算・上位相関ペア抽出"""
    import json
    try:
        req = CorrelationRequest(**json.loads(request_json))
        contents = await file.read()
        
        df = pd.read_csv(io.BytesIO(contents), float_precision='high') if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))
        df_num = df.select_dtypes(include=[np.number])
        
        if df_num.shape[1] < 2:
            return {"warning": "Insufficient numeric columns", "top_pairs": [], "target_correlation": None}
            
        corr = df_num.corr(method=req.method).round(4)
        pairs = []
        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                v = corr.loc[c1, c2]
                if not np.isnan(v) and abs(v) >= req.min_abs_corr:
                    pairs.append({"feature1": c1, "feature2": c2, "correlation": float(v), "abs_corr": float(abs(v))})
        pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
        
        target_corr = None
        if target_col and target_col in corr.columns:
            target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False).head(10).to_dict()
            
        return {
            "method": req.method,
            "matrix_columns": cols,
            "matrix_values": corr.values.tolist(),
            "top_pairs": pairs[:req.top_k],
            "target_correlation": target_corr
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outliers")
async def detect_outliers(
    file: UploadFile = File(...),
    method: str = Query("iqr", pattern="^(iqr|zscore)$"),
    threshold: float = Query(1.5, ge=0.5, le=3.0)
):
    """外れ値検出（IQR/Z-score）"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), float_precision='high') if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))
        df_num = df.select_dtypes(include=[np.number]).dropna()
        results = {}
        
        if method == "iqr":
            for col in df_num.columns:
                q1, q3 = df_num[col].quantile(0.25), df_num[col].quantile(0.75)
                iqr = q3 - q1
                mask = (df_num[col] < q1 - threshold*iqr) | (df_num[col] > q3 + threshold*iqr)
                results[col] = {
                    "method": "IQR", "threshold": threshold,
                    "bounds": {"lower": float(q1 - threshold*iqr), "upper": float(q3 + threshold*iqr)},
                    "outlier_count": int(mask.sum()),
                    "outlier_rate": float(mask.mean() * 100)
                }
        elif method == "zscore":
            from scipy import stats
            for col in df_num.columns:
                z = np.abs(stats.zscore(df_num[col]))
                mask = z > threshold
                results[col] = {
                    "method": "Z-score", "threshold": threshold,
                    "outlier_count": int(mask.sum()),
                    "outlier_rate": float(mask.mean() * 100)
                }
                
        return {"method": method, "threshold": threshold, "results": results}
    except ImportError:
        raise HTTPException(status_code=503, detail="Missing optional dependency: scipy")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
