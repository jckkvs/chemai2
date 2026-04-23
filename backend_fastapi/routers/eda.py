"""backend_fastapi/routers/eda.py - 次元削減エンドポイント追加"""
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
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
            "metrics": {
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "target_distribution": target_stats,
                "missing_rate": (df.isna().mean() * 100).round(2).to_dict()
            },
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols
        }
    except Exception as e:
        logger.error(f"EDA metrics failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/correlation")
async def get_correlation(file: UploadFile = File(...), req_json: str = Form(...), target_col: Optional[str] = Query(None)):
    try:
        req = CorrelationReq.model_validate_json(req_json)
        contents = await file.read()
        df_num = _load_df(file, contents).select_dtypes(include=[np.number])
        if df_num.shape[1] < 2:
            return {"top_pairs": [], "target_correlation": None, "warning": "Insufficient numeric columns"}
            
        corr = df_num.corr(method=req.method).round(4)
        pairs = []
        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                v = corr.loc[c1, c2]
                if not np.isnan(v) and abs(v) >= req.min_abs:
                    pairs.append({"feature1": c1, "feature2": c2, "correlation": float(v)})
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        target_corr = None
        if target_col and target_col in cols:
            target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False).head(10).to_dict()
            
        return {
            "method": req.method, 
            "matrix_cols": cols, 
            "top_pairs": pairs[:req.top_k], 
            "target_correlation": target_corr
        }
    except Exception as e:
        logger.error(f"EDA correlation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dim_reduction")
async def calculate_dim_reduction(
    file: UploadFile = File(...),
    method: Literal["pca", "tsne"] = Query("pca"),
    n_components: int = Query(2),
    perplexity: float = Query(30.0),
    target_col: Optional[str] = Query(None)
):
    """次元削減計算（PCA/t-SNE）"""
    try:
        contents = await file.read()
        df = _load_df(file, contents)
        df_num = df.select_dtypes(include=[np.number]).dropna()
        
        if df_num.shape[0] < 5 or df_num.shape[1] < n_components:
            raise ValueError("データが次元削減に必要なサイズを満たしていません")
            
        if method == "pca":
            model = PCA(n_components=n_components, random_state=42)
            embeddings = model.fit_transform(df_num)
            explained = model.explained_variance_ratio_.tolist()
        else:
            # t-SNEは計算コストが高いためサブサンプリング
            max_samples = 5000
            sample_df = df_num.sample(min(len(df_num), max_samples), random_state=42)
            model = TSNE(n_components=n_components, perplexity=min(perplexity, len(sample_df)/3 - 1), random_state=42, n_jobs=-1)
            embeddings = model.fit_transform(sample_df)
            explained = []
            
        result = pd.DataFrame(embeddings, columns=[f"comp_{i}" for i in range(n_components)])
        if target_col and target_col in df.columns:
            # Align target values with result (if sampled)
            target_vals = df.loc[df_num.index, target_col].iloc[:len(result)].tolist()
            result["target"] = target_vals
            
        return {
            "embeddings": result.to_dict(orient="records"),
            "explained_variance": explained,
            "n_samples": len(result),
            "method": method
        }
    except Exception as e:
        logger.error(f"Dimension reduction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outliers")
async def detect_outliers(
    file: UploadFile = File(...),
    method: Literal["iqr", "zscore"] = Form("iqr"),
    threshold: float = Form(1.5)
):
    """外れ値検出（IQR/Z-Score）"""
    try:
        contents = await file.read()
        df = _load_df(file, contents)
        df_num = df.select_dtypes(include=[np.number])
        results = {}
        
        for col in df_num.columns:
            series = df_num[col].dropna()
            if len(series) == 0: continue
            
            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
            else:
                mean = series.mean()
                std = series.std()
                lower = mean - threshold * std
                upper = mean + threshold * std
            
            outliers = series[(series < lower) | (series > upper)]
            results[col] = {
                "outlier_count": len(outliers),
                "outlier_rate": (len(outliers) / len(series)) * 100,
                "bounds": {"lower": float(lower), "upper": float(upper)}
            }
            
        return {"method": method, "threshold": threshold, "results": results}
    except Exception as e:
        logger.error(f"Outlier detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
