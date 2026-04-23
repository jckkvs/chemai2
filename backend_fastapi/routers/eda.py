"""
backend_fastapi/routers/eda.py
探索的データ分析（EDA）関連APIエンドポイント
既存 backend.eda モジュールをラップ
"""
from fastapi import APIRouter, HTTPException, Query, File, Form, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import logging
import base64
import json
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/eda", tags=["eda"])

class EDAMetricsRequest(BaseModel):
    """EDA統計量計算リクエスト"""
    target_col: str
    exclude_cols: List[str] = Field(default_factory=list)
    numeric_only: bool = False

class CorrelationRequest(BaseModel):
    """相関分析リクエスト"""
    method: str = Field("pearson", pattern="^(pearson|spearman|kendall)$")
    min_abs_corr: float = Field(0.3, ge=0, le=1)
    top_k: int = Field(20, ge=1, le=100)

@router.post("/metrics")
async def calculate_eda_metrics(
    file: UploadFile = File(...),
    request_json: str = Form(...)
):
    """基本統計量・欠損値・型情報を計算"""
    try:
        request = json.loads(request_json)
        request_obj = EDAMetricsRequest(**request)
        file_bytes = await file.read()


        # データ読み込み
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes), float_precision='high')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format")
        
        # 除外列の削除
        if request_obj.exclude_cols:
            df = df.drop(columns=[c for c in request_obj.exclude_cols if c in df.columns], errors='ignore')
        
        # 数値列のみフィルタ
        if request_obj.numeric_only:
            df = df.select_dtypes(include=[np.number])
        
        # 基本統計量
        stats = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing": df.isna().sum().to_dict(),
            "missing_rate": (df.isna().mean() * 100).round(2).to_dict(),
            "unique": df.nunique().to_dict(),
        }
        
        # 数値列の詳細統計
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols and request_obj.target_col in numeric_cols:
            target = df[request_obj.target_col].dropna()
            stats["target_distribution"] = {
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
                "median": float(target.median()),
                "q25": float(target.quantile(0.25)),
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
        logger.error(f"EDA metrics calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/correlation")
async def calculate_correlation(
    file: UploadFile = File(...),
    request_json: str = Form(...),
    target_col: Optional[str] = Query(None)
):
    """相関行列計算・上位相関ペア抽出"""
    try:
        request = json.loads(request_json)
        request_obj = CorrelationRequest(**request)
        file_bytes = await file.read()


        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes), float_precision='high')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format")
        
        # 数値列のみ対象
        df_num = df.select_dtypes(include=[np.number])
        if df_num.shape[1] < 2:
            return {"matrix": [], "top_pairs": [], "warning": "Insufficient numeric columns"}
        
        # 相関行列計算
        corr = df_num.corr(method=request_obj.method).round(4)
        
        # 上位相関ペア抽出（絶対値ベース）
        pairs = []
        cols = corr.columns.tolist()
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                val = corr.loc[col1, col2]
                if abs(val) >= request_obj.min_abs_corr and not np.isnan(val):
                    pairs.append({
                        "feature1": col1,
                        "feature2": col2,
                        "correlation": float(val),
                        "abs_corr": float(abs(val))
                    })
        
        pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
        top_pairs = pairs[:request_obj.top_k]
        
        # 目的変数との相関（指定時）
        target_corr = None
        if target_col and target_col in corr.columns:
            target_corr = (
                corr[target_col]
                .drop(target_col)
                .abs()
                .sort_values(ascending=False)
                .head(10)
                .to_dict()
            )
        
        return {
            "method": request_obj.method,
            "matrix_columns": cols,
            "matrix_values": corr.values.tolist(),
            "top_pairs": top_pairs,
            "target_correlation": target_corr,
        }
    except Exception as e:
        logger.error(f"Correlation calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outliers")
async def detect_outliers(
    file: UploadFile = File(...),
    filename: str = Query(...),
    method: str = Query("iqr", pattern="^(iqr|zscore|isolation_forest)$"),
    threshold: float = Query(1.5, ge=0.5, le=3.0)
):
    """外れ値検出（IQR/Z-score/Isolation Forest）"""
    try:
        file_bytes = await file.read()
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes), float_precision='high')
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format")
        
        df_num = df.select_dtypes(include=[np.number]).dropna()
        results = {}
        
        if method == "iqr":
            for col in df_num.columns:
                q1 = df_num[col].quantile(0.25)
                q3 = df_num[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                outliers = df_num[(df_num[col] < lower) | (df_num[col] > upper)]
                results[col] = {
                    "method": "IQR",
                    "threshold": threshold,
                    "bounds": {"lower": float(lower), "upper": float(upper)},
                    "outlier_count": len(outliers),
                    "outlier_rate": float(len(outliers) / len(df_num) * 100),
                }
        elif method == "zscore":
            from scipy import stats
            for col in df_num.columns:
                z = np.abs(stats.zscore(df_num[col].dropna()))
                outlier_mask = z > threshold
                results[col] = {
                    "method": "Z-score",
                    "threshold": threshold,
                    "outlier_count": int(outlier_mask.sum()),
                    "outlier_rate": float(outlier_mask.mean() * 100),
                }
        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            if len(df_num) > 10:
                clf = IsolationForest(contamination=float(1/threshold), random_state=42)
                preds = clf.fit_predict(df_num)
                outlier_count = (preds == -1).sum()
                results["global"] = {
                    "method": "IsolationForest",
                    "contamination": float(1/threshold),
                    "outlier_count": int(outlier_count),
                    "outlier_rate": float(outlier_count / len(df_num) * 100),
                }
        
        return {
            "method": method,
            "threshold": threshold,
            "results": results,
            "total_columns": len(df_num.columns),
        }
    except ImportError as e:
        logger.warning(f"Optional dependency missing for outlier detection: {e}")
        raise HTTPException(status_code=503, detail=f"Missing dependency: {e}")
    except Exception as e:
        logger.error(f"Outlier detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
