"""
データ関連APIエンドポイント
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/data", tags=["data"])

@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """データをアップロード"""
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # データ統計情報を返す
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(5).to_dict(orient="records"),
            "missing_values": df.isna().sum().to_dict()
        }
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/columns/{target_col}/detect")
async def detect_column_types(target_col: str):
    """列の型を自動検出"""
    # TODO: 実際のデータに基づいて検出
    return {
        "target_col": target_col,
        "task_type": "regression",  # or "classification"
        "smiles_col": None,
        "numeric_cols": [],
        "categorical_cols": []
    }
