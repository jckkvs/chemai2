"""backend_fastapi/routers/chem_descriptors.py
SMILES記述子計算API。既存 backend.chem adapters をスレッドプールで実行。
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import io
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chem", tags=["chem-descriptors"])

class DescReq(BaseModel):
    smiles_col: str
    engines: List[str] = Field(default=["rdkit"])
    options: Dict[str, Any] = Field(default_factory=dict)

executor = ThreadPoolExecutor(max_workers=2)

def _run_adapters(df: pd.DataFrame, col: str, engines: List[str], opts: Dict) -> pd.DataFrame:
    results = []
    for eng in engines:
        try:
            # エンジン名からアダプタークラスを動的にロード
            module_name = f"backend.chem.{eng}_adapter"
            cls_name = f"{eng.capitalize()}Adapter"
            mod = __import__(module_name, fromlist=[cls_name])
            adapter_cls = getattr(mod, cls_name)
            adapter = adapter_cls(**opts.get(eng, {}))
            
            df_eng = adapter.compute(df[col].dropna().tolist())
            if df_eng is not None and not df_eng.empty:
                # カラムの重複を避けるためにプレフィックスを付与
                df_eng.columns = [f"{eng}_{c}" if not c.startswith(f"{eng}_") else c for c in df_eng.columns]
                results.append(df_eng.reset_index(drop=True))
        except Exception as e:
            logger.warning(f"Engine {eng} failed: {e}")
            
    if not results:
        return df[[col]]
        
    return pd.concat([df.reset_index(drop=True), *results], axis=1)

@router.post("/compute")
async def compute_descriptors(file: UploadFile = File(...), req_json: str = Form(...)):
    try:
        req = DescReq.model_validate_json(req_json)
        contents = await file.read()
        
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        if req.smiles_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{req.smiles_col}' not found")
            
        loop = asyncio.get_event_loop()
        result_df = await loop.run_in_executor(executor, _run_adapters, df, req.smiles_col, req.engines, req.options)
        
        return {
            "columns": result_df.columns.tolist(),
            "shape": result_df.shape,
            "preview": result_df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Descriptor calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
