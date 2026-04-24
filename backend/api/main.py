# backend/api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import uuid
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChemAI Nexus API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
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
            "preview": []
        }
    return SESSIONS[session_id]

class UploadResponse(BaseModel):
    success: bool
    filename: str
    rows: int
    cols: int
    target_col: str
    task_type: str
    metrics: dict
    preview: List[dict]
    columns: List[str]

class PipelineConfig(BaseModel):
    cv_folds: int = 5
    num_scaler: str = "standard"
    num_imputer: str = "median"
    cat_encoder: str = "onehot"
    feature_selector: str = "none"
    selected_models: List[str] = []
    monotonic_constraints: Dict[str, int] = {}

@app.post("/api/session/init")
async def init_session():
    session_id = str(uuid.uuid4())
    get_session(session_id)
    logger.info(f"Initialized session: {session_id}")
    return {"session_id": session_id}

@app.post("/api/upload", response_model=UploadResponse)
async def upload_data(session_id: str = Body(...), file: UploadFile = File(...)):
    session = get_session(session_id)
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents), float_precision='high')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # 数値精度保証（data_tab.py から移植）
        for col in df.select_dtypes(include=['float16', 'float32', 'int8', 'int16', 'int32', 'int64']).columns:
            df[col] = df[col].astype('float64')

        # 自動カラム検出
        target_col = df.columns[-1]
        smiles_col = None
        for col in df.columns:
            if col.lower() == "smiles":
                smiles_col = col
                break
        task_type = "regression" if pd.api.types.is_float_dtype(df[target_col]) else "classification"

        session["df"] = df
        session["filename"] = file.filename
        session["target_col"] = target_col
        session["task_type"] = task_type
        session["smiles_col"] = smiles_col
        
        numeric_cols = df.select_dtypes(include='number').shape[1]
        missing_rate = float(df.isna().mean().mean())
        session["metrics"] = {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_rate": missing_rate,
            "numeric_cols": numeric_cols
        }

        preview = df.head(8).to_dict(orient="records")
        for row in preview:
            for k, v in row.items():
                if pd.isna(v): row[k] = None
                elif isinstance(v, float): row[k] = round(v, 4)
        session["preview"] = preview

        logger.info(f"✅ Data uploaded successfully: {file.filename}")

        return UploadResponse(
            success=True,
            filename=file.filename,
            rows=len(df),
            cols=len(df.columns),
            target_col=target_col,
            task_type=task_type,
            metrics=session["metrics"],
            preview=preview,
            columns=list(df.columns)
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/info")
async def get_data_info(session_id: str):
    session = get_session(session_id)
    if session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    return {
        "filename": session["filename"],
        "columns": list(session["df"].columns),
        "target_col": session["target_col"],
        "task_type": session["task_type"],
        "metrics": session["metrics"],
        "preview": session["preview"]
    }

@app.post("/api/config/columns")
async def update_columns(session_id: str = Body(...), target_col: str = Body(...), task_type: Optional[str] = None):
    session = get_session(session_id)
    if session["df"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    session["target_col"] = target_col
    if task_type:
        session["task_type"] = task_type
    return {"status": "updated", "target_col": session["target_col"], "task_type": session["task_type"]}

@app.post("/api/pipeline/config")
async def update_pipeline_config(session_id: str = Body(...), config: PipelineConfig = Body(...)):
    session = get_session(session_id)
    session["config"].update(config.model_dump())
    return {"status": "updated", "config": session["config"]}

@app.get("/api/pipeline/config")
async def get_pipeline_config(session_id: str):
    session = get_session(session_id)
    return session["config"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
