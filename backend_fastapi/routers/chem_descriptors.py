'''
backend_fastapi/routers/chem_descriptors.py
SMILES文字列から化学記述子を計算するエンドポイント
既存 backend.chem.* adapters をラップして提供
'''

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Literal
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chem", tags=["chem-descriptors"])

# ── Request/Response Schemas ──────────────────────────────
class SMILESRequest(BaseModel):
    """SMILES記述子計算リクエスト"""
    smiles_list: List[str] = Field(..., min_items=1, max_items=10000, description="SMILES文字列リスト")
    engines: List[Literal[
        "rdkit", "mordred", "group_contrib", "descriptastorus",
        "molai", "skfp", "uma", "mol2vec", "padel", "molfeat",
        "xtb", "unipka", "cosmo", "chemprop"
    ]] = Field(default=["rdkit"], description="使用する記述子エンジン")
    compute_fp: bool = Field(True, description="フィンガープリントも計算するか")
    n_components: int = Field(6, ge=1, le=100, description="次元削減時の成分数（MolAI等）")

    @validator('smiles_list')
    def validate_smiles(cls, v):
        # 空文字・Noneをフィルタ
        return [s.strip() for s in v if s and s.strip()]

class DescriptorResult(BaseModel):
    """記述子計算結果"""
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: float = Field(ge=0, le=1)
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

class DescriptorComputeRequest(BaseModel):
    """バッチ計算リクエスト（ファイルベース）"""
    file_hash: str
    smiles_column: str
    engines: List[str]
    options: Dict[str, Any] = Field(default_factory=dict)

# ── Cache & Job Management ──────────────────────────────
_descriptor_cache: Dict[str, Any] = {}
_job_status: Dict[str, DescriptorResult] = {}

def _compute_hash(smiles_list: List[str], engines: List[str], options: Dict) -> str:
    """入力パラメータからキャッシュキーを生成"""
    content = f"{sorted(smiles_list)}|{sorted(engines)}|{sorted(options.items())}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

async def _execute_descriptor_calculation(
    smiles_list: List[str],
    engines: List[str],
    options: Dict[str, Any],
    job_id: str
) -> Any:
    """既存 backend.chem adapters を呼び出して記述子を計算"""
    from backend_fastapi.services.chem_service import ChemDescriptorService
    service = ChemDescriptorService()

    def on_progress(step: int, total: int, message: str):
        if job_id in _job_status:
            _job_status[job_id].progress = step / total
            _job_status[job_id].message = message
            _job_status[job_id].updated_at = datetime.utcnow().isoformat()

    loop = asyncio.get_event_loop()
    result_df = await loop.run_in_executor(
        None,
        lambda: service.compute_descriptors(smiles_list, engines, options, on_progress)
    )
    return result_df

# ── Endpoints ────────────────────────────────────────────
@router.post("/descriptors/compute", response_model=DescriptorResult)
async def compute_descriptors(
    request: SMILESRequest,
    background_tasks: BackgroundTasks
):
    """SMILESリストから記述子を非同期計算"""
    job_id = hashlib.sha256(f"{datetime.utcnow().isoformat()}_{hash(str(request))}".encode()).hexdigest()[:12]

    # キャッシュチェック
    cache_key = _compute_hash(request.smiles_list, request.engines, request.dict())
    if cache_key in _descriptor_cache:
        return DescriptorResult(
            job_id=job_id,
            status="completed",
            progress=1.0,
            message="キャッシュから取得",
            result={"columns": _descriptor_cache[cache_key].columns.tolist(), "shape": _descriptor_cache[cache_key].shape},
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )

    # ジョブ登録
    _job_status[job_id] = DescriptorResult(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="初期化中...",
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )

    async def _run():
        try:
            _job_status[job_id].status = "running"
            result_df = await _execute_descriptor_calculation(
                request.smiles_list,
                request.engines,
                {"compute_fp": request.compute_fp, "n_components": request.n_components},
                job_id
            )
            _descriptor_cache[cache_key] = result_df
            _job_status[job_id].status = "completed"
            _job_status[job_id].progress = 1.0
            _job_status[job_id].message = "計算完了"
            _job_status[job_id].result = {
                "columns": result_df.columns.tolist(),
                "shape": result_df.shape,
                "sample": result_df.head(3).to_dict(orient="records")
            }
        except Exception as e:
            logger.error(f"Descriptor calculation failed: {e}", exc_info=True)
            _job_status[job_id].status = "failed"
            _job_status[job_id].error = str(e)
    background_tasks.add_task(_run)
    return _job_status[job_id]

@router.get("/descriptors/status/{job_id}", response_model=DescriptorResult)
async def get_descriptor_status(job_id: str):
    """計算ジョブのステータス取得"""
    if job_id not in _job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_status[job_id]

@router.get("/descriptors/result/{job_id}")
async def get_descriptor_result(job_id: str, format: Literal["json", "csv"] = "json"):
    """計算結果の取得（CSVダウンロード対応）"""
    if job_id not in _job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _job_status[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job.status}")
    cache_key = None
    for k, v in _descriptor_cache.items():
        if job.result and v.shape == tuple(job.result.get("shape", [])):
            cache_key = k
            break
    if not cache_key or cache_key not in _descriptor_cache:
        raise HTTPException(status_code=404, detail="Result data not found")
    df = _descriptor_cache[cache_key]
    if format == "csv":
        from fastapi.responses import Response
        import io
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return Response(
            content=csv_buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=descriptors_{job_id}.csv"}
        )
    return {
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "data": df.head(100).to_dict(orient="records") if len(df) > 100 else df.to_dict(orient="records")
    }

@router.post("/descriptors/batch")
async def batch_compute_descriptors(request: DescriptorComputeRequest):
    """ファイルベースのバッチ計算（大規模データ用）"""
    return {
        "task_id": f"batch_{hashlib.sha256(request.file_hash.encode()).hexdigest()[:12]}",
        "status": "queued",
        "message": "バッチキューに登録されました"
    }
