# backend/routers/llm.py

"""
FastAPI router for LLM benchmarking and management.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends, Body

from backend.llm.manager import LLMManager, LLMState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/llm",
    tags=["LLM Optimization"]
)
manager = LLMManager()

@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get status of LLM engine and available models"""
    return manager._get_status()

@router.post("/initialize")
async def initialize_llm(
    preferred_model: Optional[str] = Query(None),
    force_cpu: bool = Query(False),
) -> Dict[str, Any]:
    """Initialize LLM system"""
    return await manager.initialize(preferred_model=preferred_model, force_cpu=force_cpu)

@router.get("/benchmarks")
async def list_benchmarks() -> Dict[str, Any]:
    """List user's cached benchmark results"""
    return {
        "cached_results": manager._benchmark_runner.list_cached_benchmarks(),
        "recommendation": manager._benchmark_runner.get_user_recommendation(),
    }


@router.post("/benchmarks/run")
async def run_benchmark(
    model_name: str,
    test_prompt: Optional[str] = Query(None, description="Custom test prompt"),
) -> Dict[str, Any]:
    """Run benchmark for specific model on current hardware"""
    try:
        return await manager.run_benchmark(model_name, test_prompt=test_prompt)
    except Exception as e:
        logger.exception(f"Benchmark failed for {model_name}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.post("/benchmarks/clear")
async def clear_benchmarks(
    model_name: Optional[str] = Query(None, description="Clear specific model only"),
) -> Dict[str, str]:
    """Clear cached benchmark results"""
    manager._benchmark_runner.clear_cache(model_name)
    return {"status": "cleared", "model": model_name or "all"}


@router.post("/chat")
async def chat_with_llm(
    message: str = Body(..., embed=True),
    temperature: Optional[float] = Query(None),
    max_tokens: Optional[int] = Query(None),
) -> Dict[str, str]:
    """Chat with LLM (simple non-streaming response for now)"""
    try:
        full_response = ""
        async for token in manager.stream_chat(message, temperature=temperature, max_tokens=max_tokens):
            full_response += token
        return {"response": full_response}
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hardware/profiles")
async def list_hardware_profiles() -> Dict[str, Any]:
    """List pre-surveyed hardware profiles for reference"""
    from backend.llm.benchmarks import HARDWARE_BENCHMARKS
    return {
        name: {
            "category": b.category,
            "specs": {
                "cpu_cores": b.cpu_cores,
                "ram_gb": b.ram_total_gb,
                "gpu": b.gpu_name,
                "vram_gb": b.vram_total_gb,
            },
            "recommended_models": list(b.recommended_settings.keys()),
            "notes": b.notes,
        }
        for name, b in HARDWARE_BENCHMARKS.items()
    }
