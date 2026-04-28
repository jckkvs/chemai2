# backend/llm/model_selector.py

"""
Model selection logic for LLM engine.
Selects best model based on task, hardware, and performance history.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from backend.llm.hardware_detector import detect_hardware, HardwareProfile
from backend.llm.benchmarks import (
    get_benchmark_for_hardware,
    apply_benchmark_settings,
    HARDWARE_BENCHMARKS,
)
from backend.llm.benchmark_runner import BenchmarkRunner

logger = logging.getLogger(__name__)

@dataclass
class LLMModelConfig:
    """Configuration for a specific LLM model"""
    repo_id: str
    filename: str
    expected_size_gb: float
    description: str
    context_length: int = 4096
    n_gpu_layers: int = -1
    chemistry_fine_tuned: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)

# Default registry of supported models (Updated 2026-01-22)
MODEL_REGISTRY: Dict[str, LLMModelConfig] = {
    "qwen3.5-3b": LLMModelConfig(
        repo_id="Qwen/Qwen3.5-3B-Instruct-GGUF",
        filename="qwen3.5-3b-instruct-q4_k_m.gguf",
        expected_size_gb=2.2,
        description="Next-gen lightweight model, ultra-fast.",
    ),
    "qwen3.5-7b": LLMModelConfig(
        repo_id="Qwen/Qwen3.5-7B-Instruct-GGUF",
        filename="qwen3.5-7b-instruct-q4_k_m.gguf",
        expected_size_gb=4.8,
        description="New standard for MI research, balanced performance.",
    ),
    "qwen3.5-14b": LLMModelConfig(
        repo_id="Qwen/Qwen3.5-14B-Instruct-GGUF",
        filename="qwen3.5-14b-instruct-q4_k_m.gguf",
        expected_size_gb=9.5,
        description="High precision reasoning for complex tasks.",
    ),
    "deepseek-v3.2-3b": LLMModelConfig(
        repo_id="deepseek-ai/DeepSeek-V3.2-3B-Instruct-GGUF",
        filename="deepseek-v3.2-3b-instruct-q6_k.gguf",
        expected_size_gb=3.1,
        description="Superior mathematical and logical reasoning.",
    ),
    "solar-10.7b": LLMModelConfig(
        repo_id="upstage/Solar-10.7B-Instruct-v1.0-GGUF",
        filename="solar-10.7b-instruct-v1.0-q4_k_m.gguf",
        expected_size_gb=6.2,
        description="Optimized for Japanese instructions.",
    ),
}

def select_optimal_model(
    task: str = "general",
    profile: Optional[HardwareProfile] = None,
) -> LLMModelConfig:
    """
    Select the best available model for the given task and hardware.
    """
    if profile is None:
        profile = detect_hardware()
    
    # Selection logic based on ENV ID (Hardware Catalog)
    env_id = profile.env_id
    
    if task == "reasoning":
        selected = MODEL_REGISTRY["deepseek-v3.2-3b"]
    elif task == "japanese":
        selected = MODEL_REGISTRY["solar-10.7b"]
    elif env_id == "ENV020": # High-end (RTX 5080)
        selected = MODEL_REGISTRY["qwen3.5-14b"]
    elif env_id == "ENV007": # Standard (RTX 3060)
        selected = MODEL_REGISTRY["qwen3.5-7b"]
    else: # Entry or CPU
        selected = MODEL_REGISTRY["qwen3.5-3b"]

    # 【拡張点】ベンチマークベースの最適化適用
    try:
        # 1. ユーザー実測ベンチマークを最優先
        benchmark_runner = BenchmarkRunner()
        user_rec = benchmark_runner.get_user_recommendation(
            task="chemistry" if selected.chemistry_fine_tuned else "general",
            max_latency_ms=200.0,
        )
        
        if user_rec and user_rec["model_name"] in MODEL_REGISTRY:
            logger.info(f"Using user-benchmarked model: {user_rec['model_name']}")
            selected = MODEL_REGISTRY[user_rec["model_name"]]
        
        # 2. 事前調査済みベンチマークを次点で参照
        elif profile:
            matched_benchmark = get_benchmark_for_hardware(
                profile.cpu_cores,
                profile.ram_total_gb,
                profile.gpu_name,
                profile.vram_total_gb,
                profile.architecture,
            )
            if matched_benchmark:
                selected = apply_benchmark_settings(selected, matched_benchmark, profile)
                logger.debug(f"Applied pre-surveyed benchmark settings: {matched_benchmark.name}")
    
    except Exception as e:
        logger.warning(f"Benchmark integration failed, using dynamic selection: {e}")
        # Fallback to original dynamic selection (selected already set)
    
    return selected

def run_model_benchmark(
    model_name: str,
    profile: Optional[HardwareProfile] = None,
    test_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run benchmark for specific model and return results
    
    Convenience function for UI/API integration.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    config = MODEL_REGISTRY[model_name]
    runner = BenchmarkRunner()
    
    result = runner.run_benchmark(
        model_config=config,
        profile=profile,
        test_prompt=test_prompt,
    )
    
    return {
        "model": model_name,
        "speed_tps": result.speed_tps,
        "memory_gb": result.memory_peak_gb,
        "quality_score": result.quality_score,
        "recommendation": runner.get_user_recommendation(),
    }
