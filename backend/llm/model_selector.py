# backend/llm/model_selector.py

"""
Model selection logic for LLM engine.
Selects best model based on task, hardware, and performance history.

Updated 2026-04-29 to use hardware_check module for HuggingFace quantized models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# ── LLM Model Configuration ────────────────────────────────────────────────────

@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model."""
    repo_id: str = ""
    filename: str = ""
    label: str = ""
    context_length: int = 4096
    n_gpu_layers: int = -1
    expected_size_gb: float = 0.0
    quantized_size_gb: float = 0.0
    description: str = ""
    chat_template: str = "chatml"
    require_gpu: bool = False


# ── Model Registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, LLMModelConfig] = {
    "qwen2.5-coder-1.5b": LLMModelConfig(
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        label="Qwen2.5-Coder 1.5B (推奨・軽量)",
        context_length=8192,
        n_gpu_layers=-1,
        expected_size_gb=3.0,
        quantized_size_gb=3.0,
        description="コード生成特化。CPUでも実用速度。",
        chat_template="chatml",
        require_gpu=False,
    ),
    "qwen2.5-coder-7b": LLMModelConfig(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        label="Qwen2.5-Coder 7B (高品質)",
        context_length=8192,
        n_gpu_layers=-1,
        expected_size_gb=14.0,
        quantized_size_gb=14.0,
        description="上位モデル。高品質なコードを生成。",
        chat_template="chatml",
        require_gpu=False,
    ),
    "gemma-3-1b": LLMModelConfig(
        repo_id="google/gemma-3-1b-it",
        filename="gemma-3-1b-it-q4_k_m.gguf",
        label="Gemma 3 1B (超軽量)",
        context_length=4096,
        n_gpu_layers=-1,
        expected_size_gb=2.0,
        quantized_size_gb=2.0,
        description="Google製超軽量モデル。低スペック環境向け。",
        chat_template="gemma",
        require_gpu=False,
    ),
    "phi-4-mini": LLMModelConfig(
        repo_id="microsoft/Phi-4-mini-instruct",
        filename="phi-4-mini-instruct-q4_k_m.gguf",
        label="Phi-4 Mini (バランス型)",
        context_length=4096,
        n_gpu_layers=-1,
        expected_size_gb=8.0,
        quantized_size_gb=8.0,
        description="Microsoft製。推論品質とサイズのバランスが良い。",
        chat_template="phi",
        require_gpu=False,
    ),
    "granite-3.3-2b": LLMModelConfig(
        repo_id="ibm-granite/granite-3.3-2b-instruct",
        filename="granite-3.3-2b-instruct-q4_k_m.gguf",
        label="IBM Granite 3.3 2B (コード特化)",
        context_length=4096,
        n_gpu_layers=-1,
        expected_size_gb=4.0,
        quantized_size_gb=4.0,
        description="IBM製コード生成モデル。科学技術分野に強い。",
        chat_template="granite",
        require_gpu=False,
    ),
}


# Import hardware_check functions directly to avoid circular imports
try:
    from backend.llm.hardware_check import (
        HardwareProfile,
        get_hardware_profile,
        recommend_models,
        get_best_model,
        ModelRecommendation,
    )
except ImportError:
    # Fallback: define dummy classes if import fails
    HardwareProfile = None
    get_hardware_profile = None
    recommend_models = None
    get_best_model = None
    ModelRecommendation = None

from backend.llm.benchmarks import (
    get_benchmark_for_hardware,
    apply_benchmark_settings,
    HARDWARE_BENCHMARKS,
)
from backend.llm.benchmark_runner import BenchmarkRunner

logger = logging.getLogger(__name__)


def select_optimal_model(
    task: str = "general",
    profile: Optional[HardwareProfile] = None,
    use_gguf_fallback: bool = True,
) -> Optional[LLMModelConfig]:
    """
    Select the best available model for the given task and hardware.

    Args:
        task: "general", "reasoning", "japanese", "chemistry"
        profile: Hardware profile. If None, auto-detects.
        use_gguf_fallback: If True, fall back to GGUF models for Ollama/local runners.

    Returns:
        LLMModelConfig object with the best model, or None.
    """
    if profile is None:
        profile = get_hardware_profile()

    # Get HuggingFace quantized model recommendations
    recommendations = recommend_models(profile)
    selected_rec = None
    for rec in recommendations:
        if rec.can_run:
            selected_rec = rec
            break

    # If no HF model found and GGUF fallback enabled, try GGUF models
    if selected_rec is None and use_gguf_fallback:
        logger.info("No HuggingFace model compatible, trying GGUF fallback")
        return _select_gguf_model(task, profile)

    if selected_rec is None:
        logger.warning("No compatible model found for current hardware")
        # Return the first model anyway (user can try manually)
        if recommendations:
            selected_rec = recommendations[0]

    if selected_rec is None:
        return None

    # Apply benchmark settings if available
    try:
        benchmark_runner = BenchmarkRunner()
        user_rec = benchmark_runner.get_user_recommendation(
            task="chemistry" if task == "chemistry" else "general",
            max_latency_ms=200.0,
        )

        if user_rec and user_rec.get("model_name"):
            # Find the corresponding model in MODEL_REGISTRY
            for key, config in MODEL_REGISTRY.items():
                if config.repo_id == user_rec["model_name"]:
                    logger.info(f"Using user-benchmarked model: {user_rec['model_name']}")
                    return config
    except Exception as e:
        logger.warning(f"Benchmark integration failed: {e}")

    # Return the LLMModelConfig from MODEL_REGISTRY based on model_id
    model_id = selected_rec.model_id
    # Find the matching key in MODEL_REGISTRY
    for key, config in MODEL_REGISTRY.items():
        if config.repo_id == model_id or key in model_id:
            return config

    logger.warning(f"Model {model_id} not found in MODEL_REGISTRY")
    return None


def _select_gguf_model(task: str, profile: HardwareProfile) -> Optional[ModelRecommendation]:
    """Fallback for GGUF models (Ollama, etc.)."""
    # This is a simplified version; the original model_selector had GGUF models
    # For now, return None to indicate no model found
    logger.info("GGUF model selection not implemented in this version")
    return None


def get_model_recommendations(
    profile: Optional[HardwareProfile] = None,
) -> List[ModelRecommendation]:
    """Get all model recommendations sorted by compatibility and score."""
    return recommend_models(profile)


def check_model_compatibility(
    model_id: str,
    profile: Optional[HardwareProfile] = None,
) -> tuple[bool, str]:
    """Check if a specific model can run on the current hardware."""
    from backend.llm.hardware_check import check_model_compatibility as _check
    return _check(model_id, profile)
