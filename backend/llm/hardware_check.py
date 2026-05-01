"""
backend/llm/hardware_check.py

Hardware-adaptive LLM selection module.
Checks system resources (RAM, CPU, GPU) and recommends quantized models
that can run on the current machine, similar to canirun.ai.

Usage:
    from backend.llm.hardware_check import get_hardware_profile, recommend_models
    profile = get_hardware_profile()
    models = recommend_models(profile)
"""

from __future__ import annotations

import platform
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import psutil


@dataclass
class HardwareProfile:
    """System hardware profile."""
    total_ram_gb: float
    available_ram_gb: float
    cpu_cores: int
    cpu_threads: int
    gpu_available: bool
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_count: int = 0
    system: str = ""
    architecture: str = ""
    python_version: str = ""
    is_64bit: bool = True

    def to_dict(self) -> dict:
        return {
            "total_ram_gb": round(self.total_ram_gb, 1),
            "available_ram_gb": round(self.available_ram_gb, 1),
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": round(self.gpu_vram_gb, 1),
            "gpu_count": self.gpu_count,
            "system": self.system,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "is_64bit": self.is_64bit,
        }


def get_hardware_profile() -> HardwareProfile:
    """Detect current system hardware profile."""
    # RAM
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024 ** 3)
    available_ram_gb = mem.available / (1024 ** 3)

    # CPU
    cpu_cores = psutil.cpu_count(logical=False) or 1
    cpu_threads = psutil.cpu_count(logical=True) or 1

    # GPU
    gpu_available = False
    gpu_name = ""
    gpu_vram_gb = 0.0
    gpu_count = 0

    # Try CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else ""
            try:
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            except Exception:
                pass
    except ImportError:
        pass

    # Fallback: try to detect GPU via other means if no CUDA
    if not gpu_available:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                gpu_count = len(lines)
                gpu_available = True
                parts = lines[0].split(", ")
                if len(parts) >= 2:
                    gpu_name = parts[0].strip()
                    try:
                        gpu_vram_gb = float(parts[1].strip()) / 1024
                    except ValueError:
                        pass
        except Exception:
            pass

    return HardwareProfile(
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_count=gpu_count,
        system=f"{platform.system()} {platform.release()}",
        architecture=platform.machine(),
        python_version=sys.version.split()[0],
        is_64bit=sys.maxsize > 2 ** 32,
    )


# Model requirements: rough estimates for quantized models
# format: model_id -> (min_ram_gb, recommended_ram_gb, requires_gpu, min_vram_gb)
MODEL_REQUIREMENTS = {
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {
        "min_ram_gb": 4.0,
        "recommended_ram_gb": 6.0,
        "requires_gpu": False,
        "min_vram_gb": 0.0,
        "quantized_size_gb": 3.0,
        "description": "1.5B parameters, 4-bit quantized ~3GB",
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "min_ram_gb": 16.0,
        "recommended_ram_gb": 20.0,
        "requires_gpu": False,
        "min_vram_gb": 8.0,
        "quantized_size_gb": 14.0,
        "description": "7B parameters, 4-bit quantized ~14GB",
    },
    "google/gemma-3-1b-it": {
        "min_ram_gb": 3.0,
        "recommended_ram_gb": 4.0,
        "requires_gpu": False,
        "min_vram_gb": 0.0,
        "quantized_size_gb": 2.0,
        "description": "1B parameters, 4-bit quantized ~2GB",
    },
    "microsoft/Phi-4-mini-instruct": {
        "min_ram_gb": 10.0,
        "recommended_ram_gb": 12.0,
        "requires_gpu": False,
        "min_vram_gb": 6.0,
        "quantized_size_gb": 8.0,
        "description": "Mini model, 4-bit quantized ~8GB",
    },
    "ibm-granite/granite-3.3-2b-instruct": {
        "min_ram_gb": 6.0,
        "recommended_ram_gb": 8.0,
        "requires_gpu": False,
        "min_vram_gb": 4.0,
        "quantized_size_gb": 4.0,
        "description": "2B parameters, 4-bit quantized ~4GB",
    },
}


@dataclass
class ModelRecommendation:
    """Recommendation result for a single model."""
    model_id: str
    label: str
    can_run: bool
    reason: str = ""
    score: float = 0.0  # Higher is better (0.0 - 1.0)
    size_gb: float = 0.0
    description: str = ""


def recommend_models(
    profile: Optional[HardwareProfile] = None,
    model_catalog: Optional[list[dict]] = None,
) -> list[ModelRecommendation]:
    """
    Recommend LLM models based on hardware profile.

    Args:
        profile: Hardware profile. If None, auto-detects.
        model_catalog: Model catalog from hf_provider.HF_MODEL_CATALOG.
                      If None, uses MODEL_REQUIREMENTS keys.

    Returns:
        List of ModelRecommendation sorted by score (best first).
    """
    if profile is None:
        profile = get_hardware_profile()

    if model_catalog is None:
        # Use default catalog
        from backend.llm.providers.hf_provider import HF_MODEL_CATALOG
        model_catalog = HF_MODEL_CATALOG

    recommendations = []

    for model_info in model_catalog:
        model_id = model_info["id"]
        label = model_info.get("label", model_id)
        size_gb = model_info.get("size_gb", 0.0)
        requires_gpu = model_info.get("require_gpu", False)
        description = model_info.get("description", "")

        # Get requirements
        req = MODEL_REQUIREMENTS.get(model_id, {
            "min_ram_gb": size_gb * 1.5,
            "recommended_ram_gb": size_gb * 2.0,
            "requires_gpu": requires_gpu,
            "min_vram_gb": size_gb,
            "quantized_size_gb": size_gb,
            "description": description,
        })

        # Check compatibility
        can_run = True
        reasons = []

        # Check RAM: need at least model size + 2GB for system overhead
        min_ram_needed = req["min_ram_gb"]
        if profile.available_ram_gb < min_ram_needed:
            can_run = False
            reasons.append(f"Available RAM ({profile.available_ram_gb:.1f}GB) < minimum required ({min_ram_needed:.1f}GB)")

        # Check GPU requirement
        if requires_gpu and not profile.gpu_available:
            can_run = False
            reasons.append("GPU required but not available")

        # Check VRAM if GPU is available
        if profile.gpu_available and req["min_vram_gb"] > 0:
            if profile.gpu_vram_gb < req["min_vram_gb"]:
                can_run = False
                reasons.append(f"GPU VRAM ({profile.gpu_vram_gb:.1f}GB) < minimum required ({req['min_vram_gb']:.1f}GB)")

        # Calculate score (0.0 - 1.0)
        if can_run:
            # Higher score for models that fit well
            ram_ratio = min(1.0, profile.available_ram_gb / req["recommended_ram_gb"])
            score = ram_ratio * 0.7 + (1.0 if not requires_gpu else 0.3)
            if profile.gpu_available:
                score = min(1.0, score + 0.2)
        else:
            score = 0.0

        recommendation = ModelRecommendation(
            model_id=model_id,
            label=label,
            can_run=can_run,
            reason="; ".join(reasons) if reasons else "Compatible",
            score=score,
            size_gb=size_gb,
            description=description,
        )
        recommendations.append(recommendation)

    # Sort by: can_run (True first), then score (descending)
    recommendations.sort(key=lambda x: (-int(x.can_run), -x.score))

    return recommendations


def get_best_model(
    profile: Optional[HardwareProfile] = None,
    model_catalog: Optional[list[dict]] = None,
) -> Optional[str]:
    """
    Get the best model ID that can run on current hardware.

    Returns:
        Model ID string, or None if no compatible model found.
    """
    recommendations = recommend_models(profile, model_catalog)
    for rec in recommendations:
        if rec.can_run:
            return rec.model_id
    return None


def check_model_compatibility(
    model_id: str,
    profile: Optional[HardwareProfile] = None,
) -> tuple[bool, str]:
    """
    Check if a specific model can run on the current hardware.

    Returns:
        (can_run, reason) tuple.
    """
    if profile is None:
        profile = get_hardware_profile()

    recommendations = recommend_models(profile)
    for rec in recommendations:
        if rec.model_id == model_id:
            return rec.can_run, rec.reason

    return False, f"Model {model_id} not found in catalog"


# ── Formatted output for UI ─────────────────────────────────────────────────────

def format_hardware_report(profile: Optional[HardwareProfile] = None, lang: str = "en") -> str:
    """Generate a human-readable hardware report.

    Args:
        profile: Hardware profile. If None, auto-detects.
        lang: "en" or "ja" (note: Japanese may have encoding issues on Windows).
    """
    if profile is None:
        profile = get_hardware_profile()

    if lang == "ja":
        lines = [
            "=== ハードウェア情報 ===",
            f"システム: {profile.system} ({profile.architecture})",
            f"Python: {profile.python_version} ({'64-bit' if profile.is_64bit else '32-bit'})",
            "",
            f"RAM: {profile.total_ram_gb:.1f}GB 総容量, {profile.available_ram_gb:.1f}GB 利用可能",
            f"CPU: {profile.cpu_cores} コア / {profile.cpu_threads} スレッド",
        ]

        if profile.gpu_available:
            lines.append(f"GPU: {profile.gpu_name} ({profile.gpu_vram_gb:.1f}GB VRAM, {profile.gpu_count} 台)")
        else:
            lines.append("GPU: 利用不可")

        lines.append("")
        lines.append("=== モデル推奨 ===")

        recommendations = recommend_models(profile)
        for rec in recommendations:
            status = "[OK]" if rec.can_run else "[NG]"
            lines.append(f"{status} {rec.label} ({rec.size_gb:.1f}GB) - {rec.reason}")
    else:
        lines = [
            "=== Hardware Profile ===",
            f"System: {profile.system} ({profile.architecture})",
            f"Python: {profile.python_version} ({'64-bit' if profile.is_64bit else '32-bit'})",
            "",
            f"RAM: {profile.total_ram_gb:.1f}GB total, {profile.available_ram_gb:.1f}GB available",
            f"CPU: {profile.cpu_cores} cores / {profile.cpu_threads} threads",
        ]

        if profile.gpu_available:
            lines.append(f"GPU: {profile.gpu_name} ({profile.gpu_vram_gb:.1f}GB VRAM, {profile.gpu_count} devices)")
        else:
            lines.append("GPU: Not available")

        lines.append("")
        lines.append("=== Model Recommendations ===")

        recommendations = recommend_models(profile)
        for rec in recommendations:
            status = "[OK]" if rec.can_run else "[NG]"
            lines.append(f"{status} {rec.label} ({rec.size_gb:.1f}GB) - {rec.reason}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(format_hardware_report())
