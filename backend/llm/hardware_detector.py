# backend/llm/hardware_detector.py
"""
Hardware detection utility for LLM optimization.
Detects CPU cores, RAM, GPU type, and VRAM.

This module is a compatibility layer that re-exports from hardware_check.py.
New code should import directly from hardware_check.
"""

from backend.llm.hardware_check import (
    HardwareProfile,
    get_hardware_profile,
    recommend_models,
    ModelRecommendation,
    get_best_model,
    check_model_compatibility,
    format_hardware_report,
)

# Backward-compatible aliases
detect_hardware = get_hardware_profile

__all__ = [
    "HardwareProfile",
    "get_hardware_profile",
    "detect_hardware",
    "recommend_models",
    "ModelRecommendation",
    "get_best_model",
    "check_model_compatibility",
    "format_hardware_report",
]
