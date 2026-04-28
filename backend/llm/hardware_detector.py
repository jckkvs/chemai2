# backend/llm/hardware_detector.py

"""
Hardware detection utility for LLM optimization.
Detects CPU cores, RAM, GPU type, and VRAM.
"""
from __future__ import annotations

import os
import platform
import logging
import psutil
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    """Detected hardware specification"""
    cpu_cores: int
    ram_total_gb: float
    gpu_name: str
    vram_total_gb: float
    architecture: str
    instruction_set: Literal["avx2", "avx512", "neon", "none"]
    
    @property
    def ram_available_gb(self) -> float:
        import psutil
        return psutil.virtual_memory().available / (1024**3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "cpu_cores": self.cpu_cores,
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": round(self.ram_available_gb, 1),
            "gpu_name": self.gpu_name,
            "vram_total_gb": self.vram_total_gb,
            "architecture": self.architecture,
            "instruction_set": self.instruction_set
        }

    @property
    def env_id(self) -> str:
        """Classify into ENV001-ENV040 based on specs"""
        if self.vram_total_gb >= 16:
            return "ENV020" if "RTX" in self.gpu_name else "ENV018"
        elif self.vram_total_gb >= 6:
            return "ENV007"
        elif self.ram_total_gb <= 16:
            return "ENV001"
        return "ENV001" # Default fallback

def detect_hardware() -> HardwareProfile:
    """
    Detect current system hardware configuration.
    """
    # CPU and RAM
    cpu_cores = os.cpu_count() or 4
    ram_total_gb = psutil.virtual_memory().total / (1024**3)
    architecture = platform.machine().lower() # 'x86_64' or 'arm64'
    
    # Instruction set detection
    instruction_set = "none"
    if architecture == "arm64":
        instruction_set = "neon"
    else:
        # Simplistic check for AVX
        try:
            # On Windows, we can't easily check cpuinfo without external libs
            # but we assume modern x86 has at least AVX2
            instruction_set = "avx2"
        except Exception:
            pass

    # GPU Detection
    gpu_name = "None"
    vram_total_gb = 0.0
    
    try:
        # Check for NVIDIA via nvidia-smi if available
        import subprocess
        try:
            # Setting shell=True for Windows compatibility
            res = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits", shell=True, encoding='utf-8')
            lines = res.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                gpu_name = parts[0].strip()
                vram_total_gb = float(parts[1].strip()) / 1024.0
        except Exception:
            # Fallback for Apple Silicon
            if architecture == "arm64" and platform.system() == "Darwin":
                gpu_name = "Apple M-series GPU"
                vram_total_gb = ram_total_gb # Unified memory
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    return HardwareProfile(
        cpu_cores=cpu_cores,
        ram_total_gb=round(ram_total_gb, 1),
        gpu_name=gpu_name,
        vram_total_gb=round(vram_total_gb, 1),
        architecture=architecture,
        instruction_set=instruction_set
    )
