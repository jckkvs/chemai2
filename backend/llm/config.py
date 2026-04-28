# backend/llm/config.py

"""
LLM-specific persistent configuration management

Handles loading, validation, and saving of LLM preferences.
Stored in: ~/.config/chemai/llm_settings.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class LLMSettings(BaseModel):
    """LLM configuration schema with validation"""
    # Model selection
    preferred_model: Optional[str] = Field(None, description="Override auto-selection")
    prefer_chemistry_model: bool = Field(True, description="Boost chemistry-fine-tuned models")
    force_cpu: bool = Field(False, description="Disable GPU even if available")
    
    # Performance
    max_latency_ms: float = Field(200.0, ge=50, le=1000, description="Max acceptable latency per token")
    default_temperature: float = Field(0.7, ge=0.0, le=1.5)
    default_max_tokens: int = Field(1024, ge=64, le=4096)
    context_override: Optional[int] = Field(None, ge=2048, le=16384)
    
    # Paths & cache
    cache_dir: str = Field("~/.cache/chemai/llm")
    config_dir: str = Field("~/.config/chemai")
    auto_download: bool = Field(True, description="Automatically download missing models")
    
    # State tracking
    last_loaded_model: Optional[str] = None
    last_benchmark_run: Optional[str] = None
    
    @validator('cache_dir', 'config_dir')
    def resolve_tilde(cls, v):
        return str(Path(v).expanduser())
    
    class Config:
        extra = "forbid"


# Global config paths
CONFIG_DIR = Path("~/.config/chemai").expanduser()
CONFIG_FILE = CONFIG_DIR / "llm_settings.json"

def load_settings() -> LLMSettings:
    """Load settings from disk with fallback to defaults"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CONFIG_FILE.exists():
        logger.info("No LLM settings found, creating defaults.")
        default = LLMSettings()
        save_settings(default)
        return default
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LLMSettings(**data)
    except Exception as e:
        logger.warning(f"Failed to load LLM settings, using defaults: {e}")
        return LLMSettings()

def save_settings(settings: LLMSettings):
    """Persist settings to disk"""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(settings.dict(), f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save LLM settings: {e}")
