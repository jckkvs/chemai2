"""Shared Pydantic schemas"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal

# Import from main to avoid duplication if main is already defined
# But since main depends on these, it's better to have them here or re-export
# The user's specification says "Re-export main schemas for easy import"
# So I'll just follow that but I might need to move them here if circular imports occur.
# For now, I'll just put the re-exports as requested.

from .main import (
    MetricsSchema, UploadResponse, ColumnConfig, PipelineConfig, 
    AnalysisResult, APIError
)

__all__ = [
    "MetricsSchema", "UploadResponse", "ColumnConfig", "PipelineConfig",
    "AnalysisResult", "APIError"
]
