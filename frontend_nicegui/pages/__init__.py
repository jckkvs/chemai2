"""
Pages package for ChemAI2 NiceGUI frontend
"""

from .welcome import page_welcome
from .data_upload import page_data_upload
from .llm_interview import page_llm_interview
from .preprocessing import page_preprocessing
from .eda import page_eda
from .decision_support import page_decision_support
from .ml_modeling import page_ml_modeling
from .doe import page_doe
from .results import page_results
from .settings import page_settings

__all__ = [
    'page_welcome',
    'page_data_upload',
    'page_llm_interview',
    'page_preprocessing',
    'page_eda',
    'page_decision_support',
    'page_ml_modeling',
    'page_doe',
    'page_results',
    'page_settings',
]
