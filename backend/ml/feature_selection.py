"""
Feature Selection Registry - chemai2/backend/ml/feature_selection.py
"""
from typing import Dict, Type, Any
from sklearn.base import BaseEstimator

SELECTOR_REGISTRY: Dict[str, Any] = {}

def get_selector_class(name: str) -> Any:
    return None
