"""
config/__init__.py
Global configuration paths and constants
"""
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# LLM Catalog Paths
CATALOG_PATH = str(PROJECT_ROOT / "config" / "llm_catalog_40env.json")
BENCHMARK_CACHE_PATH = str(PROJECT_ROOT / "config" / "benchmark_cache.json")
ENVIRONMENTS_CONFIG_PATH = str(PROJECT_ROOT / "config" / "environments_40.json")

# LLM Analyzer Settings
LLM_ANALYZER_CONFIG = str(PROJECT_ROOT / "config" / "llm_analyzer.yaml")
LLM_SETTINGS_JSON = str(PROJECT_ROOT / "config" / "llm_settings.json")
