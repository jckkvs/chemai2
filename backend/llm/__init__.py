# backend/llm/__init__.py
"""
LLM統合モジュール。

将来のLLM実装のための差し込み口（プロバイダーパターン）。
現在はスタブ実装のみ。実際のLLM連携はプロバイダーを追加して実現する。

使用例:
    from backend.llm import get_llm_provider, LLMDescriptorGenerator
    provider = get_llm_provider("openai")  # or "local", "stub"
    gen = LLMDescriptorGenerator(provider)
    code = gen.generate("沸点を予測する記述子を作成してください")
"""
from backend.llm.provider import LLMProvider, StubLLMProvider
from backend.llm.generator import LLMDescriptorGenerator
from backend.llm.registry import LLMProviderRegistry
from backend.llm.reviewer import LLMCodeReviewer, CodeReviewResult
from backend.llm.hardware_detector import detect_hardware, HardwareProfile
from backend.llm.benchmark_runner import BenchmarkRunner
from backend.llm.model_registry import (
    OllamaModelInfo,
    OLLAMA_MODELS,
    get_recommended_models,
    get_model_by_name,
    get_tier_label,
)

# Lazy imports to avoid circular imports
def get_llm_manager():
    """LLMManagerを遅延インポートで取得。"""
    from backend.llm.manager import LLMManager, LLMState
    return LLMManager()

def select_optimal_model(task="general", profile=None, use_gguf_fallback=True):
    """モデル選択を遅延インポートで実行。"""
    from backend.llm.model_selector import select_optimal_model as _som
    return _som(task=task, profile=profile, use_gguf_fallback=use_gguf_fallback)

def get_LLMModelConfig():
    """LLMModelConfigを遅延インポートで取得。"""
    from backend.llm.model_selector import LLMModelConfig
    return LLMModelConfig

_registry = LLMProviderRegistry()
_registry.register("stub", StubLLMProvider)

# HuggingFaceプロバイダーを登録（transformers がインストール済みの場合のみ）
try:
    from backend.llm.providers.hf_provider import HuggingFaceProvider
    _registry.register("huggingface", HuggingFaceProvider)
except ImportError:
    pass

# Ollamaプロバイダーを登録（httpx がインストール済みの場合のみ）
try:
    from backend.llm.providers.ollama_provider import OllamaProvider
    _registry.register("ollama", OllamaProvider)
except ImportError:
    pass


def get_llm_provider(name: str = "stub") -> LLMProvider:
    """登録済みLLMプロバイダーを取得する。"""
    return _registry.get(name)


def register_llm_provider(name: str, cls: type) -> None:
    """新しいLLMプロバイダーを登録する（プラグイン拡張用）。"""
    _registry.register(name, cls)


__all__ = [
    "LLMProvider",
    "StubLLMProvider",
    "LLMDescriptorGenerator",
    "LLMProviderRegistry",
    "LLMCodeReviewer",
    "CodeReviewResult",
    "get_llm_manager",
    "HardwareProfile",
    "BenchmarkRunner",
    "OllamaModelInfo",
    "OLLAMA_MODELS",
    "get_recommended_models",
    "get_model_by_name",
    "get_tier_label",
    "get_llm_provider",
    "register_llm_provider",
    "detect_hardware",
    "select_optimal_model",
    "get_LLMModelConfig",
]
