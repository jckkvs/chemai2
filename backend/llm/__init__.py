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

_registry = LLMProviderRegistry()
_registry.register("stub", StubLLMProvider)


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
    "get_llm_provider",
    "register_llm_provider",
]
