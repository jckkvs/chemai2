"""
backend/llm/providers/openai_provider.py

OpenAI APIを使ったLLMプロバイダー実装。

使用方法:
    import os
    os.environ["OPENAI_API_KEY"] = "sk-..."

    from backend.llm import register_llm_provider
    from backend.llm.providers.openai_provider import OpenAIProvider
    register_llm_provider("openai", OpenAIProvider)

    # その後
    from backend.llm import get_llm_provider
    provider = get_llm_provider("openai")

必要パッケージ:
    pip install openai
"""
from __future__ import annotations

import logging
import os

from backend.llm.provider import LLMProvider, LLMProviderError, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI Chat Completions APIを使ったプロバイダー。

    対応モデル: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo 等
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return f"OpenAI ({self.model})"

    @property
    def description(self) -> str:
        return "OpenAI GPT-4o/4o-mini を使って記述子コードを生成します。"

    @property
    def is_available(self) -> bool:
        """OPENAI_API_KEY が設定されており、openaiパッケージが存在するか確認。"""
        if not self._api_key:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """OpenAI Chat Completions APIを呼び出す。"""
        try:
            import openai
        except ImportError as e:
            raise LLMProviderError(
                "openaiパッケージが未インストールです。`pip install openai` を実行してください。"
            ) from e

        if not self._api_key:
            raise LLMProviderError(
                "OPENAI_API_KEY が設定されていません。"
                "環境変数 OPENAI_API_KEY を設定するか、"
                "OpenAIProvider(api_key='sk-...') で指定してください。"
            )

        client = openai.OpenAI(api_key=self._api_key)

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            content = completion.choices[0].message.content or ""
            is_truncated = completion.choices[0].finish_reason != "stop"
            tokens_used = (
                completion.usage.total_tokens if completion.usage else 0
            )
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                is_truncated=is_truncated,
                raw=completion,
            )
        except openai.APIError as e:
            raise LLMProviderError(f"OpenAI APIエラー: {e}") from e
