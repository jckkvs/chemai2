"""
backend/llm/providers/ollama_provider.py

Ollamaプロバイダー実装。
ローカルで動作するOllamaサーバー経由でLLMを利用する。

Ollama API: http://localhost:11434
ドキュメント: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import logging
import json
from typing import Any

import httpx

from backend.llm.provider import LLMProvider, LLMRequest, LLMResponse, LLMProviderError
from backend.llm.model_registry import OLLAMA_MODELS, get_model_by_name

logger = logging.getLogger(__name__)

OLLAMA_API_BASE = "http://localhost:11434/api"


class OllamaProvider(LLMProvider):
    """
    Ollamaプロバイダー。

    使用方法:
        from backend.llm import register_llm_provider
        from backend.llm.providers.ollama_provider import OllamaProvider
        register_llm_provider("ollama", OllamaProvider)
    """

    def __init__(self, base_url: str = OLLAMA_API_BASE) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=120.0)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def is_available(self) -> bool:
        """Ollamaサーバーが起動しているかチェック。"""
        try:
            response = self._client.get(f"{self._base_url}/tags", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollamaサーバーに接続できません: {e}")
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Ollama APIを使用してテキストを生成する。

        Args:
            request: LLMRequestオブジェクト

        Returns:
            LLMResponseオブジェクト

        Raises:
            LLMProviderError: APIエラー・接続エラー等
        """
        model_name = request.extra.get("model", "qwen3:8b")

        if get_model_by_name(model_name) is None:
            available = [m.ollama_name for m in OLLAMA_MODELS]
            raise LLMProviderError(
                f"未知のOllamaモデル: {model_name}。"
                f"利用可能: {available}"
            )

        prompt = request.user_prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{request.user_prompt}"

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            logger.info(f"[Ollama] Generating with model: {model_name}")
            response = self._client.post(
                f"{self._base_url}/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("response", ""),
                model=model_name,
                tokens_used=data.get("eval_count", 0),
                is_truncated=False,
                raw=data,
            )

        except httpx.TimeoutException as e:
            raise LLMProviderError(f"Ollamaリクエストがタイムアウトしました: {e}") from e
        except httpx.HTTPStatusError as e:
            raise LLMProviderError(
                f"Ollama APIエラー (status {e.response.status_code}): {e.response.text}"
            ) from e
        except Exception as e:
            raise LLMProviderError(f"Ollama生成中にエラー: {e}") from e

    def list_local_models(self) -> list[str]:
        """
        ローカルにpull済みのOllamaモデル一覧を返す。

        Returns:
            モデル名のリスト（例: ["qwen3:8b", "deepseek-r1:7b"]）
        """
        try:
            response = self._client.get(f"{self._base_url}/tags", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"モデル一覧の取得に失敗: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """
        指定されたモデルをOllamaにpullする。

        Args:
            model_name: Ollamaモデル名（例: "qwen3:8b"）

        Returns:
            pullに成功したかどうか
        """
        try:
            logger.info(f"[Ollama] Pulling model: {model_name}")
            response = self._client.post(
                f"{self._base_url}/pull",
                json={"name": model_name},
                timeout=600.0,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"モデルpullに失敗: {e}")
            return False
