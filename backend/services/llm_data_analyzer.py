# backend/services/llm_data_analyzer.py
import os
import json
import logging
import yaml
from typing import Dict, Any, Optional
import pandas as pd
from backend.llm.base import LLMProvider
from backend.llm.providers.transformers_backend import TransformersLLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """あなたは化学・材料科学・データサイエンスの専門アナリストです。
提供されたデータセットの構造・統計・ドメイン特性を評価し、最適な解析方針を策定してください。
出力は以下のJSON形式のみで返してください。外部テキストを含めないこと。
{
  "data_overview": "データの種類・サイズ・欠損状況・SMILESの有無などの概要",
  "preprocessing": "推奨前処理ステップ",
  "feature_engineering": "推奨記述子生成・選択戦略",
  "model_candidates": ["候補モデル1", "候補モデル2"],
  "validation_strategy": "検証手法と理由",
  "interpretation_plan": "XAI/解釈性の適用方針",
  "cautions": "注意点・ドメイン制約"
}"""

class LLMDataAnalyzer:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.provider: LLMProvider = self._init_provider()

    def _load_config(self, path: Optional[str]) -> Dict[str, Any]:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "provider": "transformers",
            "model_id": "./models/bonsai-8b-1.58bit",
            "max_tokens": 512,
            "temperature": 0.3
        }

    def _init_provider(self) -> LLMProvider:
        provider_type = self.config.get("provider", "transformers").lower()
        model_id = self.config.get("model_id", "")
        if provider_type == "transformers":
            return TransformersLLMProvider(model_id=model_id)
        # 将来の拡張ポイント: ollama, openai_compatible 等の分岐を追加
        raise ValueError(f"Unsupported LLM provider: {provider_type}")

    async def analyze(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.provider.is_available():
            return {"error": "LLMプロバイダーが初期化されていません。設定またはモデルパスを確認してください。"}
        
        context = self.provider.prepare_dataframe_context(df, metadata)
        prompt = f"{SYSTEM_PROMPT}\n\nデータメタデータ:\n{json.dumps(context, indent=2, ensure_ascii=False)}\n\n解析方針をJSON形式で出力してください。"
        
        try:
            raw_output = await self.provider.generate(prompt, max_tokens=self.config.get("max_tokens", 512))
            json_start = raw_output.find("{")
            json_end = raw_output.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                return json.loads(raw_output[json_start:json_end])
            return {"raw_output": raw_output, "warning": "JSONパースに失敗しました。出力テキストを確認してください。"}
        except Exception as e:
            logger.error(f"LLM解析実行エラー: {e}")
            return {"error": str(e)}
