# backend/llm/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class LLMProvider(ABC):
    """LLMバックエンド抽象基底クラス。複数モデル/サービスに対応する拡張インタフェースを提供する。"""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def prepare_dataframe_context(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """DataFrameメタデータをLLMプロンプト用コンテキストに変換"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "sample_rows": df.head(3).to_dict(orient="records"),
            "metadata": metadata or {}
        }
