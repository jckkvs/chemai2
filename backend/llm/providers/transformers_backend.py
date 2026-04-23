# backend/llm/providers/transformers_backend.py
import os
import asyncio
import logging
import json
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from backend.llm.base import LLMProvider
from backend.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

class TransformersLLMProvider(LLMProvider):
    def __init__(self, model_id: str = "jckkvs/bonsai-8b-1.58bit", local_dir: Optional[str] = None, **kwargs):
        self.model_id = model_id
        self.local_dir = local_dir
        self.load_kwargs = kwargs
        self.model_manager = ModelManager(model_id, local_dir)
        self.pipeline = None
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return
        try:
            # 同期コンテキストから非同期ダウンロード保証を呼び出す
            # 実際には setup_model ページで既にダウンロードされていることが期待される
            model_path = asyncio.get_event_loop().run_until_complete(self.model_manager.ensure_downloaded())
            self.local_dir = model_path
        except Exception as e:
            logger.error(f"モデルダウンロード/解決失敗: {e}")
            self.pipeline = None
            return

        logger.info(f"Transformers モデル読込開始: {self.local_dir}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                # [TODO: PyTorch 2.4+] torch.jit.script 非推奨に伴い、
                # 将来的には torch.compile() への移行を検討する。
                # 現状は既存コードを維持し、警告は compatibility.py で抑制。
                **self.load_kwargs
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.9
            )
            self._loaded = True
            logger.info("Transformers モデル読込完了")
        except Exception as e:
            logger.error(f"Transformers モデル読込失敗: {e}")
            self.pipeline = None

    def is_available(self) -> bool:
        self._load_model()
        return self.pipeline is not None

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        self._load_model()
        if self.pipeline is None:
            raise RuntimeError("Transformers モデルが利用できません。")
        
        loop = asyncio.get_event_loop()
        def _inference():
            gen_config = {"max_new_tokens": max_tokens, "temperature": temperature, "top_p": 0.9, "do_sample": True}
            return self.pipeline(prompt, **gen_config)[0]["generated_text"]
        return await loop.run_in_executor(None, _inference)
