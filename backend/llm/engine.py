# backend/llm/engine.py

"""
LLM Execution Engine based on llama-cpp-python

Handles the low-level lifecycle of the GGUF model, inference,
and chat history management.
"""
from __future__ import annotations

import os
import logging
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, List
from pathlib import Path

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

from backend.llm.hardware_detector import HardwareProfile
from backend.llm.model_selector import LLMModelConfig

logger = logging.getLogger(__name__)

class LLMEngine:
    """
    Low-level engine for LLM inference using llama-cpp-python
    """
    
    def __init__(self):
        self._model: Optional[Llama] = None
        self._history: List[Dict[str, str]] = []
        self._config: Optional[LLMModelConfig] = None
    
    async def initialize(
        self, 
        hardware: HardwareProfile, 
        model_config: LLMModelConfig,
        force_cpu: bool = False
    ):
        """Initialize the model with hardware-optimized settings"""
        if not HAS_LLAMA:
            raise ImportError("llama-cpp-python is not installed. Please install it to use the local LLM.")
        
        self.unload_model()
        
        # Determine model path (assuming they are in ~/.cache/chemai/llm)
        cache_dir = Path.home() / ".cache" / "chemai" / "llm"
        model_path = cache_dir / model_config.filename
        
        if not model_path.exists():
            # In a real app, we might trigger a download here
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Calculate GPU layers if not specified
        n_gpu_layers = model_config.n_gpu_layers
        if force_cpu:
            n_gpu_layers = 0
        elif n_gpu_layers == -1:
            # Simple heuristic: if we have an NVIDIA GPU and some VRAM
            if "nvidia" in hardware.gpu_name.lower() and hardware.vram_total_gb > 0:
                # Roughly offload most layers if model fits in VRAM
                if model_config.expected_size_gb < hardware.vram_total_gb * 0.8:
                    n_gpu_layers = 35 # Typical for 7-8B models
                else:
                    n_gpu_layers = 15 # Partial offload
            elif "apple" in hardware.gpu_name.lower():
                n_gpu_layers = -1 # Metal handles this
            else:
                n_gpu_layers = 0
        
        logger.info(f"Loading model {model_config.repo_id} with {n_gpu_layers} GPU layers...")
        
        # Load model in a separate thread to avoid blocking the event loop
        def _load():
            return Llama(
                model_path=str(model_path),
                n_ctx=model_config.context_length,
                n_gpu_layers=n_gpu_layers,
                n_threads=max(1, hardware.cpu_cores // 2),
                use_mmap=True,
                verbose=False,
            )
        
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, _load)
        self._config = model_config
        logger.info("Model loaded successfully.")

    def unload_model(self):
        """Free model memory"""
        if self._model:
            # llama-cpp-python's Llama object doesn't have an explicit unload
            # but deleting it and running GC helps.
            del self._model
            self._model = None
            import gc
            gc.collect()
            logger.info("Model unloaded.")

    async def stream_chat(
        self, 
        message: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1024
    ) -> AsyncGenerator[str, None]:
        """Stream chat tokens from the model"""
        if not self._model:
            raise RuntimeError("Model not initialized.")
        
        # Update history
        self._history.append({"role": "user", "content": message})
        
        # Build prompt from history (simplified)
        prompt = ""
        for m in self._history:
            role = "Assistant" if m["role"] == "assistant" else "User"
            prompt += f"{role}: {m['content']}\n"
        prompt += "Assistant: "
        
        # Stream response
        loop = asyncio.get_event_loop()
        
        # Use a queue to bridge the generator
        queue = asyncio.Queue()
        
        def _target():
            stream = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["User:", "\nUser:"],
                stream=True
            )
            for chunk in stream:
                token = chunk['choices'][0]['text']
                loop.call_soon_threadsafe(queue.put_nowait, token)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(asyncio.to_thread(_target))
        
        full_response = ""
        while True:
            token = await queue.get()
            if token is None:
                break
            full_response += token
            yield token
            
        self._history.append({"role": "assistant", "content": full_response})

    async def generate_report(self, analysis_data: Dict[str, Any], template: str = "standard") -> str:
        """Generate a structured report based on analysis results"""
        prompt = f"Generate a technical report for the following chemical machine learning analysis results:\n{analysis_data}\nTemplate: {template}"
        
        if not self._model:
            return f"Model not loaded. Analysis Summary: {list(analysis_data.keys())}"
        
        # For reports, we use non-streaming call
        loop = asyncio.get_event_loop()
        def _gen():
            res = self._model(prompt, max_tokens=2048, temperature=0.3)
            return res['choices'][0]['text']
            
        return await loop.run_in_executor(None, _gen)

    def clear_history(self):
        """Reset conversation history"""
        self._history = []
        logger.info("Chat history cleared.")
