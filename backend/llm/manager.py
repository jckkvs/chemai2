# backend/llm/manager.py

"""
Unified LLM Service Manager

Orchestrates hardware detection, benchmark caching, model loading,
chat streaming, and report generation in a single async-safe interface.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional, Literal
from pathlib import Path
from enum import Enum

from backend.llm.hardware_detector import detect_hardware, HardwareProfile
from backend.llm.model_selector import select_optimal_model, LLMModelConfig, MODEL_REGISTRY
from backend.llm.benchmark_runner import BenchmarkRunner
from backend.llm.engine import LLMEngine
from backend.llm.config import LLMSettings, load_settings, save_settings
from backend.llm.domain_injector import domain_injector
from backend.llm.prompts import SYSTEM_PROMPT_ANALYSIS, ANALYSIS_REPORT_TEMPLATE

logger = logging.getLogger(__name__)


class LLMState(str, Enum):
    UNINITIALIZED = "uninitialized"
    DETECTING_HARDWARE = "detecting_hardware"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    BENCHMARKING = "benchmarking"
    ERROR = "error"


class LLMManager:
    """
    Singleton manager for all LLM-related operations
    
    Provides state-aware initialization, model switching, streaming chat,
    and benchmark execution with persistent settings.
    """
    _instance: Optional["LLMManager"] = None
    _state: LLMState = LLMState.UNINITIALIZED
    _state_lock: asyncio.Lock = asyncio.Lock()
    _settings: LLMSettings
    _hardware: Optional[HardwareProfile] = None
    _current_model: Optional[LLMModelConfig] = None
    _engine: Optional[LLMEngine] = None
    _benchmark_runner: BenchmarkRunner
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings = load_settings()
            cls._instance._benchmark_runner = BenchmarkRunner()
            cls._instance._engine = None
        return cls._instance
    
    @property
    def state(self) -> LLMState:
        return self._state
    
    @property
    def hardware(self) -> Optional[HardwareProfile]:
        return self._hardware
    
    @property
    def current_model(self) -> Optional[LLMModelConfig]:
        return self._current_model
    
    async def initialize(
        self,
        preferred_model: Optional[str] = None,
        force_cpu: bool = False,
        skip_benchmark_check: bool = False,
    ) -> Dict[str, Any]:
        """
        Initialize LLM system with hardware detection and model loading
        
        Args:
            preferred_model: Override auto-selection
            force_cpu: Disable GPU even if available
            skip_benchmark_check: Skip benchmark recommendation lookup
        
        Returns:
            Status dict with model info and performance expectations
        """
        async with self._state_lock:
            if self._state == LLMState.READY:
                return self._get_status()
            
            self._state = LLMState.DETECTING_HARDWARE
            logger.info("Starting LLM initialization...")
            
            try:
                # 1. Detect hardware
                self._hardware = detect_hardware()
                logger.info(f"Hardware detected: {self._hardware.gpu_name} | {self._hardware.ram_available_gb:.1f}GB RAM")
                
                # 2. Check user benchmarks for recommendation
                if not skip_benchmark_check:
                    rec = self._benchmark_runner.get_user_recommendation(
                        task="chemistry" if self._settings.prefer_chemistry_model else "general",
                        max_latency_ms=self._settings.max_latency_ms,
                    )
                    if rec and not preferred_model:
                        preferred_model = rec["model_name"]
                        logger.info(f"Using benchmark-recommended model: {preferred_model}")
                
                # 3. Select optimal model
                self._state = LLMState.LOADING_MODEL
                self._current_model = select_optimal_model(
                    self._hardware, preferred_model, force_cpu
                )
                logger.info(f"Selected model: {self._current_model.description}")
                
                # 4. Initialize engine
                if self._engine is None:
                    self._engine = LLMEngine()
                
                await self._engine.initialize(self._hardware, self._current_model, force_cpu)
                
                # 5. Update settings
                self._settings.last_loaded_model = self._current_model.repo_id.split("/")[-1]
                self._settings.force_cpu = force_cpu
                save_settings(self._settings)
                
                self._state = LLMState.READY
                logger.info("LLM initialization completed successfully.")
                return self._get_status()
                
            except Exception as e:
                self._state = LLMState.ERROR
                logger.error(f"LLM initialization failed: {e}", exc_info=True)
                raise RuntimeError(f"LLM initialization failed: {e}")
    
    async def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Switch to a different model with proper cleanup"""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        
        async with self._state_lock:
            if self._state not in (LLMState.READY, LLMState.ERROR):
                raise RuntimeError(f"Cannot switch model while in state: {self._state}")
            
            logger.info(f"Switching model to: {model_name}")
            self._state = LLMState.LOADING_MODEL
            
            try:
                # Unload current
                if self._engine:
                    self._engine.unload_model()
                
                # Select & load new
                self._current_model = MODEL_REGISTRY[model_name]
                await self._engine.initialize(self._hardware, self._current_model, self._settings.force_cpu)
                
                self._settings.last_loaded_model = model_name
                save_settings(self._settings)
                self._state = LLMState.READY
                
                return self._get_status()
            except Exception as e:
                self._state = LLMState.ERROR
                raise RuntimeError(f"Model switch failed: {e}")
    
    async def stream_chat(
        self,
        message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response with state validation"""
        if self._state != LLMState.READY or not self._engine or not self._engine._model:
            raise RuntimeError("LLM not ready. Please initialize first.")
        
        temp = temperature or self._settings.default_temperature
        tokens = max_tokens or self._settings.default_max_tokens
        
        async for token in self._engine.stream_chat(message, temperature=temp, max_tokens=tokens):
            yield token
    
    async def generate_report(
        self,
        analysis_data: Dict[str, Any],
        template_name: str = "standard",
    ) -> str:
        """Generate structured report with domain enrichment"""
        if self._state != LLMState.READY or not self._engine or not self._engine._model:
            raise RuntimeError("LLM not ready. Please initialize first.")
        
        # 1. Enrich prompt with domain knowledge
        enriched_prompt = domain_injector.enrich_analysis_prompt(
            SYSTEM_PROMPT_ANALYSIS, 
            analysis_data
        )
        
        # 2. Add template instructions
        if template_name == "standard":
            # Simple heuristic for now: just append template
            full_prompt = f"{enriched_prompt}\n\nPlease follow this template:\n{ANALYSIS_REPORT_TEMPLATE}"
        else:
            full_prompt = enriched_prompt
            
        return await self._engine.generate_report(analysis_data, template=full_prompt)
    
    async def run_benchmark(
        self,
        model_name: str,
        test_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run benchmark with state management"""
        if self._state == LLMState.LOADING_MODEL or self._state == LLMState.BENCHMARKING:
            raise RuntimeError("Another LLM operation is in progress.")
        
        self._state = LLMState.BENCHMARKING
        try:
            result = self._benchmark_runner.run_benchmark(
                MODEL_REGISTRY[model_name],
                self._hardware,
                test_prompt,
            )
            return {
                "speed_tps": result.speed_tps,
                "memory_gb": result.memory_peak_gb,
                "quality_score": result.quality_score,
                "recommendation": self._benchmark_runner.get_user_recommendation(),
            }
        finally:
            self._state = LLMState.READY if self._engine and self._engine._model else LLMState.UNINITIALIZED
    
    def clear_chat_history(self):
        """Reset conversation history"""
        if self._engine:
            self._engine.clear_history()
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current status snapshot"""
        return {
            "state": self._state.value,
            "hardware": self._hardware.to_dict() if self._hardware else None,
            "current_model": self._current_model.description if self._current_model else None,
            "loaded": self._engine is not None and self._engine._model is not None,
            "settings": {
                "temperature": self._settings.default_temperature,
                "max_tokens": self._settings.default_max_tokens,
                "force_cpu": self._settings.force_cpu,
            }
        }
