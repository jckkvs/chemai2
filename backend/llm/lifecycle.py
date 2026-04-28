# backend/llm/lifecycle.py

"""
LLM Service Lifecycle Manager

Handles startup/shutdown logic for LLM engines and caches.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from backend.llm.manager import LLMManager

logger = logging.getLogger(__name__)

class LLMLifecycle:
    """Orchestrates LLM service initialization and cleanup"""
    
    @asynccontextmanager
    async def lifespan(self) -> AsyncGenerator[None, None]:
        """Lifespan context manager for FastAPI integration"""
        logger.info("Initializing LLM Service...")
        
        # We don't necessarily want to load a heavy model during API startup
        # but we can initialize the manager and detect hardware.
        manager = LLMManager()
        
        try:
            # Detect hardware early to cache it
            from backend.llm.hardware_detector import detect_hardware
            manager._hardware = detect_hardware()
            logger.info(f"LLM Hardware detected: {manager._hardware.gpu_name}")
            
            yield
            
        finally:
            logger.info("Shutting down LLM Service...")
            if manager._engine:
                manager._engine.unload_model()

# Global lifecycle instance
lifecycle = LLMLifecycle()
