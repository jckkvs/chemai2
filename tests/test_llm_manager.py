# tests/test_llm_manager.py

"""
Integration tests for LLMManager service layer

Tests cover:
- State transitions and locking
- Model switching and cleanup
- Benchmark execution with caching
- Error handling and graceful degradation
"""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from backend.llm.manager import LLMManager, LLMState
from backend.llm.hardware_detector import HardwareProfile
from backend.llm.model_selector import LLMModelConfig
from backend.llm.config import LLMSettings


@pytest.fixture
def mock_hardware():
    """Mock hardware profile"""
    hw = MagicMock(spec=HardwareProfile)
    hw.cpu_cores = 8
    hw.ram_total_gb = 32.0
    hw.ram_available_gb = 24.0
    hw.gpu_name = "RTX 3060"
    hw.vram_total_gb = 6.0
    hw.vram_available_gb = 5.5
    hw.device = "cuda"
    hw.architecture = "x86_64"
    hw.estimated_model_capacity_gb = 7.0
    return hw


@pytest.fixture
def mock_model_config():
    """Mock model configuration"""
    cfg = MagicMock(spec=LLMModelConfig)
    cfg.repo_id = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    cfg.filename = "qwen2.5-3b-instruct-q4_k_m.gguf"
    cfg.context_length = 8192
    cfg.n_gpu_layers = -1
    cfg.expected_size_gb = 2.1
    cfg.description = "Test Model"
    return cfg


@pytest.fixture
def llm_manager(mock_hardware, mock_model_config):
    """Provide initialized LLMManager with mocks"""
    manager = LLMManager()
    manager._hardware = mock_hardware
    manager._current_model = mock_model_config
    manager._state = LLMState.UNINITIALIZED
    manager._settings = LLMSettings()
    
    # Mock engine
    mock_engine = AsyncMock()
    mock_engine._model = MagicMock()
    mock_engine.initialize = AsyncMock()
    mock_engine.stream_chat = AsyncMock()
    async def mock_stream():
        yield "Hello "
        yield "world!"
    mock_engine.stream_chat.return_value = mock_stream()
    manager._engine = mock_engine
    
    yield manager


@pytest.mark.asyncio
class TestLLMManagerState:
    """Tests for state management and locking"""
    
    async def test_initial_state(self, llm_manager):
        """Manager should start in UNINITIALIZED state"""
        assert llm_manager.state == LLMState.UNINITIALIZED
    
    async def test_ready_state_returns_status(self, llm_manager):
        """When READY, get_status should return valid dict"""
        llm_manager._state = LLMState.READY
        status = llm_manager._get_status()
        assert status["state"] == "ready"
        assert status["loaded"] is True
        assert status["current_model"] is not None
    
    async def test_concurrent_initialize_lock(self, llm_manager, mock_hardware, mock_model_config):
        """Multiple concurrent initialize calls should not cause race conditions"""
        call_count = 0
        original_init = llm_manager._engine.initialize
        
        async def slow_init(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return original_init(*args, **kwargs)
        
        llm_manager._engine.initialize = slow_init
        llm_manager._state = LLMState.UNINITIALIZED
        
        # Run 3 concurrent initializations
        tasks = [llm_manager.initialize() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Only one should succeed in loading (lock protection)
        successful = sum(1 for r in results if isinstance(r, dict) and r["state"] == "ready")
        assert successful >= 1
        assert call_count == 1  # Engine initialized exactly once


@pytest.mark.asyncio
class TestModelSwitching:
    """Tests for model switching functionality"""
    
    async def test_switch_model_success(self, llm_manager, mock_model_config):
        """Should unload current and load new model"""
        llm_manager._state = LLMState.READY
        
        with patch("backend.llm.manager.MODEL_REGISTRY", {"test_model": mock_model_config}):
            result = await llm_manager.switch_model("test_model")
        
        assert result["state"] == "ready"
        assert llm_manager._engine.unload_model.call_count == 1
        assert llm_manager._engine.initialize.call_count == 1
    
    async def test_switch_model_invalid_name(self, llm_manager):
        """Should raise ValueError for unknown model"""
        with pytest.raises(ValueError, match="Unknown model"):
            await llm_manager.switch_model("nonexistent_model")
    
    async def test_switch_model_while_loading(self, llm_manager):
        """Should raise RuntimeError if switching during load"""
        llm_manager._state = LLMState.LOADING_MODEL
        with pytest.raises(RuntimeError, match="Cannot switch model"):
            await llm_manager.switch_model("qwen2.5-3b")


@pytest.mark.asyncio
class TestBenchmarkIntegration:
    """Tests for benchmark execution via manager"""
    
    async def test_run_benchmark_changes_state(self, llm_manager, mock_model_config):
        """Benchmark should transition state to BENCHMARKING and back"""
        llm_manager._state = LLMState.READY
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.speed_tps = 45.0
        mock_result.memory_peak_gb = 2.5
        mock_result.quality_score = 0.85
        mock_runner.run_benchmark.return_value = mock_result
        mock_runner.get_user_recommendation.return_value = {"model_name": "qwen2.5-3b"}
        llm_manager._benchmark_runner = mock_runner
        
        with patch("backend.llm.manager.MODEL_REGISTRY", {"test_model": mock_model_config}):
            result = await llm_manager.run_benchmark("test_model")
        
        assert result["speed_tps"] == 45.0
        assert llm_manager._state == LLMState.READY  # Restored after benchmark
    
    async def test_benchmark_blocked_during_load(self, llm_manager):
        """Benchmark should fail if model is loading"""
        llm_manager._state = LLMState.LOADING_MODEL
        with pytest.raises(RuntimeError, match="Another LLM operation"):
            await llm_manager.run_benchmark("test_model")


@pytest.mark.asyncio
class TestErrorHandling:
    """Tests for graceful error recovery"""
    
    async def test_stream_chat_uninitialized(self, llm_manager):
        """Should raise RuntimeError if not ready"""
        llm_manager._state = LLMState.UNINITIALIZED
        with pytest.raises(RuntimeError, match="LLM not ready"):
            async for _ in llm_manager.stream_chat("test"):
                pass
    
    async def test_stream_chat_error_propagation(self, llm_manager):
        """Should propagate engine errors correctly"""
        llm_manager._state = LLMState.READY
        llm_manager._engine.stream_chat = AsyncMock(side_effect=RuntimeError("GPU OOM"))
        
        with pytest.raises(RuntimeError, match="GPU OOM"):
            async for _ in llm_manager.stream_chat("test"):
                pass
    
    async def test_initialize_failure_sets_error_state(self, llm_manager, mock_hardware):
        """Initialization failure should set state to ERROR"""
        with patch("backend.llm.manager.detect_hardware", side_effect=RuntimeError("No hardware")):
            with pytest.raises(RuntimeError):
                await llm_manager.initialize()
        
        assert llm_manager.state == LLMState.ERROR
