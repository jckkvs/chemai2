# backend/llm/benchmark_runner.py

"""
Hardware benchmark runner for user-specific performance tuning

Allows users to test models on their hardware and cache results
for future optimal model selection.
"""
from __future__ import annotations

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, asdict

try:
    from llama_cpp import Llama
    import psutil
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

from backend.llm.hardware_detector import detect_hardware
if TYPE_CHECKING:
    from backend.llm.hardware_detector import HardwareProfile
    from backend.llm.model_selector import LLMModelConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from running a benchmark on specific hardware"""
    hardware_hash: str
    model_name: str
    speed_tps: float  # Tokens per second
    memory_peak_gb: float
    quality_score: float  # 0.0-1.0 (heuristic or human-rated)
    context_length: int
    quantization: str
    n_gpu_layers: int
    timestamp: str
    notes: List[str] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []


class BenchmarkRunner:
    """
    Run benchmarks on user hardware and cache results
    
    Results are stored in ~/.cache/chemai/llm/benchmarks.json
    and used to improve future model selection.
    """
    
    CACHE_DIR = Path.home() / ".cache" / "chemai" / "llm" / "benchmarks"
    CACHE_FILE = CACHE_DIR / "user_benchmarks.json"
    
    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cached_results: Dict[str, BenchmarkResult] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, BenchmarkResult]:
        """Load previously cached benchmark results"""
        if not self.CACHE_FILE.exists():
            return {}
        
        try:
            with open(self.CACHE_FILE, "r") as f:
                data = json.load(f)
            
            results = {}
            for key, result_data in data.items():
                results[key] = BenchmarkResult(**result_data)
            return results
        except Exception as e:
            logger.warning(f"Failed to load benchmark cache: {e}")
            return {}
    
    def _save_cache(self):
        """Persist cached results to disk"""
        try:
            data = {k: asdict(v) for k, v in self._cached_results.items()}
            with open(self.CACHE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save benchmark cache: {e}")
    
    def _compute_hardware_hash(self, profile: "HardwareProfile") -> str:
        """Generate unique hash for hardware profile"""
        key_data = f"{profile.cpu_cores}:{profile.ram_total_gb:.1f}:{profile.gpu_name}:{profile.vram_total_gb:.1f}:{profile.architecture}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:12]
    
    def _estimate_quality_score(self, model_name: str, task: str = "general") -> float:
        """
        Heuristic quality score based on model characteristics
        """
        # Base scores from model size/architecture (rough proxy)
        base_scores = {
            "qwen2.5-1.5b": 0.72,
            "qwen2.5-3b": 0.81,
            "llama3.1-8b": 0.94,
            "qwen2.5-14b": 0.97,
            "chemllm-7b": 0.96,  # Chemistry-tuned bonus
        }
        
        base = base_scores.get(model_name, 0.80)
        
        # Task-specific adjustments
        if task == "chemistry" and "chem" in model_name.lower():
            base += 0.02  # Small bonus for domain match
        
        return min(1.0, base)
    
    def run_benchmark(
        self,
        model_config: "LLMModelConfig",
        profile: Optional["HardwareProfile"] = None,
        test_prompt: str = None,
        max_tokens: int = 128,
        warmup_tokens: int = 32,
    ) -> BenchmarkResult:
        """
        Run performance benchmark for a specific model on current hardware
        """
        if not BENCHMARK_AVAILABLE:
            raise ImportError("llama-cpp-python required for benchmarking")
        
        if profile is None:
            profile = detect_hardware()
        
        hardware_hash = self._compute_hardware_hash(profile)
        model_name = model_config.repo_id.split("/")[-1].replace("-GGUF", "").lower()
        
        # Check cache first
        cache_key = f"{hardware_hash}:{model_name}:{model_config.filename}"
        if cache_key in self._cached_results:
            logger.info(f"Using cached benchmark result for {model_name}")
            return self._cached_results[cache_key]
        
        # Download model if needed (simplified stub for this implementation)
        model_path = self.CACHE_DIR.parent / model_config.filename
        if not model_path.exists():
            logger.info(f"Downloading {model_config.filename} for benchmarking...")
            # Actual download logic would go here
        
        # Load model with benchmark settings
        logger.info(f"Loading {model_name} for benchmarking...")
        llama = Llama(
            model_path=str(model_path),
            n_ctx=model_config.context_length,
            n_gpu_layers=model_config.n_gpu_layers,
            n_threads=max(1, os.cpu_count() // 2),
            use_mlock=True,
            verbose=False,
        )
        
        # Prepare test prompt
        if test_prompt is None:
            test_prompt = "Explain the relationship between molecular weight and solubility in organic compounds."
        
        # Warmup generation
        logger.info("Running warmup generation...")
        _ = llama(test_prompt, max_tokens=warmup_tokens, stop=["</s>"], stream=False)
        
        # Memory tracking
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**3)
        
        # Timed generation
        logger.info(f"Generating {max_tokens} tokens for timing...")
        start_time = time.time()
        
        output = llama(
            test_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["</s>", "\n\n"],
            stream=False,
        )
        
        elapsed = time.time() - start_time
        actual_tokens = len(output["choices"][0]["text"].split())
        speed_tps = actual_tokens / elapsed if elapsed > 0 else 0
        
        memory_after = process.memory_info().rss / (1024**3)
        memory_peak_gb = memory_after - memory_before + model_config.expected_size_gb
        
        # Quality estimation (heuristic)
        quality_score = self._estimate_quality_score(model_name)
        
        # Cleanup
        del llama
        import gc; gc.collect()
        
        # Create result
        result = BenchmarkResult(
            hardware_hash=hardware_hash,
            model_name=model_name,
            speed_tps=round(speed_tps, 2),
            memory_peak_gb=round(memory_peak_gb, 2),
            quality_score=quality_score,
            context_length=model_config.context_length,
            quantization=model_config.filename.split(".")[-2] if "." in model_config.filename else "unknown",
            n_gpu_layers=model_config.n_gpu_layers,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            notes=[f"Tested with prompt: {test_prompt[:50]}..."],
        )
        
        # Cache result
        self._cached_results[cache_key] = result
        self._save_cache()
        
        logger.info(
            f"Benchmark complete: {model_name} @ {speed_tps:.1f} tokens/sec, "
            f"{memory_peak_gb:.2f}GB peak memory"
        )
        
        return result
    
    def get_user_recommendation(
        self,
        task: str = "general",
        max_latency_ms: float = 200.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Get best model recommendation based on user's cached benchmarks
        """
        if not self._cached_results:
            return None
        
        # Group results by model
        model_results: Dict[str, List[BenchmarkResult]] = {}
        for result in self._cached_results.values():
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result)
        
        # Find best model per task/latency constraint
        candidates = []
        for model_name, results in model_results.items():
            # Use best speed from cached runs
            best_speed = max(r.speed_tps for r in results)
            avg_memory = sum(r.memory_peak_gb for r in results) / len(results)
            avg_quality = sum(r.quality_score for r in results) / len(results)
            
            latency_per_token = 1000 / best_speed if best_speed > 0 else float('inf')
            
            if latency_per_token <= max_latency_ms:
                task_bonus = 1.2 if (task == "chemistry" and "chem" in model_name.lower()) else 1.0
                score = avg_quality * task_bonus / (latency_per_token / 100)
                candidates.append((score, model_name, best_speed, avg_memory, avg_quality))
        
        if not candidates:
            return None
        
        # Return best candidate
        best = max(candidates, key=lambda x: x[0])
        return {
            "model_name": best[1],
            "expected_speed_tps": best[2],
            "expected_memory_gb": round(best[3], 2),
            "estimated_quality": round(best[4], 2),
            "latency_per_token_ms": round(1000 / best[2], 1) if best[2] > 0 else None,
        }
    
    def list_cached_benchmarks(self) -> List[Dict[str, Any]]:
        """List all cached benchmark results"""
        return [asdict(r) for r in self._cached_results.values()]
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cached benchmarks (optionally for specific model)"""
        if model_name:
            keys_to_remove = [k for k in self._cached_results if model_name.lower() in k.lower()]
            for key in keys_to_remove:
                del self._cached_results[key]
        else:
            self._cached_results.clear()
        self._save_cache()
        logger.info(f"Cleared benchmark cache{' for ' + model_name if model_name else ''}")
