"""
backend/core/hardware_detector.py
CPU/GPU/RAM/VRAMを自動検出し、LLM実行可否を判定
既存機能と共存：既存の環境チェックは維持し、拡張として実装
"""
import platform
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import re

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class HardwareSpec:
    """検出されたハードウェア仕様"""
    # システム情報
    os: str = ""
    os_version: str = ""
    python_version: str = ""
    
    # CPU
    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_arch: str = ""  # 'x86_64', 'arm64', etc.
    cpu_flags: List[str] = field(default_factory=list)  # avx2, avx512, etc.
    
    # RAM
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    
    # GPU (複数対応)
    gpus: List[Dict] = field(default_factory=list)
    # 各GPU: {name, vendor, vram_total_gb, vram_available_gb, compute_capability, driver_version}
    
    # 推論エンジン対応
    supports_cuda: bool = False
    supports_metal: bool = False
    supports_rocm: bool = False
    supports_vulkan: bool = False
    supports_openvino: bool = False
    
    # 総合評価
    inference_tier: str = "unknown"  # 'cpu_only', 'entry_gpu', 'mid_gpu', 'high_gpu', 'multi_gpu', 'apple_silicon', 'datacenter'
    recommended_max_model_size: str = ""  # '8B', '32B', '70B', etc.
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HardwareSpec':
        return cls(**data)


class HardwareDetector:
    """
    ハードウェア仕様を自動検出するクラス
    llama.cpp, Ollama, vLLM等の実行可否を事前判定
    """
    
    # 検出フラグのマッピング
    CUDA_FLAGS = ['avx2', 'avx512f', 'avx512_vnni']
    METAL_PLATFORMS = ['darwin']
    ROCM_HINTS = ['amd', 'radeon', 'instinct']
    
    def __init__(self):
        self.spec = HardwareSpec()
        self._detect_system()
        self._detect_cpu()
        self._detect_ram()
        self._detect_gpus()
        self._evaluate_tier()
    
    def _detect_system(self):
        """OS・Python情報を検出"""
        self.spec.os = platform.system()
        self.spec.os_version = platform.version()
        self.spec.python_version = platform.python_version()
        self.spec.cpu_arch = platform.machine()
    
    def _detect_cpu(self):
        """CPU情報を検出"""
        if HAS_PSUTIL:
            self.spec.cpu_name = platform.processor() or "Unknown"
            self.spec.cpu_cores = psutil.cpu_count(logical=False) or 0
            self.spec.cpu_threads = psutil.cpu_count(logical=True) or 0
        else:
            # fallback: subprocessで取得
            try:
                if platform.system() == 'Windows':
                    result = subprocess.run(['wmic', 'cpu', 'get', 'Name,NumberOfCores,NumberOfLogicalProcessors'], 
                                          capture_output=True, text=True, check=True)
                    lines = result.stdout.strip().split('\n')[1:]
                    if lines:
                        parts = lines[0].split()
                        self.spec.cpu_name = ' '.join(parts[:-2])
                        self.spec.cpu_cores = int(parts[-2]) if len(parts) >= 2 else 0
                        self.spec.cpu_threads = int(parts[-1]) if len(parts) >= 1 else 0
                else:
                    result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)
                    for line in result.stdout.split('\n'):
                        if 'CPU(s):' in line and 'NUMA' not in line:
                            self.spec.cpu_threads = int(line.split(':')[1].strip())
                        elif 'Core(s) per socket:' in line:
                            self.spec.cpu_cores = int(line.split(':')[1].strip())
            except:
                self.spec.cpu_name = "Unknown"
                self.spec.cpu_cores = 0
                self.spec.cpu_threads = 0
        
        # CPUフラグ検出（AVX2/AVX-512）
        try:
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    if 'avx2' in content:
                        self.spec.cpu_flags.append('avx2')
                    if 'avx512' in content:
                        self.spec.cpu_flags.append('avx512')
            elif platform.system() == 'Darwin':
                # Apple SiliconはNEONサポート
                if self.spec.cpu_arch == 'arm64':
                    self.spec.cpu_flags.append('neon')
        except:
            pass
    
    def _detect_ram(self):
        """RAM情報を検出"""
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            self.spec.ram_total_gb = round(mem.total / (1024**3), 2)
            self.spec.ram_available_gb = round(mem.available / (1024**3), 2)
        else:
            try:
                if platform.system() == 'Windows':
                    result = subprocess.run(['wmic', 'memorychip', 'get', 'Capacity'], 
                                          capture_output=True, text=True, check=True)
                    total = sum(int(line.strip()) for line in result.stdout.split('\n')[1:] if line.strip())
                    self.spec.ram_total_gb = round(total / (1024**3), 2)
                else:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if 'MemTotal:' in line:
                                self.spec.ram_total_gb = round(int(line.split()[1]) / (1024**2), 2)
                                break
                self.spec.ram_available_gb = self.spec.ram_total_gb * 0.7  # 簡易見積もり
            except:
                self.spec.ram_total_gb = 0.0
                self.spec.ram_available_gb = 0.0
    
    def _detect_gpus(self):
        """GPU情報を検出（NVIDIA/AMD/Intel/Apple）"""
        gpus = []
        
        # NVIDIA GPU検出（nvidia-smi）
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', 
                                   '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True, check=True, timeout=10)
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'name': parts[0],
                        'vendor': 'nvidia',
                        'vram_total_gb': round(float(parts[1]) / 1024, 2),
                        'vram_available_gb': round(float(parts[2]) / 1024, 2),
                        'driver_version': parts[3],
                        'compute_capability': self._get_cuda_capability(parts[0])
                    })
            self.spec.supports_cuda = True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        # Apple Metal検出
        if platform.system() == 'Darwin' and self.spec.cpu_arch == 'arm64':
            try:
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True, check=True)
                if 'Metal Support' in result.stdout or 'Apple M' in result.stdout:
                    # Unified MemoryをVRAMとして扱う
                    gpus.append({
                        'name': f"Apple {self.spec.cpu_name or 'Silicon'}",
                        'vendor': 'apple',
                        'vram_total_gb': self.spec.ram_total_gb,  # 共有メモリ
                        'vram_available_gb': self.spec.ram_available_gb * 0.8,  # 80%をGPU利用可能と仮定
                        'driver_version': platform.mac_ver()[0],
                        'unified_memory': True
                    })
                    self.spec.supports_metal = True
            except:
                pass
        
        # AMD ROCm検出（簡易）
        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True, check=True, timeout=10)
            if 'AMD' in result.stdout:
                # 簡易解析：VRAMは仮値
                gpus.append({
                    'name': 'AMD GPU (ROCm)',
                    'vendor': 'amd',
                    'vram_total_gb': 16.0,  # 仮値、実際はrocminfoから解析
                    'vram_available_gb': 14.0,
                    'driver_version': 'ROCm',
                })
                self.spec.supports_rocm = True
        except:
            pass
        
        # Intel GPU検出（OpenVINO用）
        try:
            if platform.system() == 'Linux':
                result = subprocess.run(['lspci'], capture_output=True, text=True, check=True)
                if 'Intel' in result.stdout and ('Graphics' in result.stdout or 'Iris' in result.stdout):
                    gpus.append({
                        'name': 'Intel Integrated GPU',
                        'vendor': 'intel',
                        'vram_total_gb': 2.0,  # 共有メモリ
                        'vram_available_gb': 1.5,
                        'driver_version': 'OpenVINO',
                    })
                    self.spec.supports_openvino = True
        except:
            pass
        
        # Vulkan検出（llama.cpp用）
        try:
            result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True, check=True, timeout=10)
            if 'GPU id' in result.stdout:
                self.spec.supports_vulkan = True
        except:
            pass
        
        self.spec.gpus = gpus
    
    def _get_cuda_capability(self, gpu_name: str) -> Optional[str]:
        """GPU名からCompute Capabilityを推定"""
        # 簡易マッピング（実際はtorch.cuda.get_device_capability()を使用）
        mappings = {
            'RTX 5090': '10.0', 'RTX 5080': '10.0', 'RTX 5070': '10.0',
            'RTX 4090': '8.9', 'RTX 4080': '8.9', 'RTX 4070': '8.9',
            'RTX 3090': '8.6', 'RTX 3080': '8.6', 'RTX 3070': '8.6',
            'A100': '8.0', 'H100': '9.0', 'V100': '7.0',
        }
        for pattern, cap in mappings.items():
            if pattern.lower() in gpu_name.lower():
                return cap
        return None
    
    def _evaluate_tier(self):
        """検出ハードウェアから推論ティアを評価"""
        gpus = self.spec.gpus
        total_vram = sum(g['vram_total_gb'] for g in gpus)
        max_single_vram = max((g['vram_total_gb'] for g in gpus), default=0)
        
        # Apple Silicon判定
        if self.spec.supports_metal:
            self.spec.inference_tier = 'apple_silicon'
            if total_vram >= 96:
                self.spec.recommended_max_model_size = '120B'
            elif total_vram >= 48:
                self.spec.recommended_max_model_size = '70B'
            elif total_vram >= 24:
                self.spec.recommended_max_model_size = '35B'
            else:
                self.spec.recommended_max_model_size = '12B'
            return
        
        # 複数GPU判定
        if len(gpus) >= 2:
            self.spec.inference_tier = 'multi_gpu'
            if total_vram >= 120:
                self.spec.recommended_max_model_size = '120B'
            elif total_vram >= 60:
                self.spec.recommended_max_model_size = '70B'
            elif total_vram >= 32:
                self.spec.recommended_max_model_size = '35B'
            else:
                self.spec.recommended_max_model_size = '12B'
            return
        
        # 単一GPU判定
        if max_single_vram >= 30:
            self.spec.inference_tier = 'high_gpu'  # RTX 5090 32GB等
            self.spec.recommended_max_model_size = '50B'
        elif max_single_vram >= 20:
            self.spec.inference_tier = 'mid_gpu'  # RTX 4090/5080 16-24GB
            self.spec.recommended_max_model_size = '32B'
        elif max_single_vram >= 12:
            self.spec.inference_tier = 'entry_gpu'  # RTX 3060/4060 Ti 12-16GB
            self.spec.recommended_max_model_size = '14B'
        else:
            self.spec.inference_tier = 'cpu_only'
            self.spec.recommended_max_model_size = '4B'
        
        # CPUのみの場合の補足
        if self.spec.inference_tier == 'cpu_only':
            if self.spec.ram_total_gb >= 64:
                self.spec.recommended_max_model_size = '14B'
                self.spec.notes.append("64GB RAMあり: 14BモデルまでCPU推論可能（低速）")
            elif self.spec.ram_total_gb >= 32:
                self.spec.recommended_max_model_size = '8B'
                self.spec.notes.append("32GB RAM: 8Bモデルまで推奨")
            else:
                self.spec.notes.append("RAM不足: 4B以下の軽量モデルのみ推奨")
    
    def get_spec(self) -> HardwareSpec:
        """検出結果を返す"""
        return self.spec
    
    def to_json(self, path: Optional[str] = None) -> str:
        """JSON形式で出力"""
        json_str = json.dumps(self.spec.to_dict(), ensure_ascii=False, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_json(cls, path: str) -> 'HardwareDetector':
        """JSONから復元（テスト用）"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        detector = cls.__new__(cls)
        detector.spec = HardwareSpec.from_dict(data)
        return detector
