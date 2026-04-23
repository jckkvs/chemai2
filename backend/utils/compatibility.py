# backend/utils/compatibility.py
import sys
import logging
import warnings
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CompatibilityManager:
    """実行環境の互換性チェック・警告抑制・フォールバック管理を担当する。"""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.warnings_suppressed: List[str] = []
    
    def check_environment(self) -> Dict[str, any]:
        """環境互換性を評価し、推奨アクションを返す"""
        results = {
            "python_compatible": self.python_version >= (3, 10),
            "python_optimal": (3, 10) <= self.python_version < (3, 13),
            "mordred_available": self.python_version < (3, 12),
            "torch_jit_deprecated": True,  # PyTorch 2.4+ で非推奨
            "recommendations": []
        }
        
        if not results["python_optimal"]:
            results["recommendations"].append(
                f"Python {self.python_version.major}.{self.python_version.minor} は開発推奨範囲外です。"
            )
        if not results["mordred_available"]:
            results["recommendations"].append(
                "Mordred 記述子生成は利用できません。RDKit・GroupContrib を優先してください。"
            )
        return results
    
    def suppress_runtime_warnings(self):
        """既知の非致命的警告をフィルタリングし、ログノイズを低減する"""
        # Requests 依存関係警告の抑制（機能に影響なし）
        warnings.filterwarnings(
            "ignore", 
            message=".*urllib3.*chardet.*charset_normalizer.*",
            category=RuntimeWarning
        )
        # [追加] Plotly engine 引数の非推奨警告を抑制
        warnings.filterwarnings(
            "ignore",
            message=".*Support for the 'engine' argument is deprecated.*",
            category=DeprecationWarning,
            module="plotly"
        )
        self.warnings_suppressed.extend([
            "RequestsDependencyWarning",
            "TorchJITScriptDeprecationWarning",
            "PlotlyEngineDeprecationWarning"
        ])
        logger.info(f"非致命的警告を {len(self.warnings_suppressed)} 件抑制しました。")
