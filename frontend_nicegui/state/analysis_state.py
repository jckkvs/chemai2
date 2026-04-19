"""
NiceGUI用 分析状態管理クラス
リアクティブなUI更新と非同期処理の橋渡し
"""
from nicegui import ui
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import asyncio
import pandas as pd
import logging

from backend.pipeline.unified_pipeline import UnifiedAnalysisPipeline, PipelineResult
from backend.chem.descriptor_sets import DescriptorSet
from backend.models.monotonic_constraints import UnifiedConstraintManager

logger = logging.getLogger(__name__)

@dataclass
class AnalysisState:
    """分析セッションの状態を保持"""
    raw_data: Optional[pd.DataFrame] = None
    smiles_column: Optional[str] = None
    numeric_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    
    selected_set: Optional[DescriptorSet] = None
    constraint_manager: UnifiedConstraintManager = field(default_factory=UnifiedConstraintManager)
    
    merged_result: Optional[Any] = None  # MergedDataResult
    pipeline_result: Optional[PipelineResult] = None
    
    is_processing: bool = False
    log_messages: List[str] = field(default_factory=list)
    
    # UI更新用コールバック
    _on_update: Optional[callable] = None
    
    def set_update_callback(self, callback: callable):
        """UI更新トリガーを設定"""
        self._on_update = callback
    
    def _notify_ui(self):
        """状態変更をUIに反映"""
        if self._on_update:
            self._on_update()
        ui.refresh()
    
    async def run_analysis(
        self,
        model_type: str = 'lightgbm',
        cv_folds: int = 5
    ) -> PipelineResult:
        """非同期で分析パイプラインを実行"""
        if self.raw_data is None:
            raise ValueError("データが読み込まれていません")
        
        self.is_processing = True
        self.log_messages = []
        self._notify_ui()
        
        try:
            pipeline = UnifiedAnalysisPipeline()
            
            def progress(step: int, total: int, msg: str):
                self.log_messages.append(msg)
                self._notify_ui()
            
            import concurrent.futures
            loop = asyncio.get_running_loop()
            
            # io_boundで実行（NiceGUIのブロック防止）
            # utils/run_io_bound に相当する処理
            def run_sync():
                return pipeline.run(
                    df=self.raw_data,
                    smiles_column=self.smiles_column,
                    numeric_columns=self.numeric_columns,
                    descriptor_set=self.selected_set,
                    target_column=self.target_column,
                    constraints=self.constraint_manager,
                    model_type=model_type,
                    cv_folds=cv_folds,
                    progress_callback=progress
                )
                
            result = await loop.run_in_executor(None, run_sync)
            
            self.pipeline_result = result
            self.merged_result = result.merged_data
            self.log_messages.append("✅ 分析完了")
            
        except Exception as e:
            self.log_messages.append(f"❌ エラー: {str(e)}")
            logger.exception("Analysis failed")
            raise
        finally:
            self.is_processing = False
            self._notify_ui()
        
        return result
    
    def reset(self):
        """状態をリセット"""
        self.raw_data = None
        self.pipeline_result = None
        self.merged_result = None
        self.log_messages = []
        self.is_processing = False
        self.constraint_manager = UnifiedConstraintManager()
        self._notify_ui()


# グローバル状態インスタンス（NiceGUIアプリ内で共有）
state = AnalysisState()
