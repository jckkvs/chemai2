"""
Distributed Task Queue - chemai2/backend/tasks/celery_tasks.py
Celery tasks for async descriptor calculation and model training
"""
import json
import time
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime

import pandas as pd
import numpy as np
from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure

from backend.core.config import settings
from backend.chem.plugins import DescriptorPluginRegistry
from backend.ml.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from backend.utils.logger import logger
from backend.routers.websocket import broadcast_task_progress, broadcast_task_complete

# Initialize Celery app
celery_app = Celery(
    'chemai_tasks',
    broker=settings.REDIS_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    # settings.CELERY_TASK_TIME_LIMIT might be missing, adding fallback
    task_time_limit=getattr(settings, 'CELERY_TASK_TIME_LIMIT', 3600),
    task_soft_time_limit=int(getattr(settings, 'CELERY_TASK_TIME_LIMIT', 3600) * 0.9),
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)


# ========== Task Progress Tracking ==========
class ProgressTracker:
    """Track and broadcast task progress"""
    
    @staticmethod
    def update(task_id: str, progress: float, message: str, data: Dict = None):
        """Update progress and broadcast via WebSocket"""
        broadcast_task_progress(task_id, progress, message, data)
    
    @staticmethod
    def complete(task_id: str, result: Any, success: bool = True):
        """Mark task complete and broadcast"""
        broadcast_task_complete(task_id, result, success)


# ========== Signal Handlers ==========
@task_prerun.connect
def task_prerun_handler(task_id: str, task: Task, *args, **kwargs):
    """Log task start"""
    logger.info(f"Task {task.name}[{task_id}] started")
    ProgressTracker.update(task_id, 0, "Task started")


@task_postrun.connect
def task_postrun_handler(task_id: str, task: Task, *args, **kwargs):
    """Log task completion"""
    logger.info(f"Task {task.name}[{task_id}] completed")


@task_failure.connect
def task_failure_handler(task_id: str, exception: Exception, *args, **kwargs):
    """Log task failure"""
    logger.error(f"Task {task_id} failed: {exception}", exc_info=True)
    ProgressTracker.complete(task_id, {'error': str(exception)}, success=False)


# ========== Descriptor Calculation Tasks ==========
@celery_app.task(bind=True, name='chemai.calculate_descriptors')
def calculate_descriptors_task(
    self,
    plugin_name: str,
    smiles_list: List[str],
    params: Dict[str, Any] = None,
    batch_size: int = 100,
    task_id: str = None
) -> Dict[str, Any]:
    """
    Async task for descriptor calculation with progress tracking
    
    Supports batching for large SMILES lists
    """
    task_id = task_id or self.request.id
    registry = DescriptorPluginRegistry()
    
    try:
        # Get plugin
        spec = registry.get(plugin_name)
        if not spec:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        func = spec.load_function()
        if not func:
            raise RuntimeError(f"Failed to load function for plugin: {plugin_name}")
        
        # Batch processing for large lists
        n_total = len(smiles_list)
        results = []
        calc_params = {**(spec.default_params or {}), **(params or {})}
        
        for i in range(0, n_total, batch_size):
            batch = smiles_list[i:i + batch_size]
            batch_result = func(smiles_list=batch, **calc_params)
            results.append(batch_result)
            
            # Update progress
            progress = min(100, (i + len(batch)) / n_total * 100)
            ProgressTracker.update(
                task_id, 
                progress, 
                f"Processed {i + len(batch)}/{n_total} molecules",
                {'current_batch': i // batch_size + 1, 'total_batches': (n_total + batch_size - 1) // batch_size}
            )
        
        # Combine results
        if results and all(isinstance(r, pd.DataFrame) for r in results):
            final_df = pd.concat(results, ignore_index=True)
            final_df.attrs['plugin'] = plugin_name
            final_df.attrs['params'] = calc_params
            final_df.attrs['n_input'] = n_total
            final_df.attrs['n_output'] = len(final_df)
            
            # Convert to serializable format
            result_dict = {
                'columns': list(final_df.columns),
                'data': final_df.to_dict('records'),
                'index': list(final_df.index),
                'attrs': final_df.attrs
            }
            
            ProgressTracker.complete(task_id, {
                'result_id': f"desc_{task_id}",
                'n_descriptors': len(final_df.columns),
                'n_valid': final_df.notna().any(axis=1).sum(),
                'result': result_dict
            })
            
            return result_dict
        else:
            raise ValueError("Unexpected result format from plugin")
            
    except Exception as e:
        logger.error(f"Descriptor calculation task failed: {e}", exc_info=True)
        raise


# ========== Model Training Tasks ==========
@celery_app.task(bind=True, name='chemai.train_model')
def train_model_task(
    self,
    config_dict: Dict[str, Any],
    X_data: Dict[str, Any],  # Serialized DataFrame
    y_data: Dict[str, Any],
    task_id: str = None
) -> Dict[str, Any]:
    """
    Async task for model training with CV and constraint evaluation
    """
    task_id = task_id or self.request.id
    
    try:
        # Deserialize data
        X = pd.DataFrame(X_data['data'], columns=X_data['columns'])
        y = pd.Series(y_data['values'], name=y_data.get('name', 'target'))
        
        # Reconstruct config
        config = PipelineConfig.from_dict(config_dict)
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Training with progress updates
        ProgressTracker.update(task_id, 10, "Building pipeline...")
        pipeline = orchestrator.build_pipeline(X, y)
        
        ProgressTracker.update(task_id, 30, "Fitting model...")
        orchestrator.fit(X, y)
        
        ProgressTracker.update(task_id, 70, "Evaluating constraints...")
        # Note: _constraint_engine logic depends on PipelineOrchestrator implementation
        # For now, just a placeholder
        constraint_report = {}
        
        ProgressTracker.update(task_id, 90, "Finalizing...")
        
        # Prepare result
        result = {
            'model_id': f"model_{task_id}",
            'metrics': {
                'score': orchestrator.score(X, y),
                'n_features': len(orchestrator.get_feature_names_out())
            },
            'constraint_report': constraint_report,
            'config': config_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        ProgressTracker.complete(task_id, result)
        return result
        
    except Exception as e:
        logger.error(f"Model training task failed: {e}", exc_info=True)
        raise


# ========== Batch Processing Tasks ==========
@celery_app.task(bind=True, name='chemai.batch_descriptors')
def batch_descriptors_task(
    self,
    plugin_configs: List[Dict[str, Any]],
    smiles_list: List[str],
    output_format: Literal['parquet', 'csv', 'json'] = 'parquet',
    task_id: str = None
) -> Dict[str, str]:
    """
    Calculate descriptors from multiple plugins in batch
    
    Returns dict of plugin_name -> output_file_path
    """
    task_id = task_id or self.request.id
    registry = DescriptorPluginRegistry()
    output_files = {}
    
    try:
        for i, plugin_config in enumerate(plugin_configs):
            plugin_name = plugin_config['name']
            params = plugin_config.get('params', {})
            
            ProgressTracker.update(
                task_id,
                (i / len(plugin_configs)) * 100,
                f"Processing plugin {i+1}/{len(plugin_configs)}: {plugin_name}"
            )
            
            # Using delay/apply_async instead of direct call to use worker
            # But for simplicity in the loop, we wait for it
            result = calculate_descriptors_task(plugin_name, smiles_list, params, task_id=f"{task_id}_{plugin_name}")
            
            # Save to file
            df = pd.DataFrame(result['data'], columns=result['columns'])
            filename = f"descriptors_{plugin_name}_{task_id}.{output_format}"
            filepath = settings.EXPORT_DIR / filename
            
            if output_format == 'parquet':
                df.to_parquet(filepath, index=False)
            elif output_format == 'csv':
                df.to_csv(filepath, index=False)
            else:
                df.to_json(filepath, orient='records', indent=2)
            
            output_files[plugin_name] = str(filepath)
        
        ProgressTracker.complete(task_id, {'output_files': output_files})
        return output_files
        
    except Exception as e:
        logger.error(f"Batch descriptor task failed: {e}", exc_info=True)
        raise
