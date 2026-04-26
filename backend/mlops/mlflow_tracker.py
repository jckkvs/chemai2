"""
MLOps Tracker - chemai2/backend/mlops/mlflow_tracker.py
MLflow integration for experiment tracking and model registry
"""
import json
import os
import tempfile
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from backend.core.config import settings
from backend.utils.logger import logger


class MLOpsTracker:
    """
    MLflow wrapper for ChemAI ML Studio experiment tracking
    
    Features:
    - Automatic param/metric/artifact logging
    - Constraint metadata tracking
    - Model registry integration
    - Cross-validation fold logging
    - Run comparison utilities
    """
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "chemai_experiments"):
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI or "sqlite:///mlruns.db"
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()
        logger.info(f"MLflow initialized: URI={self.tracking_uri}, Experiment={self.experiment_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run with ChemAI-specific tags"""
        run = mlflow.start_run(run_name=run_name)
        
        # Default tags
        default_tags = {
            "project": "chemai_ml_studio",
            "version": "2.0.0",
            "platform": "python"
        }
        if tags:
            default_tags.update(tags)
        
        mlflow.set_tags(default_tags)
        logger.info(f"MLflow run started: {run.info.run_id}")
        return run
    
    def log_pipeline_config(self, config: Dict[str, Any], prefix: str = "pipeline"):
        """Log pipeline configuration as structured JSON"""
        mlflow.log_param(f"{prefix}/config", json.dumps(config, default=str))
        
        # Log key parameters for search/filtering
        if "task_type" in config:
            mlflow.log_param(f"{prefix}/task_type", config["task_type"])
        if "estimator_name" in config:
            mlflow.log_param(f"{prefix}/estimator", config["estimator_name"])
        if "cv_strategy" in config:
            mlflow.log_param(f"{prefix}/cv_strategy", config["cv_strategy"])
    
    def log_constraints(self, constraints: Dict[str, Dict[str, Any]]):
        """Log constraint specifications"""
        mlflow.log_param("constraints/active", len(constraints))
        mlflow.log_param("constraints/details", json.dumps(constraints, default=str))
        
        # Log per-feature constraints
        for feat, spec in constraints.items():
            mlflow.log_param(f"constraints/{feat}/monotonic", spec.get("monotonic"))
            mlflow.log_param(f"constraints/{feat}/linearity", spec.get("linearity"))
            mlflow.log_param(f"constraints/{feat}/sigma_range", spec.get("sigma_range"))
    
    def log_cv_results(self, cv_results: Dict[str, Any], prefix: str = "cv"):
        """Log cross-validation metrics"""
        mlflow.log_metrics({f"{prefix}/mean_score": cv_results.get("mean_score"),
                            f"{prefix}/std_score": cv_results.get("std_score"),
                            f"{prefix}/n_folds": cv_results.get("n_splits")})
        
        if "fold_scores" in cv_results:
            mlflow.log_param(f"{prefix}/fold_scores", json.dumps(cv_results["fold_scores"]))
    
    def log_model(self, model, artifact_path: str = "model", 
                  signature_input: Optional[pd.DataFrame] = None,
                  signature_output: Optional[pd.DataFrame] = None):
        """Log trained model with optional signature"""
        import mlflow.sklearn
        
        kwargs = {"artifact_path": artifact_path}
        if signature_input is not None and signature_output is not None:
            from mlflow.models.signature import infer_signature
            kwargs["signature"] = infer_signature(signature_input, signature_output)
        
        mlflow.sklearn.log_model(model, **kwargs)
        logger.info(f"Model logged to {artifact_path}")
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: str = None):
        """Log any file as MLflow artifact"""
        mlflow.log_artifact(str(local_path), artifact_path)
    
    def log_figure(self, fig, name: str, folder: str = "plots"):
        """Log Plotly/Matplotlib figure"""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.write_image(tmp.name)  # Requires kaleido
            mlflow.log_artifact(tmp.name, artifact_path=folder)
            os.unlink(tmp.name)
    
    def register_model(self, model_uri: str, model_name: str, stage: str = "Production"):
        """Register model in MLflow registry"""
        try:
            result = mlflow.register_model(model_uri, model_name)
            client = MlflowClient()
            client.transition_model_version_stage(name=model_name, version=result.version, stage=stage)
            logger.info(f"Model {model_name} v{result.version} registered as {stage}")
            return result.version
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            return None
    
    def compare_runs(self, run_ids: List[str], metrics: List[str] = None) -> pd.DataFrame:
        """Compare multiple runs in a DataFrame"""
        runs = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            data = {
                "run_id": run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time
            }
            data.update(run.data.params)
            data.update(run.data.metrics)
            runs.append(data)
        return pd.DataFrame(runs)


# Global tracker instance
mlops_tracker = MLOpsTracker()
