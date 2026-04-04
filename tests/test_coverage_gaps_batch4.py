import pytest
import pandas as pd
import numpy as np
import warnings

def test_tuner_optuna_edge_cases():
    from backend.models.tuner import HyperparameterTuner
    from backend.pipeline.pipeline_builder import PipelineConfig
    from backend.ui.param_schema import ParamSpec

    # Edge cases in tuner initialization
    with pytest.raises(ValueError):
        tuner = HyperparameterTuner(task="regression", estimator_key="invalid_no_exist")

    df = pd.DataFrame({"f1": [1, 2, 3]*10, "f2": [4, 5, 6]*10})
    y = np.array([0, 1, 0]*10)

def test_cv_manager_edge_cases():
    from backend.models.cv_manager import CVConfig, run_cross_validation, _NestedCVSearchWrapper
    from sklearn.linear_model import Ridge
    
    cv_conf = CVConfig(cv_key="kfold", n_splits=2)
    with pytest.warns(UserWarning):
        # Trigger warnings
        res = run_cross_validation(Ridge(), pd.DataFrame({"a": [1, 2]}), np.array([1, 2]), cv_config=cv_conf)

def test_automl_edge_cases():
    from backend.models.automl import AutoMLEngine
    df = pd.DataFrame({"f1": [1, 2, 3]*4, "target": [1, 2, 3]*4, "group": [1, 1, 2]*4})
    
    # Task inference classification
    engine = AutoMLEngine(task="auto", model_keys=["ridge"])
    with pytest.raises(ValueError, match="データが少なすぎます"):
        engine.run(df, target_col="target")
