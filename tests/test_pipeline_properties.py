"""
Property-Based Tests - chemai2/tests/test_pipeline_properties.py
Hypothesis-driven testing for pipeline robustness and constraint enforcement
"""
import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

from backend.ml.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig, ColumnPreprocessingConfig
from backend.ml.constraints import ConstraintSpec
from backend.ml.cv_factory import CVConfig, run_cross_validation
from sklearn.ensemble import RandomForestRegressor


# ========== Hypothesis Strategies ==========
numeric_data = arrays(np.float64, shape=(50, 4), elements=st.floats(-100, 100, allow_nan=False))
target_data = arrays(np.float64, shape=(50,), elements=st.floats(-50, 50, allow_nan=False))
constraint_strategies = st.dictionaries(
    keys=st.sampled_from(["feat_0", "feat_1", "feat_2", "feat_3"]),
    values=st.builds(
        ConstraintSpec,
        feature_name=st.text(min_size=1, max_size=10),
        monotonic=st.sampled_from([None, "increasing", "decreasing"]),
        linearity=st.sampled_from(["none", "weak"]),
        sigma_range=st.floats(1.0, 5.0),
        strength=st.sampled_from(["weak", "strong"])
    ),
    max_size=3
)


@given(numeric_data, target_data, constraint_strategies)
@settings(deadline=2000, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_pipeline_deterministic_fit_predict(X_np, y_np, constraints):
    """Verify pipeline produces identical results given same random state"""
    X = pd.DataFrame(X_np, columns=["feat_0", "feat_1", "feat_2", "feat_3"])
    y = pd.Series(y_np, name="target")
    
    config = PipelineConfig(
        column_configs={col: ColumnPreprocessingConfig() for col in X.columns},
        estimator_name="RandomForestRegressor",
        estimator_params={"random_state": 42, "n_estimators": 5},
        constraints=constraints,
        task_type="regression"
    )
    
    orchestrator1 = PipelineOrchestrator(config)
    orchestrator1.fit(X, y)
    pred1 = orchestrator1.predict(X)
    
    orchestrator2 = PipelineOrchestrator(config)
    orchestrator2.fit(X, y)
    pred2 = orchestrator2.predict(X)
    
    np.testing.assert_allclose(pred1, pred2, atol=1e-10)


@given(numeric_data, target_data)
def test_pipeline_serialization_roundtrip(X_np, y_np):
    """Verify pipeline can be pickled/unpickled without loss of state"""
    import tempfile
    import pickle
    
    X = pd.DataFrame(X_np, columns=["feat_0", "feat_1"])
    y = pd.Series(y_np, name="target")
    
    config = PipelineConfig(
        column_configs={col: ColumnPreprocessingConfig() for col in X.columns},
        estimator_name="RandomForestRegressor",
        estimator_params={"n_estimators": 3, "random_state": 0}
    )
    
    orchestrator = PipelineOrchestrator(config)
    orchestrator.fit(X, y)
    original_pred = orchestrator.predict(X)
    
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        with open(f.name, "wb") as pf:
            pickle.dump(orchestrator, pf)
        with open(f.name, "rb") as pf:
            loaded = pickle.load(pf)
    
    loaded_pred = loaded.predict(X)
    np.testing.assert_allclose(original_pred, loaded_pred, atol=1e-10)


def test_cv_strategy_factory_validity():
    """Verify all registered CV strategies produce valid splits"""
    from backend.ml.cv_factory import CVStrategyFactory, CVConfig
    
    X = pd.DataFrame(np.random.rand(100, 3))
    y = pd.Series(np.random.choice([0, 1], 100))
    
    for strategy in CVStrategyFactory.get_available_strategies():
        if strategy in ["loo", "loogroup", "lop", "predefined"]:
            continue  # Skip strategies requiring special setup
        
        config = CVConfig(strategy=strategy, n_splits=3)
        try:
            splitter = CVStrategyFactory.create(config, X=X, y=y)
            splits = list(splitter.split(X, y))
            assert len(splits) >= 2, f"{strategy} should produce >= 2 splits"
            for train_idx, test_idx in splits:
                assert len(train_idx) > 0
                assert len(test_idx) > 0
                assert set(train_idx) & set(test_idx) == set()  # No overlap
        except Exception as e:
            pytest.fail(f"CV strategy {strategy} failed: {e}")
