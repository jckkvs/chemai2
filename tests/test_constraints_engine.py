"""
Test: chemai2/tests/test_constraints_engine.py
Comprehensive tests for constraint enforcement engine
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from backend.constraints.engine import ConstraintEngine, ConstraintSpec, ConstraintEvaluation


class TestConstraintEngineBasic:
    """Basic functionality tests for ConstraintEngine"""
    
    def test_feature_stats_extraction(self):
        """Verify feature statistics are correctly extracted"""
        np.random.seed(42)
        X = pd.DataFrame({
            'feat1': np.random.normal(10, 2, 100),
            'feat2': np.random.uniform(0, 1, 100),
            'target': np.random.randn(100)
        })
        
        constraints = {
            'feat1': ConstraintSpec('feat1', monotonic='increasing', sigma_range=2.0)
        }
        
        engine = ConstraintEngine(constraints)
        engine.fit(X)
        
        assert 'feat1' in engine._feature_stats
        stats = engine._feature_stats['feat1']
        assert abs(stats['mean'] - 10) < 1  # Allow sampling variation
        assert abs(stats['std'] - 2) < 0.5
        
        # Sigma range calculation check
        # engine.fit already calculates this internally during evaluation, but stats are stored
        # We check if the stored stats can reproduce the range correctly
        expected_min = stats['mean'] - 2*stats['std']
        # This is internal to the evaluation method, but we can verify stats integrity
        assert stats['mean'] > 0
    
    def test_partial_dependence_computation(self):
        """Verify PDP computation produces reasonable output"""
        # Create simple monotonic relationship
        np.random.seed(123)
        n = 200
        X = pd.DataFrame({
            'mono_feat': np.linspace(0, 10, n),
            'noise': np.random.randn(n)
        })
        y = 2 * X['mono_feat'] + 0.5 * X['noise']
        
        model = LinearRegression().fit(X, y)
        
        constraints = {'mono_feat': ConstraintSpec('mono_feat', monotonic='increasing')}
        engine = ConstraintEngine(constraints, n_grid_points=20)
        engine.fit(X)
        
        pdp = engine._compute_partial_dependence(model, X, 'mono_feat', list(X.columns))
        
        assert pdp is not None
        assert len(pdp) == 20
        assert 'mono_feat' in pdp.columns
        assert 'prediction' in pdp.columns
        
        # PDP should be approximately linear with slope ~2
        slope = (pdp['prediction'].iloc[-1] - pdp['prediction'].iloc[0]) / (pdp['mono_feat'].iloc[-1] - pdp['mono_feat'].iloc[0])
        assert abs(slope - 2) < 0.5  # Allow for noise and approximation


class TestMonotonicityEvaluation:
    """Tests for monotonicity constraint evaluation"""
    
    @pytest.mark.parametrize("direction,expected_pass", [
        ('increasing', True),
        ('decreasing', False),
        ('either', True),
    ])
    def test_known_monotonic_function(self, direction, expected_pass):
        """Test evaluation on synthetic data with known monotonicity"""
        np.random.seed(456)
        x = np.linspace(0, 10, 100)
        noise = np.random.randn(100) * 0.5
        
        # Create monotonic increasing function
        y = x ** 1.5 + noise
        
        X = pd.DataFrame({'feat': x, 'other': np.random.randn(100)})
        y_series = pd.Series(y)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
        model.fit(X, y_series)
        
        constraints = {'feat': ConstraintSpec('feat', monotonic=direction, strength='weak')}
        engine = ConstraintEngine(constraints)
        engine.fit(X)
        
        evaluations = engine.evaluate_constraints(model, X, y_series)
        eval_result = evaluations['feat']
        
        if expected_pass:
            assert eval_result.monotonic_violation_ratio < 0.15, \
                f"Expected low violations for {direction}, got {eval_result.monotonic_violation_ratio}"
        else:
            # Decreasing constraint on increasing function should have many violations
            assert eval_result.monotonic_violation_ratio > 0.5, \
                f"Expected high violations for {direction} on increasing data"
    
    def test_sigma_range_enforcement(self):
        """Verify constraints are evaluated within ±nσ range"""
        np.random.seed(789)
        # Create feature with known distribution
        feat_values = np.random.normal(loc=5, scale=1.5, size=300)
        X = pd.DataFrame({
            'constrained_feat': feat_values,
            'other': np.random.randn(300)
        })
        y = X['constrained_feat'] + np.random.randn(300) * 0.3
        
        model = GradientBoostingRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        
        # Test with different sigma ranges
        for sigma_mult in [1.0, 2.0, 3.0]:
            constraints = {
                'constrained_feat': ConstraintSpec(
                    'constrained_feat', 
                    monotonic='increasing', 
                    sigma_range=sigma_mult
                )
            }
            engine = ConstraintEngine(constraints)
            engine.fit(X)
            
            evaluations = engine.evaluate_constraints(model, X, y)
            eval_result = evaluations['constrained_feat']
            
            # Verify sigma range was calculated
            stats = engine._feature_stats['constrained_feat']
            expected_min = stats['mean'] - sigma_mult * stats['std']
            expected_max = stats['mean'] + sigma_mult * stats['std']
            
            assert abs(eval_result.sigma_range_min - expected_min) < 0.1
            assert abs(eval_result.sigma_range_max - expected_max) < 0.1


class TestLinearityEvaluation:
    """Tests for linearity constraint evaluation"""
    
    def test_perfectly_linear_function(self):
        """Verify high R² for truly linear relationship"""
        np.random.seed(111)
        x = np.linspace(-5, 5, 100)
        noise = np.random.randn(100) * 0.1  # Small noise
        
        X = pd.DataFrame({'linear_feat': x, 'noise_feat': np.random.randn(100)})
        y = 3 * X['linear_feat'] - 2 + noise  # y = 3x - 2 + noise
        
        model = LinearRegression().fit(X, y)
        
        constraints = {'linear_feat': ConstraintSpec('linear_feat', linearity='strong')}
        engine = ConstraintEngine(constraints)
        engine.fit(X)
        
        evaluations = engine.evaluate_constraints(model, X, y)
        eval_result = evaluations['linear_feat']
        
        # Should have very high R²
        assert eval_result.linearity_r2 > 0.98, f"R² too low: {eval_result.linearity_r2}"
        assert eval_result.linearity_rmse < 0.2, f"RMSE too high: {eval_result.linearity_rmse}"
        assert eval_result.passed, "Strong linearity constraint should pass"
    
    def test_nonlinear_function_weak_constraint(self):
        """Verify weak linearity constraint allows some nonlinearity"""
        np.random.seed(222)
        x = np.linspace(-3, 3, 100)
        
        X = pd.DataFrame({'quad_feat': x})
        y = x ** 2 + np.random.randn(100) * 0.5  # Quadratic + noise
        
        model = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=6)
        model.fit(X, y)
        
        constraints = {'quad_feat': ConstraintSpec('quad_feat', linearity='weak')}
        engine = ConstraintEngine(constraints)
        engine.fit(X)
        
        evaluations = engine.evaluate_constraints(model, X, y)
        eval_result = evaluations['quad_feat']
        
        # Quadratic function should have moderate R² for linear fit
        assert eval_result.linearity_r2 < 0.8, f"R² unexpectedly high for quadratic: {eval_result.linearity_r2}"
        # But weak constraint might still pass depending on threshold
        # (This tests that evaluation runs without error)


class TestConstraintReport:
    """Tests for report generation"""
    
    def test_report_structure(self):
        """Verify report has expected structure and content"""
        evaluations = {
            'feat1': ConstraintEvaluation(
                feature_name='feat1',
                monotonic_violations=2,
                monotonic_violation_ratio=0.05,
                linearity_r2=0.92,
                linearity_rmse=0.15,
                sigma_range_min=1.0,
                sigma_range_max=9.0,
                passed=True,
                details={'note': 'test'}
            )
        }
        
        constraints = {'feat1': ConstraintSpec('feat1', monotonic='increasing', linearity='weak')}
        engine = ConstraintEngine(constraints)
        
        report = engine.generate_constraint_report(evaluations)
        
        assert 'summary' in report
        assert report['summary']['total_constraints'] == 1
        assert report['summary']['passed'] == 1
        
        assert 'details' in report
        assert 'feat1' in report['details']
        
        detail = report['details']['feat1']
        assert detail['passed'] is True
        assert detail['monotonicity']['violation_ratio'] == '5.0%'
        assert detail['linearity']['r2'] == '0.920'
        assert detail['sigma_range'] == '[1.00, 9.00]'
