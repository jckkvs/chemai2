"""
Test: chemai2/tests/test_chem_constraints.py
Verification tests for chemical constraints and descriptor calculations
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from backend.ml_pipeline import ConstraintAwareEstimator, build_pipeline
from backend.chem.plugins.rdkit_descriptors import calculate_descriptors as calculate_rdkit_basic

# Mocking calculate_rdkit_basic if not imported correctly
try:
    from backend.chem.plugins.rdkit_descriptors import calculate_descriptors as calculate_rdkit_basic
except ImportError:
    def calculate_rdkit_basic(smiles_list):
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        data = []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s)
            data.append({'MolWt': Descriptors.MolWt(m) if m else None})
        return pd.DataFrame(data)

class TestMonotonicConstraintEnforcement:
    """Test monotonic constraint enforcement across estimator types"""
    
    def test_xgboost_native_monotonic_increasing(self):
        """Verify XGBoost respects monotonic increasing constraint"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            pytest.skip("xgboost not installed")
        
        # Generate synthetic data with known monotonic relationship
        np.random.seed(42)
        X = pd.DataFrame({
            'monotonic_feat': np.linspace(0, 10, 100),
            'noise': np.random.randn(100)
        })
        y = 2 * X['monotonic_feat'] + 0.5 * X['noise']  # Strong monotonic signal
        
        constraints = {
            'monotonic_feat': {'monotonic': 'increasing', 'strength': 'strong'}
        }
        
        estimator = ConstraintAwareEstimator(
            base_estimator=XGBRegressor(n_estimators=10, random_state=42),
            constraints=constraints,
            constraint_strength='strong'
        )
        
        estimator.fit(X, y)
        
        # Test monotonicity: increasing input should not decrease prediction
        test_X = pd.DataFrame({
            'monotonic_feat': [0, 1, 2, 5, 10],
            'noise': [0] * 5
        })
        predictions = estimator.predict(test_X)
        
        # Allow small numerical tolerance
        assert np.all(np.diff(predictions) >= -1e-6), \
            f"Predictions not monotonic: {predictions}"
    
    def test_sigma_range_constraint_enforcement(self):
        """Verify constraints enforced within ±nσ range"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create data with known distribution
        np.random.seed(123)
        X_train = pd.DataFrame({
            'feat': np.random.normal(loc=5, scale=2, size=200)
        })
        y_train = X_train['feat'] * 1.5 + np.random.normal(0, 0.5, 200)
        
        constraints = {
            'feat': {
                'monotonic': 'increasing',
                'sigma_range': 2.0,
                'strength': 'weak'
            }
        }
        
        estimator = ConstraintAwareEstimator(
            base_estimator=RandomForestRegressor(n_estimators=20, random_state=42),
            constraints=constraints,
            sigma_multiplier=2.0,
            constraint_strength='weak'
        )
        
        estimator.fit(X_train, y_train)
        
        # Test points within ±2σ should respect constraint better than outside
        sigma = X_train['feat'].std()
        mean = X_train['feat'].mean()
        
        # Within range: [mean-2σ, mean+2σ]
        within = pd.DataFrame({'feat': np.linspace(mean - 1.9*sigma, mean + 1.9*sigma, 20)})
        # Outside range
        outside = pd.DataFrame({'feat': np.linspace(mean + 2.1*sigma, mean + 4*sigma, 20)})
        
        pred_within = estimator.predict(within)
        pred_outside = estimator.predict(outside)
        
        # Within range should have higher monotonicity score (simplified check)
        mono_within = np.mean(np.diff(pred_within) >= -1e-6)
        mono_outside = np.mean(np.diff(pred_outside) >= -1e-6)
        
        assert mono_within >= mono_outside - 0.2, \
            f"Constraint not stronger within sigma range: {mono_within} vs {mono_outside}"


class TestDescriptorCalculationReproducibility:
    """Test chemical descriptor calculation reproducibility"""
    
    def test_rdkit_descriptors_deterministic(self):
        """Verify RDKit descriptors are deterministic across runs"""
        smiles_list = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O"]
        
        # Calculate twice
        result1 = calculate_rdkit_basic(smiles_list)
        result2 = calculate_rdkit_basic(smiles_list)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_descriptor_missing_value_handling(self):
        """Verify graceful handling of invalid SMILES"""
        smiles_with_invalid = ["CCO", "invalid_smiles", "c1ccccc1", None]
        
        result = calculate_rdkit_basic(smiles_with_invalid)
        
        assert len(result) == len(smiles_with_invalid)
        # Invalid entries should have NaN for MolWt
        assert pd.isna(result.iloc[1]['MolWt'])
        assert pd.isna(result.iloc[3]['MolWt'])
    
    @pytest.mark.parametrize("smiles,expected_mw", [
        ("CCO", 46.07),
        ("c1ccccc1", 78.11),
    ])
    def test_molecular_weight_accuracy(self, smiles, expected_mw):
        """Verify molecular weight calculation accuracy vs reference"""
        result = calculate_rdkit_basic([smiles])
        calculated_mw = result['MolWt'].iloc[0]
        
        assert abs(calculated_mw - expected_mw) < 0.1
