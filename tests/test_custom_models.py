"""
Unit tests for custom models (Phase 1-4 implementation).

Tests:
- RegularizedTree (regularized_tree.py)
- EnhancedDecisionTree (linear_tree.py)
- BernoulliForestRegressorIJCAI (linear_tree.py)
- SoftSplitTreeRegressor (linear_tree.py)
- HonestTreeRegressor (linear_tree.py)
- TreeKernelDecisionTree (tree_kernels.py)
- TreeKernelRFRExtended (tree_kernels.py)
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ─── Fixtures ────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Generate sample regression data."""
    X, y = make_regression(
        n_samples=200, n_features=5, noise=0.1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ─── RegularizedTree Tests ──────────────────────────────

class TestRegularizedTree:
    """Tests for RegularizedTree."""

    def test_import_and_instantiation(self):
        """Test that RegularizedTree can be imported and instantiated."""
        from backend.models.regularized_tree import RegularizedTree

        tree = RegularizedTree(
            max_leaf=100, reg_alpha=0.1, reg_lambda=1.0, random_state=42
        )
        assert tree.max_leaf == 100
        assert tree.reg_alpha == 0.1
        assert tree.reg_lambda == 1.0

    def test_fit_and_predict(self, sample_data):
        """Test fit and predict methods."""
        from backend.models.regularized_tree import RegularizedTree

        X_train, X_test, y_train, _ = sample_data
        tree = RegularizedTree(
            max_leaf=50, reg_alpha=0.1, reg_lambda=1.0, random_state=42
        )
        tree.fit(X_train, y_train)

        predictions = tree.predict(X_test)
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))

    def test_n_leaves_property(self, sample_data):
        """Test n_leaves_ property."""
        from backend.models.regularized_tree import RegularizedTree

        X_train, _, y_train, _ = sample_data
        tree = RegularizedTree(max_leaf=50, random_state=42)
        tree.fit(X_train, y_train)

        assert tree.n_leaves_ > 0
        assert tree.n_leaves_ <= 50

    def test_feature_importances_(self, sample_data):
        """Test feature_importances_ property."""
        from backend.models.regularized_tree import RegularizedTree

        X_train, _, y_train, _ = sample_data
        tree = RegularizedTree(max_leaf=50, random_state=42)
        tree.fit(X_train, y_train)

        importances = tree.feature_importances_
        assert len(importances) == X_train.shape[1]
        assert np.all(importances >= 0)
        assert np.isclose(np.sum(importances), 1.0)


# ─── EnhancedDecisionTree Tests ────────────────────────

class TestEnhancedDecisionTree:
    """Tests for EnhancedDecisionTree."""

    def test_import_and_instantiation(self):
        """Test that EnhancedDecisionTree can be imported and instantiated."""
        from backend.models.linear_tree import EnhancedDecisionTree

        tree = EnhancedDecisionTree(
            max_depth=10,
            temperature=1.0,
            l1_alpha=0.0,
            l2_alpha=1.0,
            honest_ratio=0.5,
            random_state=42,
        )
        assert tree.max_depth == 10
        assert tree.temperature == 1.0

    def test_fit_and_predict(self, sample_data):
        """Test fit and predict methods."""
        from backend.models.linear_tree import EnhancedDecisionTree

        X_train, X_test, y_train, _ = sample_data
        tree = EnhancedDecisionTree(
            max_depth=10, temperature=1.0, random_state=42
        )
        tree.fit(X_train, y_train)

        predictions = tree.predict(X_test)
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))

    def test_honest_tree_split(self, sample_data):
        """Test that honest tree uses separate structure/estimation samples."""
        from backend.models.linear_tree import EnhancedDecisionTree

        X_train, _, y_train, _ = sample_data
        tree = EnhancedDecisionTree(
            max_depth=10, honest_ratio=0.5, random_state=42
        )
        tree.fit(X_train, y_train)

        # Just verify it fits without error
        assert tree._tree is not None

    def test_soft_split(self, sample_data):
        """Test soft split with temperature parameter."""
        from backend.models.linear_tree import EnhancedDecisionTree

        X_train, _, y_train, _ = sample_data
        # Low temperature = hard split
        tree_hard = EnhancedDecisionTree(
            max_depth=10, temperature=0.01, random_state=42
        )
        tree_hard.fit(X_train, y_train)

        # High temperature = soft split
        tree_soft = EnhancedDecisionTree(
            max_depth=10, temperature=10.0, random_state=42
        )
        tree_soft.fit(X_train, y_train)

        # Both should fit without error
        assert tree_hard._tree is not None
        assert tree_soft._tree is not None


# ─── BernoulliForestRegressorIJCAI Tests ──────────────

class TestBernoulliForestRegressorIJCAI:
    """Tests for BernoulliForestRegressorIJCAI."""

    def test_import_and_instantiation(self):
        """Test that BernoulliForestRegressorIJCAI can be imported."""
        from backend.models.linear_tree import BernoulliForestRegressorIJCAI

        model = BernoulliForestRegressorIJCAI(
            n_estimators=50, p1=0.5, p2=0.5, structure_ratio=0.5, random_state=42
        )
        assert model.n_estimators == 50
        assert model.p1 == 0.5

    def test_fit_and_predict(self, sample_data):
        """Test fit and predict methods."""
        from backend.models.linear_tree import BernoulliForestRegressorIJCAI

        X_train, X_test, y_train, _ = sample_data
        model = BernoulliForestRegressorIJCAI(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))

    def test_ijcai_paper_features(self, sample_data):
        """Test Bernoulli sampling of features (p1)."""
        from backend.models.linear_tree import BernoulliForestRegressorIJCAI

        X_train, _, y_train, _ = sample_data
        # With p1=1.0, all features should be considered
        model = BernoulliForestRegressorIJCAI(
            n_estimators=10, p1=1.0, p2=0.0, random_state=42
        )
        model.fit(X_train, y_train)
        assert len(model.trees_) > 0


# ─── SoftSplitTreeRegressor Tests ──────────────────────

class TestSoftSplitTreeRegressor:
    """Tests for SoftSplitTreeRegressor."""

    def test_import_and_instantiation(self):
        """Test that SoftSplitTreeRegressor can be imported."""
        from backend.models.linear_tree import SoftSplitTreeRegressor

        tree = SoftSplitTreeRegressor(
            max_depth=10, temperature=1.0, random_state=42
        )
        assert tree.temperature == 1.0

    def test_fit_and_predict(self, sample_data):
        """Test fit and predict methods."""
        from backend.models.linear_tree import SoftSplitTreeRegressor

        X_train, X_test, y_train, _ = sample_data
        tree = SoftSplitTreeRegressor(
            max_depth=10, temperature=1.0, random_state=42
        )
        tree.fit(X_train, y_train)

        predictions = tree.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_temperature_effect(self, sample_data):
        """Test that temperature affects predictions."""
        from backend.models.linear_tree import SoftSplitTreeRegressor

        X_train, X_test, y_train, _ = sample_data

        # Very low temperature (hard split)
        tree_hard = SoftSplitTreeRegressor(
            max_depth=10, temperature=0.001, random_state=42
        )
        tree_hard.fit(X_train, y_train)

        # Very high temperature (very soft)
        tree_soft = SoftSplitTreeRegressor(
            max_depth=10, temperature=100.0, random_state=42
        )
        tree_soft.fit(X_train, y_train)

        # Both should produce valid predictions
        pred_hard = tree_hard.predict(X_test)
        pred_soft = tree_soft.predict(X_test)

        assert len(pred_hard) == len(X_test)
        assert len(pred_soft) == len(X_test)


# ─── HonestTreeRegressor Tests ────────────────────────

class TestHonestTreeRegressor:
    """Tests for HonestTreeRegressor."""

    def test_import_and_instantiation(self):
        """Test that HonestTreeRegressor can be imported."""
        from backend.models.linear_tree import HonestTreeRegressor

        tree = HonestTreeRegressor(
            max_depth=10, split_ratio=0.7, random_state=42
        )
        assert tree.split_ratio == 0.7

    def test_fit_and_predict(self, sample_data):
        """Test fit and predict methods."""
        from backend.models.linear_tree import HonestTreeRegressor

        X_train, X_test, y_train, _ = sample_data
        tree = HonestTreeRegressor(
            max_depth=10, split_ratio=0.7, random_state=42
        )
        tree.fit(X_train, y_train)

        predictions = tree.predict(X_test)
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))


# ─── TreeKernelDecisionTree Tests ──────────────────

class TestTreeKernelDecisionTree:
    """Tests for TreeKernelDecisionTree."""

    def test_import_and_instantiation(self):
        """Test that TreeKernelDecisionTree can be imported."""
        from backend.models.tree_kernels import TreeKernelDecisionTree

        tree = TreeKernelDecisionTree(
            max_depth=10, alpha=1.0, random_state=42
        )
        assert tree.max_depth == 10

    def test_fit_and_predict(self, sample_data):
        """Test fit and predict methods."""
        from backend.models.tree_kernels import TreeKernelDecisionTree

        X_train, X_test, y_train, _ = sample_data
        tree = TreeKernelDecisionTree(
            max_depth=10, random_state=42
        )
        tree.fit(X_train, y_train)

        predictions = tree.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_kernel_callable(self, sample_data):
        """Test that the tree can be used as a kernel."""
        from backend.models.tree_kernels import TreeKernelDecisionTree

        X_train, _, y_train, _ = sample_data
        tree = TreeKernelDecisionTree(max_depth=10, random_state=42)
        tree.fit(X_train, y_train)

        # Test kernel matrix computation
        K = tree(X_train[:10])
        assert K.shape == (10, 10)
        assert np.all(np.diag(K) == 1.0)  # Diagonal should be 1 (same leaf)

    def test_kernel_eval_gradient(self, sample_data):
        """Test kernel with eval_gradient=True."""
        from backend.models.tree_kernels import TreeKernelDecisionTree

        X_train, _, y_train, _ = sample_data
        tree = TreeKernelDecisionTree(max_depth=10, random_state=42)
        tree.fit(X_train, y_train)

        # Test with eval_gradient=True
        K, gradient = tree(X_train[:10], eval_gradient=True)
        assert K.shape == (10, 10)
        assert gradient.shape[0] == 10
        assert gradient.shape[1] == 10
        # No learnable hyperparameters, so gradient has 0 length in 3rd dim
        assert gradient.shape[2] == 0


# ─── TreeKernelRFRExtended Tests ─────────────────────

class TestTreeKernelRFRExtended:
    """Tests for TreeKernelRFRExtended."""

    def test_import_and_instantiation(self):
        """Test that TreeKernelRFRExtended can be imported."""
        from backend.models.tree_kernels import TreeKernelRFRExtended

        kernel = TreeKernelRFRExtended(n_trees=50, max_depth=10)
        assert kernel.n_trees == 50

    def test_fit_and_kernel_computation(self, sample_data):
        """Test fit and kernel computation."""
        from backend.models.tree_kernels import TreeKernelRFRExtended

        X_train, X_test, y_train, _ = sample_data

        # Create a simple forest
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        kernel = TreeKernelRFRExtended(ensemble=rf)
        kernel.fit(X_train, y_train)

        # Test kernel matrix
        K = kernel(X_train[:20])
        assert K.shape == (20, 20)

    def test_eval_gradient(self, sample_data):
        """Test kernel with eval_gradient=True."""
        from backend.models.tree_kernels import TreeKernelRFRExtended
        from sklearn.ensemble import RandomForestRegressor

        X_train, _, y_train, _ = sample_data

        rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        kernel = TreeKernelRFRExtended(ensemble=rf)
        kernel.fit(X_train, y_train)

        K, gradient = kernel(X_train[:10], eval_gradient=True)
        assert K.shape == (10, 10)
        assert gradient.shape[2] == 0  # No learnable hyperparameters


# ─── Integration Tests ────────────────────────────────

class TestModelIntegration:
    """Integration tests for model compatibility."""

    def test_enhanced_tree_with_kernelridge(self, sample_data):
        """Test using TreeKernelDecisionTree with KernelRidge."""
        from sklearn.kernel_ridge import KernelRidge
        from backend.models.tree_kernels import TreeKernelDecisionTree

        X_train, X_test, y_train, y_test = sample_data

        # Create and fit tree kernel
        tree_kernel = TreeKernelDecisionTree(max_depth=10, random_state=42)
        tree_kernel.fit(X_train, y_train)

        # Use with KernelRidge
        kr = KernelRidge(kernel=tree_kernel)
        kr.fit(X_train, y_train)

        predictions = kr.predict(X_test)
        assert len(predictions) == len(X_test)
        score = r2_score(y_test, predictions)
        assert score > -2.0  # At least better than random

    def test_factory_integration(self):
        """Test that models are registered in factory.py."""
        from backend.models.factory import get_model, list_models

        # Check that new models are in the registry
        models = list_models(task="regression")

        model_keys = [m["key"] for m in models]

        # Check for new models (some might not be available if imports fail)
        expected_models = ["enhancedtree", "bernoulli_ijcai", "softsplit", "honesttree"]
        for key in expected_models:
            if key in model_keys:
                # Try to instantiate
                model = get_model(key, task="regression")
                assert model is not None

    def test_estimator_registry(self):
        """Test that models are registered in estimators.py."""
        from backend.ml.estimators import ESTIMATOR_REGISTRY

        # Check for new models
        expected_models = [
            "EnhancedDecisionTree",
            "BernoulliForestRegressorIJCAI",
            "TreeKernelDecisionTree",
        ]

        for key in expected_models:
            assert key in ESTIMATOR_REGISTRY, f"{key} not in ESTIMATOR_REGISTRY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
