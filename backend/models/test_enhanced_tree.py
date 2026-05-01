"""
Test script for Enhanced Decision Tree, Rotation Forest, and Tree Kernel extensions.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Import our models
from backend.models.forests.enhanced_tree import EnhancedDecisionTree
from backend.models.forests.rotation_forest import RotationTree, RotationForest
from backend.models.tree_kernels import make_tree_kernel_model


def test_enhanced_tree_regression():
    """Test EnhancedDecisionTree for regression."""
    print("=" * 60)
    print("Test: EnhancedDecisionTree (Regression)")
    print("=" * 60)

    # Load data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Test with different configurations
    configs = [
        {"use_soft_splits": False, "use_honest_tree": False, "use_bernoulli": False},
        {"use_soft_splits": True, "use_honest_tree": False, "use_bernoulli": False},
        {"use_soft_splits": False, "use_honest_tree": True, "use_bernoulli": False},
        {"use_soft_splits": False, "use_honest_tree": False, "use_bernoulli": True, "feature_prob": 0.7},
    ]

    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        try:
            model = EnhancedDecisionTree(
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
                **config
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            print(f"  R2 Score: {score:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!\n")


def test_rotation_forest():
    """Test RotationForest."""
    print("=" * 60)
    print("Test: RotationForest")
    print("=" * 60)

    # Regression test
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    try:
        rf = RotationForest(
            n_estimators=10,
            max_depth=5,
            n_rotation_subsets=3,
            task="regression",
            random_state=42,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        score = r2_score(y_test, y_pred)
        print(f"RotationForest (Regression) R2: {score:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Classification test
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    try:
        rf = RotationForest(
            n_estimators=10,
            max_depth=5,
            n_rotation_subsets=2,
            task="classification",
            n_classes=3,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f"RotationForest (Classification) Accuracy: {score:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone!\n")


def test_tree_kernel_extensions():
    """Test TreeKernel with SVC and SVR (GPR/GPC require Kernel instance)."""
    print("=" * 60)
    print("Test: TreeKernel Extensions (SVC, SVR)")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Test SVC with tree kernel
    print("\n1. SVC with RandomForestKernel (rf):")
    try:
        svc = make_tree_kernel_model(
            model_type="svc",
            kernel_type="rf",
            n_trees=10,
            max_depth=5,
            random_state=42,
        )
        if hasattr(svc, 'kernel') and callable(svc.kernel):
            # Need to fit the kernel first
            svc.kernel.fit(X_train, y_train)
        svc.fit(X_train, y_train)
        score = svc.score(X_test, y_test)
        print(f"   SVC (RF Kernel) Accuracy: {score:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

    # Test SVR with tree kernel (regression)
    print("\n2. SVR with RandomForestKernel (rf) - Regression:")
    try:
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        X_b, y_b = housing.data, housing.target
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_b, y_b, test_size=0.3, random_state=42
        )
        svr = make_tree_kernel_model(
            model_type="svr",
            kernel_type="rf",
            n_trees=10,
            max_depth=5,
            random_state=42,
        )
        if hasattr(svr, 'kernel') and callable(svr.kernel):
            svr.kernel.fit(X_tr, y_tr)
        svr.fit(X_tr, y_tr)
        score = svr.score(X_te, y_te)
        print(f"   SVR (RF Kernel) R2: {score:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

    print("\nNote: GPR and GPC require sklearn.gaussian_process.kernels.Kernel instance.")
    print("For GPR/GPC support, a Kernel wrapper class needs to be implemented.\n")
    print("\nDone!\n")


def test_comparison_with_rf():
    """Compare EnhancedDecisionTree with RandomForest."""
    print("=" * 60)
    print("Comparison: EnhancedDecisionTree vs RandomForest")
    print("=" * 60)

    from sklearn.ensemble import RandomForestRegressor

    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # RandomForest baseline
    try:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_score = r2_score(y_test, rf_pred)
        print(f"\nRandomForest (100 trees) R2: {rf_score:.4f}")
    except Exception as e:
        print(f"Error with RandomForest: {e}")
        rf_score = 0.0

    # EnhancedDecisionTree
    try:
        et = EnhancedDecisionTree(
            max_depth=10,
            min_samples_leaf=5,
            use_soft_splits=True,
            use_honest_tree=True,
            use_bernoulli=True,
            feature_prob=0.7,
            leaf_reg='l2',
            leaf_alpha=0.01,
            random_state=42,
        )
        et.fit(X_train, y_train)
        et_pred = et.predict(X_test)
        et_score = r2_score(y_test, et_pred)
        print(f"EnhancedDecisionTree (single tree) R2: {et_score:.4f}")
        if rf_score > 0:
            print(f"\nRatio (Enhanced / RF): {et_score/rf_score:.2%}")
    except Exception as e:
        print(f"Error with EnhancedDecisionTree: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Enhanced Decision Tree & Kernel Extensions")
    print("=" * 60 + "\n")

    try:
        test_enhanced_tree_regression()
    except Exception as e:
        print(f"Error in EnhancedDecisionTree test: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_rotation_forest()
    except Exception as e:
        print(f"Error in RotationForest test: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_tree_kernel_extensions()
    except Exception as e:
        print(f"Error in TreeKernel extensions test: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_comparison_with_rf()
    except Exception as e:
        print(f"Error in comparison test: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
