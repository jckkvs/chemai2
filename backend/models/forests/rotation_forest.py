"""
forests.rotation_forest
===================

Rotation Forest implementation.

Key idea: Apply PCA-based rotation to random feature subsets for each tree.
This introduces additional diversity compared to standard random forests.

References:
Rodriguez, J.J., Kuncheva, L.I., & Alonso, C.J. (2006).
    Rotation Forest: A new classifier ensemble method.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(10), 1619-1630.

Algorithm:
1. Split features into K random subsets
2. For each subset, apply PCA to get rotation matrix
3. Project full feature set using concatenated rotation matrices
4. Train a tree on the rotated features
5. Repeat for each tree in the forest

For a single tree with rotation at each node:
1. At each node, randomly select a subset of features
2. Apply PCA to get rotation matrix
3. Project features and find best split on rotated space
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helper: PCA rotation
# ---------------------------------------------------------------------------

def _compute_rotation(
    X: np.ndarray,
    feature_indices: np.ndarray,
    n_components: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA rotation matrix for given feature subset.

    Returns:
        rotation_matrix: shape (n_features, n_components)
        X_rotated: shape (n_samples, n_components)
    """
    if len(feature_indices) < 2:
        # Cannot do PCA with < 2 features
        n_features = X.shape[1]
        return np.eye(n_features), X

    X_subset = X[:, feature_indices]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    # PCA
    n_components = n_components or min(X_subset.shape[0] - 1, len(feature_indices))
    n_components = min(n_components, len(feature_indices))

    # Compute covariance and eigendecomposition
    cov = X_scaled.T @ X_scaled / (X_subset.shape[0] - 1)
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # Sort by eigenvalue descending
    idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, idx]

    # Take top components
    rotation_subset = eigenvecs[:, :n_components]

    # Map back to full feature space (for consistent dimensions)
    n_features = X.shape[1]
    rotation_full = np.zeros((n_features, n_components))
    rotation_full[feature_indices, :] = rotation_subset

    # Rotate full feature set
    X_rotated = X @ rotation_full

    return rotation_full, X_rotated


# ---------------------------------------------------------------------------
# Rotation Tree (single tree with PCA rotation at each node)
# ---------------------------------------------------------------------------

class RotationTree(BaseEstimator, RegressorMixin):
    """
    Decision tree with PCA rotation at each node.

    Uses sklearn's DecisionTree with rotated features.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        min_samples_split: int = 20,
        n_rotation_subsets: int = 3,
        rotation_components: Optional[int] = None,
        task: str = "regression",  # "regression" or "classification"
        n_classes: int = 1,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_rotation_subsets = n_rotation_subsets
        self.rotation_components = rotation_components
        self.task = task
        self.n_classes = n_classes
        self.random_state = random_state

        self.tree_: Optional[Any] = None
        self.rotation_matrix_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RotationTree":
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        # Create rotation matrix
        n_features = X.shape[1]

        # Split features into K random subsets
        feature_indices = np.arange(n_features)
        rng.shuffle(feature_indices)

        subset_size = max(2, n_features // self.n_rotation_subsets)
        subsets = []
        for i in range(0, n_features, subset_size):
            subsets.append(feature_indices[i:i + subset_size])

        # Compute rotation for each subset
        rotations = []
        for subset in subsets:
            if len(subset) >= 2:
                rot, _ = _compute_rotation(
                    X, subset, self.rotation_components, self.random_state
                )
                rotations.append(rot)

        if rotations:
            self.rotation_matrix_ = np.hstack(rotations)
            X_rotated = X @ self.rotation_matrix_
        else:
            self.rotation_matrix_ = np.eye(n_features)
            X_rotated = X

        # Train tree on rotated features
        if self.task == "classification":
            self.tree_ = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
            self.tree_.fit(X_rotated, y)
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        else:
            self.tree_ = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
            self.tree_.fit(X_rotated, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "tree_")
        X = check_array(X)
        X_rotated = X @ self.rotation_matrix_
        return self.tree_.predict(X_rotated)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score, r2_score
        y_pred = self.predict(X)
        if self.task == "classification":
            return accuracy_score(y, y_pred)
        return r2_score(y, y_pred)


# ---------------------------------------------------------------------------
# Rotation Forest (ensemble of rotation trees)
# ---------------------------------------------------------------------------

class RotationForest(BaseEstimator):
    """
    Rotation Forest ensemble.

    Each tree:
    1. Gets a random fraction of samples (bootstrap)
    2. Features are split into K random subsets
    3. PCA is applied to each subset
    4. A tree is trained on the rotated features
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        min_samples_split: int = 20,
        n_rotation_subsets: int = 3,
        rotation_components: Optional[int] = None,
        task: str = "regression",
        n_classes: int = 1,
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_rotation_subsets = n_rotation_subsets
        self.rotation_components = rotation_components
        self.task = task
        self.n_classes = n_classes
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RotationForest":
        from joblib import Parallel, delayed

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        def _fit_one(seed):
            rng = np.random.default_rng(int(seed))
            if self.bootstrap:
                idx = rng.integers(0, X.shape[0], size=X.shape[0])
                X_s, y_s = X[idx], y[idx]
            else:
                X_s, y_s = X, y

            tree = RotationTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                n_rotation_subsets=self.n_rotation_subsets,
                rotation_components=self.rotation_components,
                task=self.task,
                n_classes=self.n_classes,
                random_state=int(seed),
            )
            tree.fit(X_s, y_s)
            return tree

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(int(s)) for s in seeds
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X)

        if self.task == "classification":
            predictions = np.zeros((X.shape[0], self.n_classes_))
            for tree in self.estimators_:
                pred = tree.predict(X)
                for i, p in enumerate(pred):
                    predictions[i, p] += 1
            return self.classes_[np.argmax(predictions, axis=1)]
        else:
            preds = [tree.predict(X) for tree in self.estimators_]
            return np.mean(preds, axis=0)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score, r2_score
        y_pred = self.predict(X)
        if self.task == "classification":
            return accuracy_score(y, y_pred)
        return r2_score(y, y_pred)
