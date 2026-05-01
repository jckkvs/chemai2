"""
forests.kernel_forest
=====================
Random Kernel Forest.

References
----------
Ustimenko, A., & Prokhorenkova, L. (2022). Random Kernel Forests.
    IEEE Transactions on Neural Networks and Learning Systems.
    DOI: 10.1109/TNNLS.2022.3185709 (IEEE 9837906)

Algorithm
---------
Key innovation: at each node, instead of an axis-aligned split, the
algorithm optimizes an SVM-like objective with a kernel function to find
a quasi-optimal split that maximizes the margin between subtree classes/regions.

This is approximated as:
1. Project X onto a random kernel feature map (e.g., RBF via random Fourier features).
2. Find the optimal hyperplane in the kernel feature space using a linear SVM
   (solved by a fast SMO-style coordinate descent).
3. Use the signed distance from the hyperplane as the split variable.
4. Find the best threshold on this projected value.

Implementation notes:
- We use Random Fourier Features (Rahimi & Recht 2007) to approximate the RBF kernel.
- The SVM-style margin is maximized using a simplified gradient step.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .base import BaseForest, ClassifierForestMixin, RegressorForestMixin, IMPURITY_FN
from .cart import CARTClassifier, CARTRegressor


# ---------------------------------------------------------------------------
# Random Fourier Features (Rahimi & Recht, 2007)
# ---------------------------------------------------------------------------

class _RFFTransform:
    """Approximate RBF kernel via Random Fourier Features.

    φ(x) = sqrt(2/D) * [cos(ω_1^T x + b_1), ..., cos(ω_D^T x + b_D)]

    ||φ(x) - φ(y)||^2 ≈ 2(1 - exp(-||x-y||^2 / (2*gamma^2)))
    """

    def __init__(self, n_components: int, gamma: float, rng: np.random.Generator, n_features: int):
        self.n_components = n_components
        self.gamma = gamma
        # Sample random frequencies from Gaussian (bandwidth = sqrt(2) * gamma)
        self.omega = rng.standard_normal((n_features, n_components)) * np.sqrt(2 * gamma)
        self.bias = rng.uniform(0, 2 * np.pi, n_components)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map X to D-dim feature space."""
        Z = np.cos(X @ self.omega + self.bias)
        return Z * np.sqrt(2.0 / self.n_components)


# ---------------------------------------------------------------------------
# SVM-style split finder in RFF space
# ---------------------------------------------------------------------------

def _kernel_svm_split(
    X: np.ndarray,
    y: np.ndarray,
    impurity_fn,
    min_samples_leaf: int,
    n_classes: Optional[int],
    rng: np.random.Generator,
    n_rff: int,
    gamma: float,
    svm_lambda: float,
    n_iter: int,
) -> Tuple[Optional[np.ndarray], Optional[float], float]:
    """Kernel split via RFF + gradient descent margin maximization.

    Implements the SVM-like loss with margin re-scaling (Ustimenko 2022).

    Returns
    -------
    (rff_transform, threshold, gain)
    where rff_transform is stored in node.extra["kernel_rff"]
    and the 1D projection is X_rff @ w.
    """
    n, p = X.shape
    if n_classes is not None:
        base_imp = impurity_fn(y, n_classes)
        # Binary label for SVM
        classes = np.unique(y)
        if len(classes) < 2:
            return None, None, 0.0
        # Use most frequent vs rest
        counts = np.bincount(y, minlength=max(classes) + 1)
        c0 = counts.argmax()
        label_svm = np.where(y == c0, 1.0, -1.0)
    else:
        base_imp = impurity_fn(y)
        # Regression: binary split by median
        median = np.median(y)
        label_svm = np.where(y >= median, 1.0, -1.0)

    # RFF transform
    rff = _RFFTransform(n_rff, gamma, rng, p)
    Z = rff.transform(X)  # (n, n_rff)

    # Initialize w via random normal
    w = rng.standard_normal(n_rff) * 0.01

    # Gradient descent: hinge loss + L2 regularization
    lr = 0.1 / max(n, 1)
    for _ in range(n_iter):
        margins = label_svm * (Z @ w)
        # Subgradient of hinge loss
        active = margins < 1.0
        grad_loss = -label_svm * active.astype(float)
        grad = Z.T @ grad_loss / n + svm_lambda * w
        w -= lr * grad

    # Project X onto learned direction
    proj = Z @ w

    # Threshold search
    best_gain = 0.0
    best_thr: Optional[float] = None
    uniq = np.unique(proj)
    if len(uniq) < 2:
        return None, None, 0.0
    thresholds = (uniq[:-1] + uniq[1:]) / 2.0
    if len(thresholds) > 20:
        thresholds = rng.choice(thresholds, 20, replace=False)

    for thr in thresholds:
        lm = proj <= thr
        nl, nr = lm.sum(), n - lm.sum()
        if nl < min_samples_leaf or nr < min_samples_leaf:
            continue
        if n_classes is not None:
            imp_l = impurity_fn(y[lm], n_classes)
            imp_r = impurity_fn(y[~lm], n_classes)
        else:
            imp_l = impurity_fn(y[lm])
            imp_r = impurity_fn(y[~lm])
        gain = base_imp - (nl / n) * imp_l - (nr / n) * imp_r
        if gain > best_gain:
            best_gain = gain
            best_thr = float(thr)

    if best_thr is None:
        return None, None, 0.0

    return (rff, w), best_thr, best_gain


# ---------------------------------------------------------------------------
# Kernel tree (base)
# ---------------------------------------------------------------------------

def _build_kernel_tree(is_classifier: bool, base_cls):
    class KernelTree(base_cls):
        def __init__(
            self,
            n_rff: int = 32,
            gamma: float = 1.0,
            svm_lambda: float = 0.01,
            svm_n_iter: int = 20,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.n_rff = n_rff
            self.gamma = gamma
            self.svm_lambda = svm_lambda
            self.svm_n_iter = svm_n_iter

        def _find_best_split(self, X, y, feature_indices, rng, **kwargs):
            fn = IMPURITY_FN[self.criterion]
            nc = self.n_classes_ if is_classifier else None
            return _kernel_svm_split(
                X[:, feature_indices] if True else X,  # use all features for kernel
                y, fn, self.min_samples_leaf, nc, rng,
                self.n_rff, self.gamma, self.svm_lambda, self.svm_n_iter,
            )

        def _build(self, X, y, depth, rng, **kwargs):
            n_samples, n_features = X.shape
            from .base import Node
            impurity = self._impurity(y)
            node = Node(
                value=self._node_value(y),
                impurity=impurity,
                n_samples=n_samples,
                depth=depth,
            )
            too_deep = self.max_depth is not None and depth >= self.max_depth
            too_few = n_samples < self.min_samples_split
            pure = impurity == 0.0
            if too_deep or too_few or pure:
                node.leaf_id = self._leaf_counter
                self._leaf_counter += 1
                return node

            feat_idx = self._select_features(n_features, rng)
            result, thr, gain = self._find_best_split(X, y, feat_idx, rng, **kwargs)
            if result is None or gain < self.min_impurity_decrease:
                node.leaf_id = self._leaf_counter
                self._leaf_counter += 1
                return node

            rff_obj, w = result
            # Project using kernel
            Z = rff_obj.transform(X)
            proj = Z @ w
            mask = proj <= thr

            lX, ly = X[mask], y[mask]
            rX, ry = X[~mask], y[~mask]
            if len(ly) < self.min_samples_leaf or len(ry) < self.min_samples_leaf:
                node.leaf_id = self._leaf_counter
                self._leaf_counter += 1
                return node

            node.extra["kernel_rff"] = rff_obj
            node.extra["kernel_w"] = w
            node.threshold = thr
            node.feature = -1  # kernel split
            node.left = self._build(lX, ly, depth + 1, rng, **kwargs)
            node.right = self._build(rX, ry, depth + 1, rng, **kwargs)
            return node

        def _predict_node(self, x, node):
            if node.is_leaf:
                return node.value
            rff_obj = node.extra.get("kernel_rff")
            if rff_obj is not None:
                Z = rff_obj.transform(x[None, :])
                proj = float(Z @ node.extra["kernel_w"])
            else:
                proj = float(x[node.feature])
            if proj <= node.threshold:
                return self._predict_node(x, node.left)
            return self._predict_node(x, node.right)

        def _apply_node(self, x, node):
            if node.is_leaf:
                return node.leaf_id
            rff_obj = node.extra.get("kernel_rff")
            if rff_obj is not None:
                Z = rff_obj.transform(x[None, :])
                proj = float(Z @ node.extra["kernel_w"])
            else:
                proj = float(x[node.feature])
            if proj <= node.threshold:
                return self._apply_node(x, node.left)
            return self._apply_node(x, node.right)

    return KernelTree


_KernelClassifierTree = _build_kernel_tree(True, CARTClassifier)
_KernelRegressorTree = _build_kernel_tree(False, CARTRegressor)


class RandomKernelForest(ClassifierForestMixin, BaseForest):
    """Random Kernel Forest Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        n_rff: int = 32,
        gamma: float = 1.0,
        svm_lambda: float = 0.01,
        svm_n_iter: int = 20,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        max_samples=None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_rff = n_rff
        self.gamma = gamma
        self.svm_lambda = svm_lambda
        self.svm_n_iter = svm_n_iter
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _make_estimator(self, random_state: int) -> _KernelClassifierTree:
        return _KernelClassifierTree(
            n_rff=self.n_rff,
            gamma=self.gamma,
            svm_lambda=self.svm_lambda,
            svm_n_iter=self.svm_n_iter,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=random_state,
        )

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        super().fit(X, y, **kwargs)
        return self


class RandomKernelForestRegressor(RegressorForestMixin, BaseForest):
    """Random Kernel Forest Regressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        n_rff: int = 32,
        gamma: float = 1.0,
        svm_lambda: float = 0.01,
        svm_n_iter: int = 20,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        max_samples=None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_rff = n_rff
        self.gamma = gamma
        self.svm_lambda = svm_lambda
        self.svm_n_iter = svm_n_iter
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _make_estimator(self, random_state: int) -> _KernelRegressorTree:
        return _KernelRegressorTree(
            n_rff=self.n_rff,
            gamma=self.gamma,
            svm_lambda=self.svm_lambda,
            svm_n_iter=self.svm_n_iter,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=random_state,
        )

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        super().fit(X, y, **kwargs)
        return self


# ---------------------------------------------------------------------------
# Honest Soft Tree Kernel Regressor
# ---------------------------------------------------------------------------
# Combines Honest Tree (data splitting), Soft Splits (sigmoid gates),
# Rotation Forest (QR feature rotation), and Ridge regression on leaf probabilities.
# Reference: forest.txt in 指示 directory.
# ---------------------------------------------------------------------------

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class HonestSoftTreeKernelRegressor:
    """Honest Soft Tree Kernel Regressor.

    Combines:
    1. Honest Tree: split data into structure/estimation parts
    2. Soft Splits: sigmoid gates at each node
    3. Rotation Forest: QR decomposition for feature rotation
    4. Ridge regression on leaf probability matrix

    Parameters
    ----------
    max_depth : int, default=6
        Maximum depth of the decision tree.
    alpha : float, default=1.0
        Regularization strength for Ridge regression.
    temperature : float, default=0.15
        Temperature for sigmoid soft splits (lower = harder splits).
    rotation_seed : int, default=42
        Random seed for rotation matrix generation.
    random_state : int, default=42
        Random seed for data splitting.
    """

    def __init__(
        self,
        max_depth: int = 6,
        alpha: float = 1.0,
        temperature: float = 0.15,
        rotation_seed: int = 42,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.alpha = alpha
        self.temperature = temperature
        self.rotation_seed = rotation_seed
        self.random_state = random_state

        self.tree_ = None
        self.ridge_ = None
        self.scaler_ = StandardScaler()
        self.rotation_matrix_ = None
        self.leaf_map_ = None

    def _rotate_features(self, X: np.ndarray) -> np.ndarray:
        """Rotation Forest extension: QR decomposition for orthogonal transformation."""
        rng = np.random.RandomState(self.rotation_seed)
        n_features = X.shape[1]
        Q, _ = np.linalg.qr(rng.randn(n_features, n_features))
        self.rotation_matrix_ = Q
        return X @ Q

    def _extract_soft_leaf_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute soft leaf probability matrix via sigmoid gates."""
        n_samples = X.shape[0]
        node_count = self.tree_.tree_.node_count
        node_probs = np.zeros((n_samples, node_count))
        node_probs[:, 0] = 1.0

        # Traverse nodes in level order
        queue = [0]
        while queue:
            node = queue.pop(0)
            left_child = self.tree_.tree_.children_left[node]
            if left_child == -1:
                continue  # leaf node

            feat_idx = self.tree_.tree_.feature[node]
            threshold = self.tree_.tree_.threshold[node]
            right_child = self.tree_.tree_.children_right[node]

            # Sigmoid soft split probability
            z = (threshold - X[:, feat_idx]) / self.temperature
            p_left = 1.0 / (1.0 + np.exp(-z))
            p_right = 1.0 - p_left

            node_probs[:, left_child] += node_probs[:, node] * p_left
            node_probs[:, right_child] += node_probs[:, node] * p_right
            queue.extend([left_child, right_child])

        # Extract only leaf nodes
        leaf_indices = np.where(self.tree_.tree_.children_left == -1)[0]
        self.leaf_map_ = {old_idx: i for i, old_idx in enumerate(leaf_indices)}
        Phi = np.zeros((n_samples, len(leaf_indices)))
        for old_idx, new_idx in self.leaf_map_.items():
            Phi[:, new_idx] = node_probs[:, old_idx]

        # Normalize rows to sum to 1 (numerical stability)
        row_sums = Phi.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return Phi / row_sums

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        # 1. Honest Tree: split data for structure/estimation
        X_honest, y_honest = X.copy(), y.copy()
        X_split, X_pred, y_split, y_pred = train_test_split(
            X_honest, y_honest, test_size=0.5, random_state=self.random_state
        )

        # 2. Standardize and rotate features
        X_split_scaled = self.scaler_.fit_transform(X_split)
        X_split_rot = self._rotate_features(X_split_scaled)

        # 3. Fit structure tree on split data
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth, random_state=self.random_state
        )
        self.tree_.fit(X_split_rot, y_split)

        # 4. Build soft leaf matrix on prediction data
        X_pred_scaled = self.scaler_.transform(X_pred)
        X_pred_rot = X_pred_scaled @ self.rotation_matrix_
        Phi_pred = self._extract_soft_leaf_matrix(X_pred_rot)

        # 5. Fit Ridge regression on leaf probabilities
        self.ridge_ = Ridge(alpha=self.alpha)
        self.ridge_.fit(Phi_pred, y_pred)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X_scaled = self.scaler_.transform(X)
        X_rot = X_scaled @ self.rotation_matrix_
        Phi = self._extract_soft_leaf_matrix(X_rot)
        return self.ridge_.predict(Phi)
