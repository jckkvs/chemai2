"""
Script to add TreeKernelRFRExtended and TreeKernelDecisionTree to tree_kernels.py
"""
import sys

code = '''

# ──────────────────────────────────────────────
# 7. Extended RFR Kernel (wraps any tree ensemble)
# ──────────────────────────────────────────────

class TreeKernelRFRExtended:
    """
    Extended RFR-Kernel that can wrap ANY tree-based model.

    Unlike RandomForestKernel which requires a sklearn RandomForest,
    this can wrap any tree ensemble that has:
    - estimators_ attribute with .apply() method
    - Or a get_leaf_indicators() method
    """

    def __init__(
        self,
        ensemble=None,
        n_trees: int = 100,
        max_depth: int = 10,
        random_state: Optional[int] = None,
        **kwargs
    ):
        self.ensemble = ensemble
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X, y=None):
        """Fit ensemble if not already fitted."""
        if self.ensemble is None:
            from sklearn.ensemble import RandomForestRegressor
            self.ensemble = RandomForestRegressor(
                n_estimators=self.n_trees,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.ensemble.fit(X, y)
        elif not hasattr(self.ensemble, 'estimators_'):
            self.ensemble.fit(X, y)
        self._is_fitted = True
        return self

    def __call__(self, X1, X2=None, eval_gradient=False):
        """Compute kernel matrix with optional gradient."""
        if not self._is_fitted:
            raise RuntimeError("Kernel not fitted. Call fit() first.")

        if X2 is None:
            X2 = X1

        # Get leaf indices
        leaves1 = self._get_leaf_indices(X1)
        leaves2 = self._get_leaf_indices(X2) if X2 is not X1 else leaves1

        # Compute kernel matrix
        K = self._compute_cooccurrence(leaves1, leaves2)

        if eval_gradient:
            # For tree kernels, hyperparameters are typically not learned
            # Return zeros with proper shape
            gradient = np.zeros((K.shape[0], K.shape[1], 0))
            return K, gradient

        return K

    def _get_leaf_indices(self, X):
        """Get leaf indices, handling various ensemble types."""
        if hasattr(self.ensemble, 'estimators_'):
            n_samples = X.shape[0]
            n_trees = len(self.ensemble.estimators_)
            leaves = np.zeros((n_samples, n_trees), dtype=int)
            for i, tree in enumerate(self.ensemble.estimators_):
                leaves[:, i] = tree.apply(X)
            return leaves
        elif hasattr(self.ensemble, 'get_leaf_indicators'):
            return self.ensemble.get_leaf_indicators(X)
        else:
            raise ValueError("Ensemble must have estimators_ or get_leaf_indicators()")

    def _compute_cooccurrence(self, leaves1, leaves2):
        """Compute kernel matrix based on leaf co-occurrence."""
        n1 = leaves1.shape[0]
        n2 = leaves2.shape[0]
        n_trees = leaves1.shape[1]

        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                matches = np.sum(leaves1[i] == leaves2[j])
                K[i, j] = matches / n_trees
        return K


# ──────────────────────────────────────────────
# 8. Tree Kernel Decision Tree
# ──────────────────────────────────────────────

class TreeKernelDecisionTree(BaseEstimator, RegressorMixin):
    """
    Decision tree with built-in RFR-Kernel concept.

    Each leaf has a ridge regression model, and the tree itself
    computes kernel similarity based on leaf co-occurrence.

    Can be used as:
    1. Standalone tree regressor
    2. Kernel for KernelRidge, SVR, etc. (via __call__ method)
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 10,
        alpha: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.random_state = random_state

        self.tree_ = None
        self.leaf_models_ = {}
        self._is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Fit base tree
        from sklearn.tree import DecisionTreeRegressor
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree_.fit(X, y)

        # Fit ridge models for each leaf
        leaf_indices = self.tree_.apply(X)
        unique_leaves = np.unique(leaf_indices)

        from sklearn.linear_model import Ridge
        for leaf_id in unique_leaves:
            mask = leaf_indices == leaf_id
            if np.sum(mask) < 2:
                continue
            X_leaf = X[mask]
            y_leaf = y[mask]
            model = Ridge(alpha=self.alpha)
            model.fit(X_leaf, y_leaf)
            self.leaf_models_[leaf_id] = model

        self._is_fitted = True
        return self

    def __call__(self, X1, X2=None, eval_gradient=False):
        """Compute kernel matrix (RFR-Kernel concept)."""
        if not self._is_fitted:
            raise RuntimeError("Kernel not fitted. Call fit() first.")

        if X2 is None:
            X2 = X1

        leaves1 = self.tree_.apply(X1)
        leaves2 = self.tree_.apply(X2) if X2 is not X1 else leaves1

        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                if leaves1[i] == leaves2[j]:
                    K[i, j] = 1.0

        if eval_gradient:
            gradient = np.zeros((n1, n2, 0))
            return K, gradient

        return K

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        leaf_indices = self.tree_.apply(X)
        predictions = np.zeros(len(X))

        for i in range(len(X)):
            leaf_id = leaf_indices[i]
            if leaf_id in self.leaf_models_:
                predictions[i] = self.leaf_models_[leaf_id].predict(X[i].reshape(1, -1))[0]
            else:
                predictions[i] = np.mean(self.tree_.predict(X[i].reshape(1, -1)))

        return predictions

    @property
    def feature_importances_(self):
        if self.tree_ is None:
            raise RuntimeError("Model not fitted")
        return self.tree_.feature_importances_
'''

with open('backend/models/tree_kernels.py', 'a', encoding='utf-8') as f:
    f.write(code)

print('Tree kernel extensions added successfully')
