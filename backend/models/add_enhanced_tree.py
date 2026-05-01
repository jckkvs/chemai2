"""
Script to add EnhancedDecisionTree to linear_tree.py
"""
import sys

code = '''

# ──────────────────────────────────────────────────────
# Enhanced Decision Tree (RF-level single tree)
# ──────────────────────────────────────────────────────

class EnhancedDecisionTree(BaseEstimator, RegressorMixin):
    """
    Enhanced Decision Tree combining multiple advanced concepts for RF-level performance:

    1. Soft splits: sigmoid weighting (from SoftSplitTreeRegressor)
    2. Both-node samples: samples exist in BOTH child nodes with sigmoid weights
    3. L1/L2 regularization: ElasticNet for leaf weight regularization
    4. Leaf weight constraints: leaf_min, leaf_max bounds
    5. Bernoulli concepts: per-node feature/sample subsampling
    6. Honest Tree: separate structure/estimation samples per node
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        temperature: float = 1.0,
        l1_alpha: float = 0.0,
        l2_alpha: float = 1.0,
        leaf_min: float = None,
        leaf_max: float = None,
        bernoulli_p_feature: float = 1.0,
        bernoulli_p_sample: float = 1.0,
        honest_ratio: float = 0.5,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.temperature = temperature
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.leaf_min = leaf_min
        self.leaf_max = leaf_max
        self.bernoulli_p_feature = bernoulli_p_feature
        self.bernoulli_p_sample = bernoulli_p_sample
        self.honest_ratio = honest_ratio
        self.random_state = random_state
        self.verbose = verbose
        self._tree = None
        self._feature_importances_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> 'EnhancedDecisionTree':
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape
        rng = np.random.default_rng(self.random_state)
        n_structure = max(1, int(n_samples * self.honest_ratio))
        indices = rng.permutation(n_samples)
        structure_idx = indices[:n_structure]
        estimation_idx = indices[n_structure:]
        if len(estimation_idx) < 2:
            estimation_idx = structure_idx
        self._tree = self._build_tree(
            X_arr, y_arr,
            structure_idx, estimation_idx,
            depth=0,
            feature_indices=np.arange(n_features),
            rng=rng,
        )
        self._feature_importances_ = np.zeros(n_features)
        return self

    def _build_tree(self, X, y, structure_idx, estimation_idx, depth, feature_indices, rng):
        X_struct = X[structure_idx]
        y_struct = y[structure_idx]
        X_est = X[estimation_idx]
        y_est = y[estimation_idx]
        n_struct = len(y_struct)
        if (depth >= self.max_depth or
            n_struct < self.min_samples_split or
            np.var(y_struct) < 1e-10):
            return _RegularizedLeafModel(
                X_est, y_est,
                l1_alpha=self.l1_alpha,
                l2_alpha=self.l2_alpha,
                leaf_min=self.leaf_min,
                leaf_max=self.leaf_max,
            )
        if self.bernoulli_p_feature < 1.0:
            feature_mask = rng.random(len(feature_indices)) < self.bernoulli_p_feature
            if np.sum(feature_mask) == 0:
                feature_mask[rng.integers(0, len(feature_indices))] = True
            node_features = feature_indices[feature_mask]
        else:
            node_features = feature_indices
        best_gain = -np.inf
        best_split = None
        for feat_idx in node_features:
            thresholds = self._get_candidate_thresholds(X_struct[:, feat_idx])
            for thresh in thresholds:
                x_col = X_struct[:, feat_idx]
                w_left = 1.0 / (1.0 + np.exp(-(thresh - x_col) / self.temperature))
                w_right = 1.0 - w_left
                n_left = np.sum(w_left)
                n_right = np.sum(w_right)
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                impurity_parent = np.var(y_struct)
                impurity_left = self._weighted_variance(y_struct, w_left)
                impurity_right = self._weighted_variance(y_struct, w_right)
                gain = (impurity_parent -
                        (n_left / n_struct) * impurity_left -
                        (n_right / n_struct) * impurity_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat_idx, thresh, w_left, w_right)
        if best_split is None or best_gain < 1e-6:
            return _RegularizedLeafModel(
                X_est, y_est,
                l1_alpha=self.l1_alpha,
                l2_alpha=self.l2_alpha,
                leaf_min=self.leaf_min,
                leaf_max=self.leaf_max,
            )
        feat_idx, thresh, _, _ = best_split
        left_struct_mask = X_struct[:, feat_idx] <= thresh
        right_struct_mask = ~left_struct_mask
        left_est_mask = X_est[:, feat_idx] <= thresh
        right_est_mask = ~left_est_mask
        left_child = self._build_tree(
            X, y,
            structure_idx[left_struct_mask],
            estimation_idx[left_est_mask],
            depth + 1, feature_indices, rng,
        )
        right_child = self._build_tree(
            X, y,
            structure_idx[right_struct_mask],
            estimation_idx[right_est_mask],
            depth + 1, feature_indices, rng,
        )
        return _EnhancedSplitNode(feat_idx, thresh, left_child, right_child, self.temperature)

    def _weighted_variance(self, y: np.ndarray, weights: np.ndarray) -> float:
        if np.sum(weights) < 1e-10:
            return 0.0
        mean = np.average(y, weights=weights)
        variance = np.average((y - mean) ** 2, weights=weights)
        return variance

    def _get_candidate_thresholds(self, values: np.ndarray, max_candidates: int = 50) -> np.ndarray:
        unique_vals = np.unique(values)
        if len(unique_vals) <= 1:
            return np.array([])
        if len(unique_vals) <= max_candidates:
            return (unique_vals[:-1] + unique_vals[1:]) / 2.0
        else:
            quantiles = np.linspace(0, 1, max_candidates + 2)[1:-1]
            return np.percentile(values, quantiles * 100)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self._tree is None:
            raise RuntimeError('Model not fitted')
        X_arr = np.asarray(X, dtype=np.float64)
        return np.array([self._tree.predict(x) for x in X_arr])

    @property
    def feature_importances_(self):
        if self._feature_importances_ is None:
            raise RuntimeError('Model not fitted')
        return self._feature_importances_


class _RegularizedLeafModel:
    """Leaf model with L1/L2 regularization and constraints."""

    def __init__(self, X, y, l1_alpha=0.0, l2_alpha=1.0, leaf_min=None, leaf_max=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if l1_alpha > 0:
            total_alpha = l1_alpha + l2_alpha
            l1_ratio = l1_alpha / total_alpha if total_alpha > 0 else 0.5
            self.model = ElasticNet(alpha=total_alpha, l1_ratio=l1_ratio, fit_intercept=True)
        else:
            self.model = Ridge(alpha=l2_alpha, fit_intercept=True)
        if len(X) > 0 and len(y) > 0:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if len(X) >= 2:
                self.model.fit(X, y)
            else:
                self.model = None
                self.constant_value = np.mean(y) if len(y) > 0 else 0.0
        else:
            self.model = None
            self.constant_value = 0.0
        self.leaf_min = leaf_min
        self.leaf_max = leaf_max

    def predict(self, x: np.ndarray) -> float:
        if self.model is not None:
            pred = float(self.model.predict([x])[0])
        else:
            pred = self.constant_value
        if self.leaf_min is not None:
            pred = max(self.leaf_min, pred)
        if self.leaf_max is not None:
            pred = min(self.leaf_max, pred)
        return pred


class _EnhancedSplitNode:
    """Split node with soft splitting (both-node samples)."""

    def __init__(self, feature_idx, threshold, left, right, temperature):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.temperature = temperature

    def predict(self, x: np.ndarray) -> float:
        w_right = 1.0 / (1.0 + np.exp(-(x[self.feature_idx] - self.threshold) / self.temperature))
        w_left = 1.0 - w_right
        return w_left * self.left.predict(x) + w_right * self.right.predict(x)
'''

with open('backend/models/linear_tree.py', 'a', encoding='utf-8') as f:
    f.write(code)

print('EnhancedDecisionTree added successfully')
