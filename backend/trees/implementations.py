"""
backend/trees/implementations.py
"""
from typing import Type
from .base import TreeEnsemble

class DefaultEnsembleImplementation(TreeEnsemble):
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        return [0]*len(X)

def get_implementation_class(base_algorithm: str) -> Type[TreeEnsemble]:
    """
    指定された文字列に基づいて実際の実装クラスを返す。
    新モデルの実装クラスをマッピング。
    """
    # 新モデルの実装クラスマッピング
    implementation_map = {
        "enhancedtree": EnhancedTreeImplementation,
        "enhancedtree_c": EnhancedTreeImplementation,
        "bernoulli_ijcai": BernoulliIJCAIImplementation,
        "softsplit": SoftSplitTreeImplementation,
        "softsplit_c": SoftSplitTreeImplementation,
        "honesttree": HonestTreeImplementation,
        "honesttree_c": HonestTreeImplementation,
        "treekernel_dt": TreeKernelDecisionTreeImplementation,
    }
    return implementation_map.get(base_algorithm, DefaultEnsembleImplementation)


class EnhancedTreeImplementation(TreeEnsemble):
    """EnhancedDecisionTreeの実装。"""
    def __init__(self, **kwargs):
        super().__init__()
        from backend.models.linear_tree import EnhancedDecisionTree
        self.model = EnhancedDecisionTree(**kwargs)

    def fit(self, X, y=None):
        if y is not None:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class BernoulliIJCAIImplementation(TreeEnsemble):
    """BernoulliForestRegressorIJCAIの実装。"""
    def __init__(self, **kwargs):
        super().__init__()
        from backend.models.linear_tree import BernoulliForestRegressorIJCAI
        self.model = BernoulliForestRegressorIJCAI(**kwargs)

    def fit(self, X, y=None):
        if y is not None:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class SoftSplitTreeImplementation(TreeEnsemble):
    """SoftSplitTreeRegressorの実装。"""
    def __init__(self, **kwargs):
        super().__init__()
        from backend.models.linear_tree import SoftSplitTreeRegressor
        self.model = SoftSplitTreeRegressor(**kwargs)

    def fit(self, X, y=None):
        if y is not None:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class HonestTreeImplementation(TreeEnsemble):
    """HonesTreeRegressorの実装。"""
    def __init__(self, **kwargs):
        super().__init__()
        from backend.models.linear_tree import HonesTreeRegressor
        self.model = HonesTreeRegressor(**kwargs)

    def fit(self, X, y=None):
        if y is not None:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class TreeKernelDecisionTreeImplementation(TreeEnsemble):
    """TreeKernelDecisionTreeの実装。"""
    def __init__(self, **kwargs):
        super().__init__()
        from backend.models.tree_kernels import TreeKernelDecisionTree
        self.model = TreeKernelDecisionTree(**kwargs)

    def fit(self, X, y=None):
        if y is not None:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
