"""
Cross-Validation Strategy Factory - chemai2/backend/ml/cv_factory.py
Factory for all sklearn cross-validation strategies with auto-detection
"""
from typing import Dict, List, Optional, Union, Any, Literal, Type, Iterator, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    # Basic splitters
    KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit,
    # Leave-one-out variants
    LeaveOneOut, LeaveOneGroupOut, LeavePGroupsOut,
    # Group-based
    GroupKFold, GroupShuffleSplit,
    # Time series
    TimeSeriesSplit,
    # Predefined
    PredefinedSplit,
    # Custom
    BaseCrossValidator, BaseShuffleSplit
)

from backend.utils.logger import logger


@dataclass
class CVConfig:
    """Configuration for cross-validation strategy"""
    # Strategy selection
    strategy: Literal[
        'kfold', 'stratified_kfold', 'shuffle', 'stratified_shuffle',
        'loo', 'loogroup', 'lop', 'group_kfold', 'group_shuffle',
        'timeseries', 'predefined', 'custom'
    ] = 'kfold'
    
    # Common parameters
    n_splits: int = 5
    random_state: Optional[int] = 42
    shuffle: bool = True
    
    # Stratified parameters
    stratify_column: Optional[str] = None  # Column name for stratification
    
    # Group parameters
    group_column: Optional[str] = None  # Column name for groups
    n_groups: Optional[int] = None  # For LeavePGroupsOut
    
    # Time series parameters
    gap: int = 0
    max_train_size: Optional[int] = None
    test_size: Optional[Union[int, float]] = None
    
    # Predefined split
    test_fold: Optional[np.ndarray] = None  # Array of fold assignments
    
    # Custom splitter (advanced)
    custom_class: Optional[Type[BaseCrossValidator]] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    name: str = 'default_cv'
    description: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CVConfig':
        """Deserialize from dict"""
        return cls(**data)


class CVStrategyFactory:
    """Factory for creating sklearn cross-validation splitters"""
    
    _registry: Dict[str, Type[Union[BaseCrossValidator, BaseShuffleSplit]]] = {
        'kfold': KFold,
        'stratified_kfold': StratifiedKFold,
        'shuffle': ShuffleSplit,
        'stratified_shuffle': StratifiedShuffleSplit,
        'loo': LeaveOneOut,
        'loogroup': LeaveOneGroupOut,
        'lop': LeavePGroupsOut,
        'group_kfold': GroupKFold,
        'group_shuffle': GroupShuffleSplit,
        'timeseries': TimeSeriesSplit,
        'predefined': PredefinedSplit,
    }
    
    @classmethod
    def create(cls, config: CVConfig, X: pd.DataFrame = None, y: pd.Series = None, 
               groups: pd.Series = None) -> Union[BaseCrossValidator, BaseShuffleSplit]:
        """
        Create CV splitter from config with optional data for auto-configuration
        
        Args:
            config: CV configuration
            X: Feature DataFrame (for auto-detection)
            y: Target Series (for stratification)
            groups: Group labels Series (for group-based CV)
        
        Returns:
            Configured sklearn CV splitter instance
        """
        strategy = config.strategy
        splitter_class = cls._registry.get(strategy)
        
        if not splitter_class:
            if config.custom_class:
                return config.custom_class(**config.custom_params)
            raise ValueError(f"Unknown CV strategy: {strategy}")
        
        # Prepare kwargs based on strategy
        kwargs = cls._prepare_kwargs(config, X, y, groups)
        
        try:
            return splitter_class(**kwargs)
        except TypeError as e:
            logger.warning(f"Failed to create {strategy} with kwargs {kwargs}: {e}")
            # Fallback to minimal config
            minimal_kwargs = {'n_splits': config.n_splits} if 'n_splits' in splitter_class.__init__.__code__.co_varnames else {}
            if config.random_state is not None and 'random_state' in splitter_class.__init__.__code__.co_varnames:
                minimal_kwargs['random_state'] = config.random_state
            return splitter_class(**minimal_kwargs)
    
    @classmethod
    def _prepare_kwargs(cls, config: CVConfig, X: pd.DataFrame, y: pd.Series, 
                       groups: pd.Series) -> Dict[str, Any]:
        """Prepare keyword arguments for splitter constructor"""
        kwargs = {}
        strategy = config.strategy
        
        # Common parameters
        if 'n_splits' in cls._get_init_params(config.strategy):
            kwargs['n_splits'] = config.n_splits
        if 'random_state' in cls._get_init_params(config.strategy) and config.random_state is not None:
            kwargs['random_state'] = config.random_state
        if 'shuffle' in cls._get_init_params(config.strategy):
            kwargs['shuffle'] = config.shuffle
        
        # Stratified strategies
        if 'stratified' in strategy and y is not None:
            # StratifiedKFold/ShuffleSplit don't take y in constructor
            # Stratification is applied during split() call
            pass
        
        # Group-based strategies
        if 'group' in strategy or strategy in ['loogroup', 'lop']:
            if groups is not None:
                kwargs['n_splits'] = config.n_groups if strategy == 'lop' else config.n_splits
            # groups is passed to split() method, not constructor
        
        # LeavePGroupsOut
        if strategy == 'lop':
            kwargs['p'] = config.n_groups or 1
        
        # Time series
        if strategy == 'timeseries':
            if config.gap > 0:
                kwargs['gap'] = config.gap
            if config.max_train_size:
                kwargs['max_train_size'] = config.max_train_size
            if config.test_size:
                kwargs['test_size'] = config.test_size
        
        # Predefined split
        if strategy == 'predefined' and config.test_fold is not None:
            kwargs['test_fold'] = config.test_fold
        
        # Custom params
        if config.custom_params:
            kwargs.update(config.custom_params)
        
        return kwargs
    
    @classmethod
    def _get_init_params(cls, strategy: str) -> List[str]:
        """Get constructor parameter names for a strategy"""
        import inspect
        splitter_class = cls._registry.get(strategy)
        if not splitter_class:
            return []
        sig = inspect.signature(splitter_class.__init__)
        return list(sig.parameters.keys())
    
    @classmethod
    def auto_detect_strategy(cls, X: pd.DataFrame, y: pd.Series,
                            groups: pd.Series = None,
                            task_type: str = 'regression') -> str:
        """
        Auto-detect appropriate CV strategy based on data characteristics

        Logic:
        - Classification + imbalanced y -> stratified_kfold
        - Time series data (datetime index) -> timeseries
        - Group labels provided -> group_kfold
        - Small dataset (<100 samples) -> loo
        - Default -> kfold
        """
        n_samples = len(y)

        # Time series detection
        if isinstance(X.index, pd.DatetimeIndex) or (hasattr(X, 'columns') and any('date' in str(c).lower() or 'time' in str(c).lower() for c in X.columns)):
            return 'timeseries'

        # Group-based detection
        if groups is not None and groups.nunique() > 1:
            return 'group_kfold'

        # Stratified for classification with imbalanced classes
        if task_type == 'classification' and y.nunique() >= 2:
            class_counts = y.value_counts()
            if (class_counts / len(y)).min() < 0.3:  # Imbalanced
                return 'stratified_kfold'

        # Leave-one-out for very small datasets
        if n_samples < 100:
            return 'loo' if n_samples < 50 else 'kfold'

        return 'kfold'

    @classmethod
    def recommend_strategy(cls, X: pd.DataFrame, y: pd.Series,
                           groups: pd.Series = None,
                           task_type: str = 'regression') -> dict:
        """
        Recommend CV strategy with detailed reason in Japanese.

        Returns:
            {
                "strategy": str,
                "reason": str,
                "confidence": "high" | "medium" | "low",
                "alternatives": list[str],
            }
        """
        n_samples = len(y)
        reasons = []
        alternatives = []
        confidence = "high"

        # Time series detection
        has_time_col = (
            isinstance(X.index, pd.DatetimeIndex) or
            any('date' in str(c).lower() or 'time' in str(c).lower() for c in X.columns)
        )
        if has_time_col:
            reasons.append(
                "📅 時間列またはDatetimeIndexが検出されました。"
                "時系列データでは未来のデータが過去の学習に使われないよう、"
                "TimeSeriesSplitを使用すべきです。"
            )
            return {
                "strategy": "timeseries",
                "reason": " ".join(reasons),
                "confidence": "high",
                "alternatives": ["kfold", "shuffle"],
            }

        # Group-based detection
        if groups is not None and groups.nunique() > 1:
            n_groups = groups.nunique()
            reasons.append(
                f"👥 グループ列が検出されました（{n_groups}グループ）。"
                "同じグループのデータが学習・検証に分割されないよう、"
                "GroupKFoldを使用すべきです。"
            )
            if n_groups <= n_samples:
                alternatives.append("loogroup")
            return {
                "strategy": "group_kfold",
                "reason": " ".join(reasons),
                "confidence": "high",
                "alternatives": alternatives or ["stratified_kfold", "kfold"],
            }

        # Classification with class imbalance
        if task_type == 'classification' and y.nunique() >= 2:
            class_counts = y.value_counts(normalize=True)
            min_ratio = class_counts.min()
            if min_ratio < 0.2:
                reasons.append(
                    f"⚖️ クラスが極めて不均衡です（最小クラス: {min_ratio:.1%}）。"
                    "各foldでクラス比率を維持するStratifiedKFoldを推奨します。"
                )
                return {
                    "strategy": "stratified_kfold",
                    "reason": " ".join(reasons),
                    "confidence": "high",
                    "alternatives": ["group_kfold", "kfold"],
                }
            elif min_ratio < 0.3:
                reasons.append(
                    f"📊 クラスにやや不均衡が見られます（最小クラス: {min_ratio:.1%}）。"
                    "StratifiedKFoldでクラス比率を維持することを推奨します。"
                )
                return {
                    "strategy": "stratified_kfold",
                    "reason": " ".join(reasons),
                    "confidence": "medium",
                    "alternatives": ["kfold", "shuffle"],
                }

        # Small dataset
        if n_samples < 30:
            reasons.append(
                f"📉 サンプル数が非常に少ないです（{n_samples}件）。"
                "LeaveOneOutですべてのデータを最大限活用できます。"
            )
            return {
                "strategy": "loo",
                "reason": " ".join(reasons),
                "confidence": "high",
                "alternatives": ["kfold"],
            }
        elif n_samples < 100:
            reasons.append(
                f"📊 サンプル数が少ないです（{n_samples}件）。"
                "KFold（5分割）で安定した評価を行います。"
            )
            return {
                "strategy": "kfold",
                "reason": " ".join(reasons),
                "confidence": "medium",
                "alternatives": ["loo", "shuffle"],
            }

        # Default for regression
        if task_type == 'regression':
            reasons.append(
                f"🔄 回帰タスク、サンプル数{n_samples}件です。"
                "KFoldで十分な評価が可能です。"
            )
        else:
            reasons.append(
                f"🔄 分類タスク、サンプル数{n_samples}件です。"
                "StratifiedKFoldでクラス比率を維持します。"
            )

        return {
            "strategy": "stratified_kfold" if task_type == 'classification' else "kfold",
            "reason": " ".join(reasons),
            "confidence": "medium",
            "alternatives": ["shuffle", "stratified_shuffle"],
        }
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """List all available CV strategies"""
        return list(cls._registry.keys())


@dataclass
class CVResults:
    """Container for cross-validation results with metadata"""
    scores: List[float]
    train_scores: Optional[List[float]] = None
    fit_times: Optional[List[float]] = None
    predict_times: Optional[List[float]] = None
    split_info: Optional[List[Dict[str, int]]] = None  # train_size, test_size per fold
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mean_score(self) -> float:
        return np.mean(self.scores)
    
    @property
    def std_score(self) -> float:
        return np.std(self.scores)
    
    @property
    def ci_95(self) -> Tuple[float, float]:
        """95% confidence interval"""
        if len(self.scores) < 2:
            return self.mean_score, self.mean_score
        from scipy import stats
        sem = stats.sem(self.scores)
        ci = stats.t.interval(0.95, len(self.scores)-1, loc=self.mean_score, scale=sem)
        return tuple(ci)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize results"""
        return {
            'scores': self.scores,
            'mean': self.mean_score,
            'std': self.std_score,
            'ci_95': self.ci_95,
            'n_folds': len(self.scores),
            'metadata': self.metadata
        }


def run_cross_validation(
    estimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv_config: CVConfig,
    groups: pd.Series = None,
    scoring: str = None,
    return_train_score: bool = False,
    **fit_params
) -> CVResults:
    """
    Run cross-validation with automatic strategy handling
    
    Returns CVResults object with scores and metadata
    """
    from sklearn.model_selection import cross_validate
    
    # Create splitter
    splitter = CVStrategyFactory.create(cv_config, 
                                       X if isinstance(X, pd.DataFrame) else None,
                                       y if isinstance(y, pd.Series) else None,
                                       groups)
    
    # Determine scoring metric
    if scoring is None:
        scoring = 'r2' if hasattr(estimator, 'predict') and not hasattr(estimator, 'predict_proba') else 'accuracy'
    
    # Run cross-validation
    results = cross_validate(
        estimator, X, y,
        cv=splitter,
        scoring=scoring,
        return_train_score=return_train_score,
        return_estimator=False,
        **fit_params
    )
    
    # Collect split info
    split_info = []
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        split_info.append({
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
    
    return CVResults(
        scores=results[f'test_{scoring}'].tolist(),
        train_scores=results[f'train_{scoring}'].tolist() if return_train_score else None,
        fit_times=results['fit_time'].tolist(),
        predict_times=results['score_time'].tolist(),
        split_info=split_info,
        metadata={
            'strategy': cv_config.strategy,
            'n_splits': cv_config.n_splits,
            'scoring': scoring,
            'estimator': type(estimator).__name__
        }
    )
