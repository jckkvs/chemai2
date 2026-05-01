"""
ML Pipeline Module - chemai2/backend/ml_pipeline.py
5-stage pipeline with constraint support and dynamic UI metadata
"""
import inspect
import json
from typing import Dict, List, Optional, Any, Union, Literal, get_type_hints
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from backend.ml.transformers import AdaptiveQuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectFromModel, RFE, SelectKBest, f_regression, f_classif,
    mutual_info_regression, mutual_info_classif
)
from pydantic import BaseModel, Field

from backend.core.config import settings
from backend.utils.logger import logger


class VariableType(Enum):
    """Detected variable types for automatic preprocessing"""
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    BINARY = "binary"
    CATEGORICAL_LOW = "categorical_low"
    CATEGORICAL_HIGH = "categorical_high"
    TEXT = "text"
    DATETIME = "datetime"


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Custom transformer for correlation-based feature selection"""
    def __init__(self, k: int = 10, task_type: str = 'regression'):
        self.k = k
        self.task_type = task_type
        self.selected_features_ = None

    def fit(self, X, y=None):
        if y is None: return self
        X_df = pd.DataFrame(X)
        y_ser = pd.Series(y)
        corrs = X_df.apply(lambda col: abs(col.corr(y_ser)))
        self.selected_features_ = corrs.sort_values(ascending=False).head(self.k).index.tolist()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df[self.selected_features_]


@dataclass
class PreprocessingConfig:
    """Configuration for column-wise preprocessing"""
    imputer_strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'mean'
    imputer_fill_value: Optional[Any] = None
    scaler: Literal['standard', 'minmax', 'robust', 'quantile_uniform', 'quantile_normal', 'power_yeojohnson', 'power_boxcox', 'none'] = 'standard'
    power_transform_lambda: Optional[float] = None
    encode_categorical: Literal['onehot', 'ordinal', 'target', 'none'] = 'onehot'
    max_categories: int = 50
    
    def to_transformer(self, columns: List[str], task_type: Literal['regression', 'classification']) -> List[tuple]:
        """Convert config to sklearn ColumnTransformer steps"""
        transformers = []
        
        # Imputation
        imputer = SimpleImputer(strategy=self.imputer_strategy, fill_value=self.imputer_fill_value)
        transformers.append(('imputer', imputer, columns))
        
        # Scaling/Transformation
        if self.scaler == 'standard':
            scaler = StandardScaler()
        elif self.scaler == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif self.scaler == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        elif self.scaler == 'quantile_uniform':
            scaler = AdaptiveQuantileTransformer(output_distribution='uniform')
        elif self.scaler == 'quantile_normal':
            scaler = AdaptiveQuantileTransformer(output_distribution='normal')
        elif self.scaler == 'power_yeojohnson':
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        elif self.scaler == 'power_boxcox':
            scaler = PowerTransformer(method='box-cox', standardize=True)
        else:
            scaler = None
        
        if scaler:
            transformers.append(('scaler', scaler, columns))
        
        return transformers


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection stage"""
    method: Literal[
        'none', 'variance_threshold', 'correlation', 'mutual_info', 
        'rfe', 'select_from_model', 'boruta', 'genetic'
    ] = 'none'
    k_features: Optional[int] = None
    estimator_for_selection: Optional[BaseEstimator] = None
    threshold: Union[str, float] = 'mean'
    max_iter: int = 100
    
    def to_selector(self, X: Optional[pd.DataFrame], y: Optional[pd.Series], task_type: str) -> Optional[TransformerMixin]:
        if self.method == 'none':
            return None
        
        if self.method == 'variance_threshold':
            from sklearn.feature_selection import VarianceThreshold
            return VarianceThreshold(threshold=0.01)
        
        elif self.method == 'correlation':
            return CorrelationSelector(k=self.k_features or 10, task_type=task_type)
        
        elif self.method == 'mutual_info':
            score_func = mutual_info_regression if task_type == 'regression' else mutual_info_classif
            return SelectKBest(score_func=score_func, k=self.k_features or 10)
        
        elif self.method == 'rfe':
            if not self.estimator_for_selection:
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                self.estimator_for_selection = RandomForestRegressor() if task_type == 'regression' else RandomForestClassifier()
            return RFE(estimator=self.estimator_for_selection, n_features_to_select=self.k_features or 10)
        
        elif self.method == 'select_from_model':
            if not self.estimator_for_selection:
                from sklearn.linear_model import Lasso, LogisticRegression
                self.estimator_for_selection = Lasso(alpha=0.1) if task_type == 'regression' else LogisticRegression(penalty='l1', solver='liblinear')
            return SelectFromModel(estimator=self.estimator_for_selection, threshold=self.threshold)
        
        elif self.method == 'boruta':
            try:
                from mlxtend.feature_selection import BorutaPy
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                estimator = RandomForestRegressor(n_jobs=1) if task_type == 'regression' else RandomForestClassifier(n_jobs=1)
                return BorutaPy(estimator=estimator, max_iter=self.max_iter)
            except ImportError:
                logger.warning("mlxtend not available, falling back to SelectFromModel")
                from sklearn.linear_model import Lasso, LogisticRegression
                est = Lasso(alpha=0.1) if task_type == 'regression' else LogisticRegression()
                return SelectFromModel(estimator=est)
        
        return None


class ConstraintAwareEstimator(BaseEstimator):
    """
    Wrapper for sklearn estimators to enforce monotonicity/linearity constraints
    """
    
    def __init__(
        self,
        base_estimator: BaseEstimator,
        constraints: Dict[str, Dict[str, Any]] = None,
        sigma_multiplier: float = 3.0,
        constraint_strength: Literal['strong', 'weak'] = 'weak'
    ):
        self.base_estimator = clone(base_estimator)
        self.constraints = constraints or {}
        self.sigma_multiplier = sigma_multiplier
        self.constraint_strength = constraint_strength
        self._feature_stats = {}
        
    def fit(self, X, y, sample_weight=None, **fit_params):
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if col in self.constraints and pd.api.types.is_numeric_dtype(X[col]):
                    self._feature_stats[col] = {
                        'mean': X[col].mean(),
                        'std': X[col].std(),
                        'min': X[col].min(),
                        'max': X[col].max()
                    }
        
        if hasattr(self.base_estimator, 'monotonic_constraints'):
            # XGBoost/LightGBM style
            constraints_tuple = self._build_monotonic_array(X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1]))
            if constraints_tuple:
                self.base_estimator.set_params(monotonic_constraints=constraints_tuple)
        
        if sample_weight is not None:
            return self.base_estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
        return self.base_estimator.fit(X, y, **fit_params)
    
    def predict(self, X):
        preds = self.base_estimator.predict(X)
        if self.constraint_strength == 'strong' and self.constraints:
            return self._enforce_constraints_posthoc(X, preds)
        return preds
    
    def _build_monotonic_array(self, feature_names):
        constraints = []
        for feat in feature_names:
            constraint = self.constraints.get(feat, {})
            direction = constraint.get('monotonic', None)
            if direction == 'increasing':
                constraints.append(1)
            elif direction == 'decreasing':
                constraints.append(-1)
            else:
                constraints.append(0)
        return tuple(constraints) if any(c != 0 for c in constraints) else None
    
    def _enforce_constraints_posthoc(self, X, predictions):
        # Implementation placeholder
        return predictions
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.base_estimator, name)


def generate_estimator_ui_metadata(estimator_class: type) -> Dict[str, Any]:
    """Dynamically generate UI metadata from estimator class signature"""
    metadata = {}
    sig = inspect.signature(estimator_class.__init__)
    doc_lines = (estimator_class.__doc__ or "").split('\n')
    param_descriptions = _parse_sklearn_docstring(doc_lines)
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self': continue
        
        param_meta = {'name': param_name}
        annotation = param.annotation
        
        if annotation == inspect.Parameter.empty:
            if param.default == inspect.Parameter.empty:
                param_meta['type'] = 'any'
            elif isinstance(param.default, bool):
                param_meta['type'] = 'bool'
            elif isinstance(param.default, int):
                param_meta['type'] = 'int'
            elif isinstance(param.default, float):
                param_meta['type'] = 'float'
            elif isinstance(param.default, str):
                param_meta['type'] = 'str'
            elif isinstance(param.default, (list, tuple)):
                param_meta['type'] = 'list'
            else:
                param_meta['type'] = 'any'
        else:
            param_meta['type'] = _python_type_to_ui_type(annotation)
        
        if param.default != inspect.Parameter.empty:
            param_meta['default'] = param.default
            if param_meta['type'] in ['int', 'float']:
                if param_name in ['n_estimators', 'max_depth', 'min_samples_split']:
                    param_meta['min'] = 1
                    param_meta['max'] = 1000
                elif 'alpha' in param_name or 'lambda' in param_name:
                    param_meta['min'] = 0.0
                    param_meta['max'] = 10.0
        
        if param_name in ['criterion', 'kernel', 'penalty', 'solver']:
            param_meta['choices'] = _get_common_choices(param_name, estimator_class)
            param_meta['type'] = 'choice'
        
        if param_name in param_descriptions:
            param_meta['description'] = param_descriptions[param_name]
        
        metadata[param_name] = param_meta
    
    return metadata


def _parse_sklearn_docstring(doc_lines: List[str]) -> Dict[str, str]:
    params = {}
    in_params_section = False
    for line in doc_lines:
        line = line.strip()
        if line.lower().startswith('parameters'):
            in_params_section = True
            continue
        if in_params_section:
            if line and not line.startswith(' ') and ':' not in line: break
            if ':' in line:
                parts = line.split(':', 1)
                params[parts[0].strip()] = parts[1].strip()
    return params


def _python_type_to_ui_type(py_type) -> str:
    if py_type == bool: return 'bool'
    elif py_type == int: return 'int'
    elif py_type == float: return 'float'
    elif py_type == str: return 'str'
    return 'any'


def _get_common_choices(param_name: str, estimator_class: type) -> List[Any]:
    choices_map = {
        'criterion': ['gini', 'entropy', 'log_loss', 'squared_error', 'absolute_error'],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'saga'],
    }
    return choices_map.get(param_name, [])


def build_pipeline(
    column_config: Dict[str, PreprocessingConfig],
    feature_generation: List[Dict[str, Any]] = None,
    feature_selection: FeatureSelectionConfig = None,
    estimator: BaseEstimator = None,
    constraints: Dict[str, Dict[str, Any]] = None,
    task_type: Literal['regression', 'classification'] = 'regression'
) -> Pipeline:
    steps = []
    
    # Preprocessing
    transformers = []
    for col_name, config in column_config.items():
        if config:
            transformers.extend(config.to_transformer([col_name], task_type))
    
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        steps.append(('preprocessor', preprocessor))
    
    # Feature Generation
    if feature_generation:
        for idx, gen_spec in enumerate(feature_generation):
            if gen_spec.get('method') == 'polynomial':
                from sklearn.preprocessing import PolynomialFeatures
                steps.append((f'poly_{idx}', PolynomialFeatures(degree=gen_spec.get('degree', 2), include_bias=False)))
    
    # Feature Selection
    if feature_selection and feature_selection.method != 'none':
        selector = feature_selection.to_selector(None, None, task_type)
        if selector:
            steps.append(('feature_selector', selector))
    
    # Estimator
    if estimator:
        wrapped_estimator = ConstraintAwareEstimator(
            base_estimator=estimator,
            constraints=constraints,
            sigma_multiplier=3.0,
            constraint_strength='weak'
        )
        steps.append(('estimator', wrapped_estimator))
    
    return Pipeline(steps=steps)
