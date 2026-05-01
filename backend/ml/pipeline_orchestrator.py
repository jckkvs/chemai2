"""
Pipeline Orchestrator - chemai2/backend/ml/pipeline_orchestrator.py
5-stage ML pipeline with automatic type detection and constraint support
"""
import json
import joblib
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from backend.ml.transformers import AdaptiveQuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    f_regression, f_classif, mutual_info_regression, mutual_info_classif,
    chi2, VarianceThreshold
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, LeaveOneGroupOut,
    GroupKFold, TimeSeriesSplit, ShuffleSplit, PredefinedSplit
)

from backend.core.config import settings
from backend.utils.logger import logger
from backend.ml.constraints import ConstraintEngine, ConstraintSpec
from backend.ml.estimators import get_estimator_class, ESTIMATOR_REGISTRY
from backend.ml.feature_selection import get_selector_class, SELECTOR_REGISTRY


class VariableType(Enum):
    """Detected variable types for automatic preprocessing"""
    NUMERIC_CONTINUOUS = auto()
    NUMERIC_DISCRETE = auto()
    NUMERIC_EXPONENTIAL = auto()  # Requires log transform
    NUMERIC_LOGNORMAL = auto()     # Already logged or needs exp
    BINARY = auto()
    CATEGORICAL_LOW = auto()       # <=10 unique values
    CATEGORICAL_HIGH = auto()      # >10 unique values
    TEXT = auto()
    DATETIME = auto()
    SMILES = auto()


@dataclass
class ColumnPreprocessingConfig:
    """Configuration for preprocessing a single column"""
    # Detection (auto or manual)
    detected_type: Optional[VariableType] = None
    user_override_type: Optional[VariableType] = None
    
    # Imputation
    imputer: Literal['mean', 'median', 'most_frequent', 'constant', 'knn', 'none'] = 'mean'
    imputer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Scaling/Transformation
    scaler: Literal['standard', 'minmax', 'robust', 'quantile_uniform', 
                   'quantile_normal', 'power_yeojohnson', 'power_boxcox', 
                   'log', 'log1p', 'exp', 'none'] = 'standard'
    scaler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Encoding (for categorical)
    encoder: Literal['onehot', 'ordinal', 'target', 'frequency', 'binary', 'none'] = 'onehot'
    encoder_params: Dict[str, Any] = field(default_factory=dict)
    max_categories: int = 50  # Max categories for one-hot before switching to ordinal
    
    # Constraint metadata
    constraint: Optional[ConstraintSpec] = None
    
    # UI metadata
    label: Optional[str] = None
    description: Optional[str] = None
    group: Optional[str] = None  # For GroupLasso etc.
    
    @property
    def effective_type(self) -> VariableType:
        """Get effective type (user override takes precedence)"""
        return self.user_override_type or self.detected_type or VariableType.NUMERIC_CONTINUOUS
    
    def to_transformer(self, column: str) -> Tuple[str, TransformerMixin, List[str]]:
        """Convert config to sklearn ColumnTransformer step"""
        transformers = []
        columns = [column]
        
        # 1. Imputation
        if self.imputer != 'none':
            if self.imputer == 'knn':
                imputer = KNNImputer(**self.imputer_params)
            else:
                imputer = SimpleImputer(strategy=self.imputer, **self.imputer_params)
            transformers.append(('imputer', imputer))
        
        # 2. Scaling/Transformation
        if self.scaler != 'none':
            if self.scaler == 'standard':
                scaler = StandardScaler(**self.scaler_params)
            elif self.scaler == 'minmax':
                scaler = MinMaxScaler(**self.scaler_params)
            elif self.scaler == 'robust':
                scaler = RobustScaler(**self.scaler_params)
            elif self.scaler == 'quantile_uniform':
                scaler = AdaptiveQuantileTransformer(output_distribution='uniform', **self.scaler_params)
            elif self.scaler == 'quantile_normal':
                scaler = AdaptiveQuantileTransformer(output_distribution='normal', **self.scaler_params)
            elif self.scaler == 'power_yeojohnson':
                scaler = PowerTransformer(method='yeo-johnson', **self.scaler_params)
            elif self.scaler == 'power_boxcox':
                scaler = PowerTransformer(method='box-cox', **self.scaler_params)
            elif self.scaler == 'log':
                from backend.ml.transformers import LogTransformer
                scaler = LogTransformer(**self.scaler_params)
            elif self.scaler == 'log1p':
                from backend.ml.transformers import Log1pTransformer
                scaler = Log1pTransformer(**self.scaler_params)
            elif self.scaler == 'exp':
                from backend.ml.transformers import ExpTransformer
                scaler = ExpTransformer(**self.scaler_params)
            else:
                scaler = None
            
            if scaler:
                transformers.append(('scaler', scaler))
        
        # 3. Encoding (for categorical types)
        eff_type = self.effective_type
        if eff_type in [VariableType.CATEGORICAL_LOW, VariableType.CATEGORICAL_HIGH] and self.encoder != 'none':
            if self.encoder == 'onehot':
                from sklearn.preprocessing import OneHotEncoder
                encoder = OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    max_categories=self.max_categories if eff_type == VariableType.CATEGORICAL_HIGH else None,
                    **self.encoder_params
                )
            elif self.encoder == 'ordinal':
                from sklearn.preprocessing import OrdinalEncoder
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, **self.encoder_params)
            elif self.encoder == 'target':
                from category_encoders import TargetEncoder
                encoder = TargetEncoder(**self.encoder_params)
            elif self.encoder == 'frequency':
                from category_encoders import CountEncoder
                encoder = CountEncoder(**self.encoder_params)
            elif self.encoder == 'binary':
                from category_encoders import BinaryEncoder
                encoder = BinaryEncoder(**self.encoder_params)
            else:
                encoder = None
            
            if encoder:
                transformers.append(('encoder', encoder))
        
        # Combine transformers into ColumnTransformer step
        if not transformers:
            return ('passthrough', 'passthrough', columns)
        
        # Create named transformer pipeline for this column
        from sklearn.pipeline import make_pipeline
        column_transformer = make_pipeline(*[t for _, t in transformers if t != 'passthrough'])
        
        return (f'col_{column}', column_transformer, columns)


@dataclass
class FeatureGenerationConfig:
    """Configuration for feature generation stage"""
    enabled: bool = True
    methods: List[Literal['polynomial', 'interaction', 'binning', 'custom']] = field(default_factory=lambda: ['polynomial'])
    
    # Polynomial features
    polynomial_degree: int = 2
    polynomial_include_bias: bool = False
    polynomial_interaction_only: bool = False
    
    # Binning
    binning_strategy: Literal['uniform', 'quantile', 'kmeans'] = 'quantile'
    binning_n_bins: int = 10
    
    # Custom function (user-defined)
    custom_function: Optional[Callable] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_transformer(self) -> Optional[TransformerMixin]:
        """Convert config to sklearn transformer"""
        if not self.enabled:
            return None
        
        transformers = []
        
        if 'polynomial' in self.methods:
            from sklearn.preprocessing import PolynomialFeatures
            transformers.append((
                'poly',
                PolynomialFeatures(
                    degree=self.polynomial_degree,
                    include_bias=self.polynomial_include_bias,
                    interaction_only=self.polynomial_interaction_only
                )
            ))
        
        if 'binning' in self.methods:
            from sklearn.preprocessing import KBinsDiscretizer
            transformers.append((
                'binning',
                KBinsDiscretizer(
                    n_bins=self.binning_n_bins,
                    encode='ordinal',
                    strategy=self.binning_strategy
                )
            ))
        
        if not transformers:
            return None
        
        if len(transformers) == 1:
            return transformers[0][1]
        
        # Chain multiple generators
        from sklearn.pipeline import make_pipeline
        return make_pipeline(*[t for _, t in transformers])


@dataclass
class PipelineConfig:
    """Complete configuration for 5-stage ML pipeline"""
    # Stage 1: Column selection
    include_columns: Optional[List[str]] = None  # None = all columns
    exclude_columns: List[str] = field(default_factory=list)
    
    # Stage 2: Column-wise preprocessing (per-column config)
    column_configs: Dict[str, ColumnPreprocessingConfig] = field(default_factory=dict)
    auto_detect_types: bool = True
    
    # Stage 3: Feature generation
    feature_generation: FeatureGenerationConfig = field(default_factory=FeatureGenerationConfig)
    
    # Stage 4: Feature selection
    feature_selection: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'method': 'none',  # none, variance, correlation, mutual_info, rfe, select_from_model, boruta, genetic
        'k_features': None,
        'estimator': None,  # For model-based selection
        'params': {}
    })
    
    # Stage 5: Estimator
    estimator_name: str = 'RandomForestRegressor'
    estimator_params: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, ConstraintSpec] = field(default_factory=dict)
    
    # Task type
    task_type: Literal['regression', 'classification'] = 'regression'
    
    # Cross-validation
    cv_strategy: str = 'kfold'
    cv_params: Dict[str, Any] = field(default_factory=lambda: {'n_splits': 5})
    
    # Metadata
    name: str = 'default_pipeline'
    description: str = ''
    version: str = '1.0.0'
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict (for storage/API)"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Deserialize config from dict"""
        # Handle nested dataclasses
        if 'column_configs' in data:
            data['column_configs'] = {
                k: ColumnPreprocessingConfig(**v) if isinstance(v, dict) else v
                for k, v in data['column_configs'].items()
            }
        if 'feature_generation' in data and isinstance(data['feature_generation'], dict):
            data['feature_generation'] = FeatureGenerationConfig(**data['feature_generation'])
        if 'constraints' in data:
            data['constraints'] = {
                k: ConstraintSpec(**v) if isinstance(v, dict) else v
                for k, v in data['constraints'].items()
            }
        return cls(**data)


class PipelineOrchestrator:
    """
    Orchestrates the 5-stage ML pipeline with automatic type detection
    and constraint-aware model training
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self._fitted = False
        self._feature_names: Optional[List[str]] = None
        self._constraint_engine: Optional[ConstraintEngine] = None
        
    def detect_variable_types(self, df: pd.DataFrame) -> Dict[str, VariableType]:
        """
        Automatically detect variable types for each column
        
        Detection logic:
        - SMILES: Valid SMILES pattern + RDKit parse success
        - Binary: Only 0/1 or True/False values
        - Categorical: Low cardinality (<10 unique) or object dtype
        - Numeric exponential: Positive values with high skew (>3)
        - Numeric lognormal: Log-transformed data (check after log)
        - Numeric discrete: Integer values with limited range
        - Numeric continuous: Default for float
        """
        detections = {}
        
        for col in df.columns:
            if col in self.config.exclude_columns:
                continue
            if self.config.include_columns and col not in self.config.include_columns:
                continue
                
            series = df[col].dropna()
            if len(series) == 0:
                detections[col] = VariableType.TEXT
                continue
            
            # SMILES detection
            if self._is_smiles_column(series):
                detections[col] = VariableType.SMILES
                continue
            
            # Binary detection
            if self._is_binary_column(series):
                detections[col] = VariableType.BINARY
                continue
            
            # Categorical detection
            if self._is_categorical_column(series):
                n_unique = series.nunique()
                detections[col] = VariableType.CATEGORICAL_LOW if n_unique <= 10 else VariableType.CATEGORICAL_HIGH
                continue
            
            # Numeric detection
            if pd.api.types.is_numeric_dtype(series):
                # Check for exponential distribution (positive, high skew)
                if (series > 0).all() and series.skew() > 3:
                    detections[col] = VariableType.NUMERIC_EXPONENTIAL
                # Check for lognormal (try log transform)
                elif (series > 0).all() and np.log1p(series).skew() < abs(series.skew()):
                    detections[col] = VariableType.NUMERIC_LOGNORMAL
                # Discrete vs continuous
                elif pd.api.types.is_integer_dtype(series) and series.nunique() < len(series) * 0.1:
                    detections[col] = VariableType.NUMERIC_DISCRETE
                else:
                    detections[col] = VariableType.NUMERIC_CONTINUOUS
                continue
            
            # Datetime detection
            if pd.api.types.is_datetime64_any_dtype(series) or self._is_datetime_string(series):
                detections[col] = VariableType.DATETIME
                continue
            
            # Default to text
            detections[col] = VariableType.TEXT
        
        return detections
    
    def _is_smiles_column(self, series: pd.Series, threshold: float = 0.8) -> bool:
        """Check if series likely contains SMILES strings"""
        try:
            from rdkit import Chem
            sample = series.dropna().sample(min(50, len(series)), random_state=42)
            valid = sum(1 for s in sample if Chem.MolFromSmiles(str(s)) is not None)
            return valid / len(sample) >= threshold
        except ImportError:
            return False
    
    def _is_binary_column(self, series: pd.Series) -> bool:
        """Check if series contains only binary values"""
        unique = set(series.dropna().unique())
        return unique <= {0, 1, True, False, 0.0, 1.0}
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Check if series should be treated as categorical"""
        if series.dtype == 'object' or series.dtype.name == 'category':
            return True
        if pd.api.types.is_numeric_dtype(series):
            # Low cardinality numeric can be categorical
            return series.nunique() / len(series) < 0.05 and series.nunique() <= 50
        return False
    
    def _is_datetime_string(self, series: pd.Series) -> bool:
        """Check if string series contains datetime patterns"""
        if series.dtype != 'object':
            return False
        sample = series.dropna().sample(min(20, len(series)), random_state=42)
        # Simple pattern check
        datetime_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}:\d{2}:\d{2}']
        import re
        matches = sum(1 for s in sample if any(re.search(p, str(s)) for p in datetime_patterns))
        return matches / len(sample) > 0.5
    
    def build_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Pipeline:
        """
        Build the complete 5-stage pipeline from config and data
        
        Stages:
        1. ColumnSelector - Input column control
        2. ColumnTransformer - Column-wise preprocessing
        3. FeatureGenerator - Polynomial/interaction features
        4. FeatureSelector - Feature selection methods
        5. Estimator - Constrained model with CV
        """
        steps = []
        
        # ========== Stage 1: Column Selection ==========
        columns_to_use = self._get_columns_to_use(X)
        if columns_to_use and len(columns_to_use) < len(X.columns):
            try:
                from mlxtend.feature_selection import ColumnSelector
                steps.append(('column_selector', ColumnSelector(columns=columns_to_use)))
            except ImportError:
                logger.warning("mlxtend not available, skipping ColumnSelector")
        
        # ========== Stage 2: Column-wise Preprocessing ==========
        # Auto-detect types if not manually specified
        if self.config.auto_detect_types:
            detected = self.detect_variable_types(X)
            for col in columns_to_use or X.columns:
                if col not in self.config.column_configs:
                    self.config.column_configs[col] = ColumnPreprocessingConfig(detected_type=detected.get(col))
                elif self.config.column_configs[col].detected_type is None:
                    self.config.column_configs[col].detected_type = detected.get(col)
        
        # Build ColumnTransformer
        transformers = []
        for col in columns_to_use or X.columns:
            if col in self.config.column_configs:
                name, transformer, cols = self.config.column_configs[col].to_transformer(col)
                if transformer != 'passthrough':
                    transformers.append((name, transformer, cols))
                else:
                    transformers.append((f'col_{col}', 'passthrough', [col]))
        
        if transformers:
            preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
            steps.append(('preprocessor', preprocessor))
        
        # ========== Stage 3: Feature Generation ==========
        if self.config.feature_generation.enabled:
            generator = self.config.feature_generation.to_transformer()
            if generator:
                steps.append(('feature_generator', generator))
        
        # ========== Stage 4: Feature Selection ==========
        if self.config.feature_selection.get('enabled', False):
            selector = self._build_feature_selector()
            if selector:
                steps.append(('feature_selector', selector))
        
        # ========== Stage 5: Estimator with Constraints ==========
        estimator = self._build_constrained_estimator()
        steps.append(('estimator', estimator))
        
        # Create pipeline
        self.pipeline = Pipeline(steps=steps, memory=str(settings.CACHE_DIR / 'pipeline_cache'))
        self._feature_names = columns_to_use or list(X.columns)
        
        # Initialize constraint engine
        if self.config.constraints:
            self._constraint_engine = ConstraintEngine(self.config.constraints)
            self._constraint_engine.fit(X[self._feature_names] if isinstance(X, pd.DataFrame) else X)
        
        return self.pipeline
    
    def _get_columns_to_use(self, X: pd.DataFrame) -> Optional[List[str]]:
        """Determine which columns to include in pipeline"""
        if self.config.include_columns:
            return [c for c in self.config.include_columns if c in X.columns and c not in self.config.exclude_columns]
        return [c for c in X.columns if c not in self.config.exclude_columns]
    
    def _build_feature_selector(self) -> Optional[TransformerMixin]:
        """Build feature selector based on config"""
        fs_config = self.config.feature_selection
        method = fs_config.get('method', 'none')
        
        if method == 'none' or not fs_config.get('enabled', False):
            return None
        
        task = self.config.task_type
        k = fs_config.get('k_features')
        params = fs_config.get('params', {})
        
        if method == 'variance':
            return VarianceThreshold(**params)
        
        elif method == 'correlation':
            score_func = f_regression if task == 'regression' else f_classif
            return SelectKBest(score_func=score_func, k=k or 'all')
        
        elif method == 'mutual_info':
            score_func = mutual_info_regression if task == 'regression' else mutual_info_classif
            return SelectKBest(score_func=score_func, k=k or 'all')
        
        elif method == 'chi2':
            if task != 'classification':
                logger.warning("chi2 only for classification, skipping")
                return None
            return SelectKBest(score_func=chi2, k=k or 'all')
        
        elif method == 'rfe':
            est_name = fs_config.get('estimator', 'RandomForestRegressor' if task == 'regression' else 'RandomForestClassifier')
            estimator = get_estimator_class(est_name)(**params.get('estimator_params', {}))
            return RFE(estimator=estimator, n_features_to_select=k, **params)
        
        elif method == 'rfecv':
            est_name = fs_config.get('estimator', 'RandomForestRegressor' if task == 'regression' else 'RandomForestClassifier')
            estimator = get_estimator_class(est_name)(**params.get('estimator_params', {}))
            return RFECV(estimator=estimator, **params)
        
        elif method == 'select_from_model':
            est_name = fs_config.get('estimator', 'Lasso' if task == 'regression' else 'LogisticRegression')
            estimator = get_estimator_class(est_name)(**params.get('estimator_params', {}))
            return SelectFromModel(estimator=estimator, **params)
        
        elif method == 'boruta':
            try:
                from mlxtend.feature_selection import BorutaPy
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                est = RandomForestRegressor(random_state=42) if task == 'regression' else RandomForestClassifier(random_state=42)
                return BorutaPy(estimator=est, max_iter=params.get('max_iter', 100), **{k: v for k, v in params.items() if k != 'max_iter'})
            except ImportError:
                logger.warning("mlxtend not available for Boruta, falling back to SelectFromModel")
                return self._build_feature_selector()  # Fallback
        
        elif method == 'genetic':
            try:
                from sklearn_genetic import GASearchCV
                # Genetic selection requires more setup, return placeholder
                logger.info("Genetic selection configured - requires GASearchCV setup")
                return None  # Placeholder for future implementation
            except ImportError:
                logger.warning("sklearn-genetic-opt not available for genetic selection")
                return None
        
        return None
    
    def _build_constrained_estimator(self) -> BaseEstimator:
        """Build estimator with constraint wrapper"""
        from backend.ml.estimators import ConstrainedEstimatorWrapper
        
        estimator = get_estimator_class(self.config.estimator_name)(**self.config.estimator_params)
        
        return ConstrainedEstimatorWrapper(
            base_estimator=estimator,
            constraints=self.config.constraints,
            task_type=self.config.task_type
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> 'PipelineOrchestrator':
        """Fit the pipeline to data"""
        if self.pipeline is None:
            self.build_pipeline(X, y)
        
        self.pipeline.fit(X, y, **fit_params)
        self._fitted = True
        
        # Evaluate constraints if configured
        if self._constraint_engine and self.config.constraints:
            evaluations = self._constraint_engine.evaluate_constraints(
                self.pipeline.named_steps['estimator'].base_estimator,
                X[self._feature_names] if isinstance(X, pd.DataFrame) else X,
                y
            )
            logger.info(f"Constraint evaluation: {sum(e.passed for e in evaluations.values())}/{len(evaluations)} passed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with constraint post-processing if needed"""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before prediction")
        
        predictions = self.pipeline.predict(X)
        
        # Apply strong constraint post-processing
        if self._constraint_engine and self.config.constraints:
            has_strong = any(c.strength == 'strong' for c in self.config.constraints.values())
            if has_strong:
                predictions = self._constraint_engine.enforce_strong_constraints_posthoc(
                    self.pipeline.named_steps['estimator'].base_estimator,
                    X[self._feature_names] if isinstance(X, pd.DataFrame) else X,
                    predictions,
                    self.config.constraints
                )
        
        return predictions
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Score the pipeline (R² for regression, accuracy for classification)"""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before scoring")
        return self.pipeline.score(X, y)
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after all transformations"""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted first")
        
        # Traverse pipeline steps to get final feature names
        names = self._feature_names or []
        
        for name, step in self.pipeline.steps:
            if hasattr(step, 'get_feature_names_out'):
                try:
                    names = step.get_feature_names_out(names)
                except:
                    pass  # Some transformers don't support this
            elif name == 'feature_generator' and hasattr(step, 'powers_'):
                # PolynomialFeatures
                from sklearn.preprocessing import PolynomialFeatures
                if isinstance(step, PolynomialFeatures):
                    names = [f"poly_{i}" for i in range(len(names))]
        
        return names
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save pipeline config and fitted model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
        
        # Save fitted pipeline if available
        if self._fitted and self.pipeline:
            model_path = path / 'model.joblib'
            joblib.dump(self.pipeline, model_path)
        
        # Save feature names
        if self._feature_names:
            names_path = path / 'feature_names.json'
            with open(names_path, 'w') as f:
                json.dump(self._feature_names, f)
        
        logger.info(f"Pipeline saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PipelineOrchestrator':
        """Load pipeline config and optionally fitted model"""
        path = Path(path)
        
        # Load config
        config_path = path / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = PipelineConfig.from_dict(config_dict)
        
        # Create orchestrator
        orchestrator = cls(config)
        
        # Load fitted pipeline if exists
        model_path = path / 'model.joblib'
        if model_path.exists():
            orchestrator.pipeline = joblib.load(model_path)
            orchestrator._fitted = True
        
        # Load feature names
        names_path = path / 'feature_names.json'
        if names_path.exists():
            with open(names_path, 'r') as f:
                orchestrator._feature_names = json.load(f)
        
        logger.info(f"Pipeline loaded from {path}")
        return orchestrator
