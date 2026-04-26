"""
Dynamic UI Metadata Generator - chemai2/backend/utils/ui_metadata.py
Auto-generates frontend form schemas from Python class signatures
"""
import ast
import inspect
import textwrap
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, Literal
from enum import Enum
import re

from pydantic import BaseModel, Field, create_model
from fastapi import HTTPException

from backend.utils.logger import logger


class UIInputType(str, Enum):
    """Supported UI input types for frontend rendering"""
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    SLIDER = "slider"
    TEXTAREA = "textarea"
    FILE = "file"
    COLOR = "color"
    DATE = "date"
    DATETIME = "datetime"


class UIParamMetadata(BaseModel):
    """Metadata for a single parameter to render in UI"""
    name: str
    label: Optional[str] = None  # Human-readable label
    input_type: UIInputType = UIInputType.TEXT
    default: Any = None
    required: bool = False
    description: Optional[str] = None
    placeholder: Optional[str] = None
    
    # Numeric constraints
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    
    # Choice options (for select/multi_select)
    options: Optional[List[Dict[str, Any]]] = None  # [{value, label, description}]
    
    # Validation rules
    pattern: Optional[str] = None  # Regex pattern for text input
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    # UI hints
    hidden: bool = False
    disabled: bool = False
    depends_on: Optional[Dict[str, Any]] = None  # Conditional display logic
    advanced: bool = False  # Show only in "advanced mode"
    
    # Grouping
    category: str = "general"
    order: int = 0
    
    class Config:
        extra = "allow"  # Allow frontend-specific extensions


class UIMetadataGenerator:
    """
    Generates UI metadata from Python classes/functions
    
    Supports:
    - sklearn-style estimators with __init__ parameters
    - Functions with type hints and docstrings
    - Pydantic models
    - Custom plugins with PLUGIN_SPEC attribute
    """
    
    # Type mapping: Python type -> UI input type
    TYPE_MAPPING: Dict[type, UIInputType] = {
        bool: UIInputType.BOOLEAN,
        int: UIInputType.INTEGER,
        float: UIInputType.NUMBER,
        str: UIInputType.TEXT,
        list: UIInputType.TEXTAREA,
        tuple: UIInputType.TEXTAREA,
    }
    
    # Common parameter patterns for auto-enhancement
    PARAM_PATTERNS: Dict[str, Dict[str, Any]] = {
        r'^(n_estimators|max_depth|min_samples|max_features)$': {
            'input_type': UIInputType.SLIDER,
            'min_value': 1,
            'max_value': 1000,
            'step': 1,
            'category': 'model_complexity'
        },
        r'^(alpha|lambda|reg_|penalty|C)$': {
            'input_type': UIInputType.SLIDER,
            'min_value': 0.0,
            'max_value': 10.0,
            'step': 0.01,
            'category': 'regularization'
        },
        r'^(learning_rate|lr|eta)$': {
            'input_type': UIInputType.SLIDER,
            'min_value': 0.001,
            'max_value': 1.0,
            'step': 0.001,
            'category': 'optimization'
        },
        r'^(random_state|seed)$': {
            'input_type': UIInputType.INTEGER,
            'min_value': 0,
            'max_value': 2**32 - 1,
            'category': 'reproducibility'
        },
        r'^(criterion|kernel|penalty|solver|loss|activation)$': {
            'input_type': UIInputType.SELECT,
            'category': 'algorithm'
        },
        r'^(verbose|debug|show_progress)$': {
            'input_type': UIInputType.BOOLEAN,
            'advanced': True,
            'category': 'debugging'
        },
    }
    
    @classmethod
    def from_class(cls, target_class: type, exclude_params: List[str] = None) -> Dict[str, UIParamMetadata]:
        """Generate UI metadata from a class __init__ signature"""
        exclude_params = exclude_params or ['self', 'cls']
        metadata = {}
        
        # Get __init__ signature
        try:
            sig = inspect.signature(target_class.__init__)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not inspect {target_class.__name__}: {e}")
            return metadata
        
        # Parse docstring for parameter descriptions
        docstring = inspect.getdoc(target_class) or ""
        param_descriptions = cls._parse_numpy_docstring(docstring)
        class_description = cls._extract_class_description(docstring)
        
        for param_name, param in sig.parameters.items():
            if param_name in exclude_params:
                continue
            
            param_meta = cls._analyze_parameter(param_name, param, param_descriptions)
            if param_meta:
                metadata[param_name] = param_meta
        
        # Add class-level metadata
        metadata['_class_info'] = {
            'name': target_class.__name__,
            'module': target_class.__module__,
            'description': class_description,
            'category': cls._infer_category(target_class)
        }
        
        return metadata
    
    @classmethod
    def from_function(cls, func: callable, exclude_params: List[str] = None) -> Dict[str, UIParamMetadata]:
        """Generate UI metadata from a function signature"""
        exclude_params = exclude_params or []
        metadata = {}
        
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not inspect {func.__name__}: {e}")
            return metadata
        
        docstring = inspect.getdoc(func) or ""
        param_descriptions = cls._parse_numpy_docstring(docstring)
        
        for param_name, param in sig.parameters.items():
            if param_name in exclude_params or param_name.startswith('_'):
                continue
            
            param_meta = cls._analyze_parameter(param_name, param, param_descriptions)
            if param_meta:
                metadata[param_name] = param_meta
        
        # Add function-level metadata
        metadata['_function_info'] = {
            'name': func.__name__,
            'module': func.__module__,
            'description': docstring.split('\n\n')[0].strip() if docstring else "",
            'returns': cls._parse_returns(docstring)
        }
        
        return metadata
    
    @classmethod
    def from_plugin_spec(cls, plugin_spec: Dict[str, Any]) -> Dict[str, UIParamMetadata]:
        """Generate UI metadata from a plugin specification dict"""
        metadata = {}
        
        param_schema = plugin_spec.get('param_schema', {})
        default_params = plugin_spec.get('default_params', {})
        
        for param_name, schema in param_schema.items():
            param_meta = UIParamMetadata(
                name=param_name,
                label=schema.get('label', param_name.replace('_', ' ').title()),
                input_type=cls._schema_type_to_ui_type(schema.get('type', 'string')),
                default=default_params.get(param_name, schema.get('default')),
                required=schema.get('required', False),
                description=schema.get('description'),
                placeholder=schema.get('placeholder'),
                min_value=schema.get('min'),
                max_value=schema.get('max'),
                step=schema.get('step'),
                options=schema.get('options'),
                pattern=schema.get('pattern'),
                min_length=schema.get('min_length'),
                max_length=schema.get('max_length'),
                hidden=schema.get('hidden', False),
                disabled=schema.get('disabled', False),
                depends_on=schema.get('depends_on'),
                advanced=schema.get('advanced', False),
                category=schema.get('category', 'general'),
                order=schema.get('order', 0),
            )
            metadata[param_name] = param_meta
        
        return metadata
    
    @classmethod
    def _analyze_parameter(
        cls, 
        param_name: str, 
        param: inspect.Parameter,
        descriptions: Dict[str, str]
    ) -> Optional[UIParamMetadata]:
        """Analyze a single parameter and generate UI metadata"""
        
        # Determine type and UI input type
        input_type, type_info = cls._infer_input_type(param)
        
        # Get default value
        default = param.default if param.default != inspect.Parameter.empty else None
        
        # Apply pattern-based enhancements
        enhancements = cls._get_pattern_enhancements(param_name)
        
        # Build options for select inputs
        options = None
        if input_type in [UIInputType.SELECT, UIInputType.MULTI_SELECT]:
            options = cls._generate_options(param_name, param, type_info)
        
        # Determine if parameter should be hidden or advanced
        hidden = param_name.startswith('_') or enhancements.get('hidden', False)
        advanced = enhancements.get('advanced', False) or param_name in ['verbose', 'n_jobs']
        
        return UIParamMetadata(
            name=param_name,
            label=param_name.replace('_', ' ').title(),
            input_type=input_type,
            default=default,
            required=param.default == inspect.Parameter.empty,
            description=descriptions.get(param_name),
            placeholder=enhancements.get('placeholder'),
            min_value=enhancements.get('min_value') or type_info.get('min'),
            max_value=enhancements.get('max_value') or type_info.get('max'),
            step=enhancements.get('step') or type_info.get('step'),
            options=options,
            pattern=enhancements.get('pattern'),
            min_length=enhancements.get('min_length'),
            max_length=enhancements.get('max_length'),
            hidden=hidden,
            disabled=enhancements.get('disabled', False),
            depends_on=enhancements.get('depends_on'),
            advanced=advanced,
            category=enhancements.get('category', 'general'),
            order=enhancements.get('order', 0),
        )
    
    @classmethod
    def _infer_input_type(cls, param: inspect.Parameter) -> tuple[UIInputType, Dict[str, Any]]:
        """Infer UI input type from parameter annotation and default value"""
        annotation = param.annotation
        default = param.default if param.default != inspect.Parameter.empty else None
        
        type_info = {}
        
        # Handle Literal types (enum-like)
        if get_origin(annotation) is Literal:
            args = get_args(annotation)
            return UIInputType.SELECT, {'options': [{'value': a, 'label': str(a)} for a in args]}
        
        # Handle Union/Optional types
        if get_origin(annotation) is Union:
            args = get_args(annotation)
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                annotation = non_none_args[0]
        
        # Direct type mapping
        if annotation in cls.TYPE_MAPPING:
            return cls.TYPE_MAPPING[annotation], type_info
        
        # Infer from default value
        if default is not None and not isinstance(default, (list, tuple, dict)):
            default_type = type(default)
            if default_type in cls.TYPE_MAPPING:
                return cls.TYPE_MAPPING[default_type], type_info
        
        # Special handling for common types
        if annotation == int or (isinstance(default, int) and default is not True and default is not False):
            type_info.update({'min': 0, 'max': 10000, 'step': 1})
            return UIInputType.INTEGER, type_info
        
        if annotation == float or isinstance(default, float):
            type_info.update({'min': 0.0, 'max': 1.0, 'step': 0.01})
            return UIInputType.NUMBER, type_info
        
        if annotation == bool or isinstance(default, bool):
            return UIInputType.BOOLEAN, type_info
        
        if annotation == str or isinstance(default, str):
            return UIInputType.TEXT, type_info
        
        # Fallback
        return UIInputType.TEXT, type_info
    
    @classmethod
    def _schema_type_to_ui_type(cls, schema_type: str) -> UIInputType:
        """Convert plugin schema type string to UIInputType"""
        mapping = {
            'string': UIInputType.TEXT,
            'str': UIInputType.TEXT,
            'integer': UIInputType.INTEGER,
            'int': UIInputType.INTEGER,
            'number': UIInputType.NUMBER,
            'float': UIInputType.NUMBER,
            'boolean': UIInputType.BOOLEAN,
            'bool': UIInputType.BOOLEAN,
            'array': UIInputType.MULTI_SELECT,
            'list': UIInputType.MULTI_SELECT,
            'enum': UIInputType.SELECT,
            'choice': UIInputType.SELECT,
            'text': UIInputType.TEXTAREA,
        }
        return mapping.get(schema_type.lower(), UIInputType.TEXT)
    
    @classmethod
    def _get_pattern_enhancements(cls, param_name: str) -> Dict[str, Any]:
        """Apply pattern-based enhancements to parameter metadata"""
        enhancements = {}
        
        for pattern, rules in cls.PARAM_PATTERNS.items():
            if re.match(pattern, param_name, re.IGNORECASE):
                enhancements.update(rules)
                break
        
        return enhancements
    
    @classmethod
    def _generate_options(
        cls, 
        param_name: str, 
        param: inspect.Parameter,
        type_info: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate options list for select-type parameters"""
        
        # Check if type_info already has options
        if 'options' in type_info:
            return type_info['options']
        
        # Common sklearn parameter options
        common_options = {
            'criterion': [
                {'value': 'gini', 'label': 'Gini impurity'},
                {'value': 'entropy', 'label': 'Information gain'},
                {'value': 'log_loss', 'label': 'Log loss'},
            ],
            'kernel': [
                {'value': 'linear', 'label': 'Linear kernel'},
                {'value': 'poly', 'label': 'Polynomial kernel'},
                {'value': 'rbf', 'label': 'RBF kernel'},
                {'value': 'sigmoid', 'label': 'Sigmoid kernel'},
            ],
            'penalty': [
                {'value': None, 'label': 'No penalty'},
                {'value': 'l1', 'label': 'L1 (Lasso)'},
                {'value': 'l2', 'label': 'L2 (Ridge)'},
                {'value': 'elasticnet', 'label': 'ElasticNet'},
            ],
        }
        
        if param_name in common_options:
            return common_options[param_name]
        
        # Try to extract from Literal type annotation
        annotation = param.annotation
        if get_origin(annotation) is Literal:
            args = get_args(annotation)
            return [{'value': a, 'label': str(a)} for a in args]
        
        return None
    
    @classmethod
    def _parse_numpy_docstring(cls, docstring: str) -> Dict[str, str]:
        """Parse numpy-style docstring to extract parameter descriptions"""
        descriptions = {}
        if not docstring:
            return descriptions
        
        lines = docstring.split('\n')
        in_params = False
        current_param = None
        current_desc = []
        
        for line in lines:
            stripped = line.strip()
            
            # Detect parameters section
            if stripped.lower() in ['parameters', 'args', 'arguments']:
                in_params = True
                continue
            
            # End of parameters section
            if in_params and stripped and not line.startswith(' ') and ':' not in stripped:
                if current_param:
                    descriptions[current_param] = ' '.join(current_desc).strip()
                break
            
            if in_params:
                # Parameter definition: "param_name : type"
                if ':' in stripped and not stripped.startswith('---'):
                    if current_param and current_desc:
                        descriptions[current_param] = ' '.join(current_desc).strip()
                    
                    parts = stripped.split(':', 1)
                    current_param = parts[0].strip().split()[0]  # Handle "param_name :"
                    current_desc = [parts[1].strip()] if len(parts) > 1 else []
                elif current_param and (stripped or line.startswith(' ')):
                    # Continuation of description
                    current_desc.append(stripped)
        
        # Don't forget the last parameter
        if current_param and current_desc:
            descriptions[current_param] = ' '.join(current_desc).strip()
        
        return descriptions
    
    @classmethod
    def _extract_class_description(cls, docstring: str) -> str:
        """Extract the first paragraph of docstring as class description"""
        if not docstring:
            return ""
        return docstring.split('\n\n')[0].strip()
    
    @classmethod
    def _parse_returns(cls, docstring: str) -> Optional[str]:
        """Parse returns section from docstring"""
        if not docstring:
            return None
        
        lines = docstring.split('\n')
        in_returns = False
        
        for i, line in enumerate(lines):
            if line.strip().lower() == 'returns':
                in_returns = True
                continue
            if in_returns:
                if line.strip() and not line.startswith(' '):
                    break
                if line.strip():
                    return line.strip()
        return None
    
    @classmethod
    def _infer_category(cls, target_class: type) -> str:
        """Infer estimator category from class name/module"""
        name = target_class.__name__.lower()
        module = target_class.__module__.lower()
        
        if any(kw in name for kw in ['regressor', 'regression']):
            return 'regression'
        elif any(kw in name for kw in ['classifier', 'classification']):
            return 'classification'
        elif any(kw in name for kw in ['cluster', 'clustering']):
            return 'clustering'
        elif 'linear' in name:
            return 'linear_models'
        elif 'tree' in name or 'forest' in name or 'boost' in name:
            return 'ensemble'
        elif 'svm' in name or 'svc' in name or 'svr' in name:
            return 'svm'
        elif 'neural' in name or 'mlp' in name:
            return 'neural_networks'
        elif 'naive_bayes' in module or 'naive' in name:
            return 'naive_bayes'
        else:
            return 'other'
