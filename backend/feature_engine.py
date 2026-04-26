"""
Feature Engineering Module - chemai2/backend/feature_engine.py
Plugin-based descriptor calculation with constraint support
"""
import importlib
import inspect
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union, Literal
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from backend.core.config import settings
from backend.utils.logger import logger


@dataclass
class FeatureConstraint:
    """Constraint metadata for a feature"""
    feature_name: str
    monotonic: Optional[Literal['increasing', 'decreasing', 'either']] = None
    linearity: Optional[Literal['strong', 'weak', 'none']] = 'none'
    sigma_range: float = 3.0  # ±n sigma range for constraint enforcement
    strength: Literal['strong', 'weak'] = 'weak'  # Constraint strength
    
    def to_model_kwargs(self) -> Dict[str, Any]:
        """Convert to model-specific constraint kwargs (e.g., for XGBoost)"""
        kwargs = {}
        if self.monotonic and self.strength == 'strong':
            # Map to XGBoost monotonic_constraints format
            direction_map = {'increasing': 1, 'decreasing': -1, 'either': 0}
            kwargs['monotonic_constraints'] = direction_map.get(self.monotonic, 0)
        return kwargs


@dataclass
class FeaturePluginSpec:
    """Specification for a feature calculation plugin"""
    name: str
    module_path: str
    function_name: str = "calculate_descriptors"
    description: str = ""
    category: Literal['rdkit', 'xtb', 'cosmo', 'ml', 'custom'] = 'custom'
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommended_for: List[str] = field(default_factory=list)
    compute_cost: Literal['low', 'medium', 'high'] = 'medium'
    requires_3d: bool = False
    
    def load_function(self) -> Optional[Callable]:
        """Dynamically load the feature calculation function"""
        try:
            module = importlib.import_module(self.module_path)
            func = getattr(module, self.function_name)
            
            # Validate function signature
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if 'smiles_list' not in params:
                logger.warning(f"Plugin {self.name} missing required 'smiles_list' parameter")
                return None
            
            return func
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to load plugin {self.name} from {self.module_path}: {e}")
            return None


class FeaturePluginRegistry:
    """Registry for feature calculation plugins with auto-discovery"""
    
    def __init__(self, plugin_dir: Path = None):
        self.plugin_dir = plugin_dir or settings.BASE_DIR / "backend" / "chem" / "plugins"
        self._plugins: Dict[str, FeaturePluginSpec] = {}
        self._loaded_functions: Dict[str, Callable] = {}
        self._discover_plugins()
    
    def _discover_plugins(self):
        """Auto-discover plugin files matching pattern *_descriptor.py or using the registry"""
        if not self.plugin_dir.exists():
            self.plugin_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Search for files ending in _descriptors.py or specified by convention
        for py_file in self.plugin_dir.glob("*.py"):
            if py_file.name == "__init__.py": continue
            try:
                spec = self._parse_plugin_file(py_file)
                if spec:
                    self._plugins[spec.name] = spec
                    logger.info(f"Registered plugin: {spec.name}")
            except Exception as e:
                logger.warning(f"Failed to parse plugin {py_file}: {e}")
    
    def _parse_plugin_file(self, file_path: Path) -> Optional[FeaturePluginSpec]:
        """Parse plugin file metadata from docstring and attributes"""
        module_name = file_path.stem
        # Adjust module path based on relative position to app
        # Assuming backend.chem.plugins structure
        module_path = f"backend.chem.plugins.{module_name}"
        
        spec_dict = {
            'name': module_name,
            'module_path': module_path,
            'function_name': 'calculate_descriptors',
            'description': '',
            'category': 'custom',
            'default_params': {},
            'param_schema': {},
            'recommended_for': [],
            'compute_cost': 'medium',
            'requires_3d': False,
        }
        
        try:
            # Check if it can be imported
            module = importlib.import_module(module_path)
            
            # Extract from docstring
            if module.__doc__:
                import yaml
                try:
                    # Clean the docstring to handle potential formatting issues
                    doc = module.__doc__.strip()
                    if doc.startswith('---'):
                        doc = doc.split('---')[1].split('---')[0]
                    meta = yaml.safe_load(doc)
                    if isinstance(meta, dict):
                        spec_dict.update({k: v for k, v in meta.items() if k in spec_dict})
                except Exception:
                    pass
            
            # Extract from PLUGIN_SPEC
            if hasattr(module, 'PLUGIN_SPEC') and isinstance(module.PLUGIN_SPEC, dict):
                spec_dict.update(module.PLUGIN_SPEC)
            
            return FeaturePluginSpec(**spec_dict)
        except Exception as e:
            logger.debug(f"Could not parse metadata for {file_path}: {e}")
            return None
    
    def register_plugin(self, spec: FeaturePluginSpec):
        """Manually register a plugin"""
        self._plugins[spec.name] = spec
        logger.info(f"Manually registered plugin: {spec.name}")
    
    def get_plugin(self, name: str) -> Optional[FeaturePluginSpec]:
        return self._plugins.get(name)
    
    def list_plugins(self, category: str = None, available_only: bool = True) -> List[FeaturePluginSpec]:
        plugins = list(self._plugins.values())
        if category:
            plugins = [p for p in plugins if p.category == category]
        if available_only:
            plugins = [p for p in plugins if p.load_function() is not None]
        return plugins
    
    def get_recommended_plugins(self, target_property: str, max_cost: str = 'high') -> List[FeaturePluginSpec]:
        recommended = []
        for plugin in self._plugins.values():
            if any(target_property.lower() in r.lower() for r in plugin.recommended_for):
                if self._cost_allowed(plugin.compute_cost, max_cost):
                    recommended.append(plugin)
        cost_order = {'low': 0, 'medium': 1, 'high': 2}
        return sorted(recommended, key=lambda p: cost_order.get(p.compute_cost, 3))
    
    def _cost_allowed(self, plugin_cost: str, max_cost: str) -> bool:
        cost_order = {'low': 0, 'medium': 1, 'high': 2}
        return cost_order.get(plugin_cost, 3) <= cost_order.get(max_cost, 3)
    
    def calculate_features(
        self,
        plugin_name: str,
        smiles_list: List[str],
        params: Dict[str, Any] = None,
        constraints: List[FeatureConstraint] = None
    ) -> pd.DataFrame:
        spec = self.get_plugin(plugin_name)
        if not spec:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        func = spec.load_function()
        if not func:
            raise RuntimeError(f"Failed to load function for plugin: {plugin_name}")
        
        calc_params = {**spec.default_params, **(params or {})}
        
        try:
            result = func(smiles_list=smiles_list, **calc_params)
            
            if isinstance(result, pd.DataFrame):
                df = result
            elif isinstance(result, (list, np.ndarray)):
                df = pd.DataFrame(result, columns=[f"{plugin_name}_{i}" for i in range(len(result[0]) if len(result) > 0 else 0)])
            else:
                raise ValueError(f"Unexpected return type: {type(result)}")
            
            if constraints:
                df.attrs['constraints'] = {c.feature_name: c for c in constraints}
            
            return df
        except Exception as e:
            logger.error(f"Feature calculation failed for {plugin_name}: {e}")
            warnings.warn(f"Plugin {plugin_name} failed: {e}")
            return pd.DataFrame()
