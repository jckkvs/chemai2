"""
Descriptor Plugin System - chemai2/backend/chem/plugins/__init__.py
Plugin-based molecular descriptor calculation framework
"""
import importlib
import inspect
import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Literal, Type
import hashlib

import numpy as np
import pandas as pd

from backend.core.config import settings
from backend.utils.logger import logger
from backend.chem.smiles_utils import validate_smiles_batch, standardize_smiles_batch


@dataclass
class DescriptorPluginSpec:
    """Specification for a descriptor calculation plugin"""
    # Identity
    name: str
    module_path: str  # e.g., "backend.chem.plugins.rdkit_basic"
    function_name: str = 'calculate_descriptors'  # Default function name
    
    # Metadata for UI/UX
    display_name: Optional[str] = None
    description: str = ''
    category: Literal['rdkit', 'xtb', 'cosmo', 'ml', 'quantum', 'custom'] = 'custom'
    tags: List[str] = field(default_factory=list)
    
    # Parameter schema for dynamic UI generation
    param_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_params: Dict[str, Any] = field(default_factory=dict)
    
    # Capability metadata
    requires_3d: bool = False
    requires_optimization: bool = False
    compute_cost: Literal['low', 'medium', 'high', 'very_high'] = 'medium'
    estimated_time_per_mol: float = 0.1  # seconds
    
    # Recommendation metadata
    recommended_for_properties: List[str] = field(default_factory=list)  # e.g., ['solubility', 'logP']
    recommended_for_tasks: List[str] = field(default_factory=list)  # e.g., ['regression', 'classification']
    feature_importance_prior: Dict[str, float] = field(default_factory=dict)  # Prior importance by target
    
    # Output specification
    output_prefix: str = ''  # Prefix for output column names
    returns_dataframe: bool = True  # True=DataFrame, False=2D array
    
    # Validation
    min_smiles: int = 1
    max_smiles: Optional[int] = None  # None = unlimited
    
    # Security/sandboxing
    sandboxed: bool = True  # Run in restricted environment
    
    @property
    def full_name(self) -> str:
        """Full identifier: category.name"""
        return f"{self.category}.{self.name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DescriptorPluginSpec':
        """Deserialize from dict"""
        return cls(**data)
    
    def load_function(self) -> Optional[Callable]:
        """Dynamically load the descriptor calculation function"""
        try:
            module = importlib.import_module(self.module_path)
            func = getattr(module, self.function_name)
            
            # Validate signature
            sig = inspect.signature(func)
            required_params = ['smiles_list']
            for param in required_params:
                if param not in sig.parameters:
                    logger.error(f"Plugin {self.name} missing required parameter: {param}")
                    return None
            
            return func
        except ImportError as e:
            logger.warning(f"Failed to import plugin {self.module_path}: {e}")
            return None
        except AttributeError as e:
            logger.warning(f"Function {self.function_name} not found in {self.module_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading plugin {self.name}: {e}")
            return None


class DescriptorPluginRegistry:
    """Registry for descriptor plugins with auto-discovery and management"""
    
    def __init__(self, plugin_dirs: List[Path] = None):
        self.plugin_dirs = plugin_dirs or [
            settings.BASE_DIR / 'backend' / 'chem' / 'plugins',
            settings.BASE_DIR / 'plugins' / 'descriptors',
        ]
        self._plugins: Dict[str, DescriptorPluginSpec] = {}
        self._loaded_functions: Dict[str, Callable] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}  # Simple in-memory cache
        
        # Auto-discover plugins
        self._discover_plugins()
    
    def _discover_plugins(self):
        """Auto-discover plugin files in registered directories"""
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            
            # Discover *_descriptor.py files
            for py_file in plugin_dir.rglob('*_descriptor.py'):
                try:
                    spec = self._parse_plugin_file(py_file)
                    if spec:
                        self.register(spec)
                except Exception as e:
                    logger.warning(f"Failed to parse plugin {py_file}: {e}")
            
            # Discover plugins/__init__.py with PLUGIN_SPECS
            init_file = plugin_dir / '__init__.py'
            if init_file.exists():
                try:
                    specs = self._parse_init_file(init_file)
                    for spec in specs:
                        self.register(spec)
                except Exception as e:
                    logger.warning(f"Failed to parse {init_file}: {e}")
    
    def _parse_plugin_file(self, file_path: Path) -> Optional[DescriptorPluginSpec]:
        """Parse plugin metadata from file docstring and attributes"""
        # Extract module path
        rel_path = file_path.relative_to(settings.BASE_DIR)
        module_path = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
        
        # Default spec
        spec_dict = {
            'name': file_path.stem,
            'module_path': module_path,
            'function_name': 'calculate_descriptors',
            'category': 'custom',
            'param_schema': {},
            'default_params': {},
        }
        
        # Try to import and extract metadata
        try:
            module = importlib.import_module(module_path)
            
            # Extract from PLUGIN_SPEC attribute
            if hasattr(module, 'PLUGIN_SPEC') and isinstance(module.PLUGIN_SPEC, dict):
                spec_dict.update(module.PLUGIN_SPEC)
            
            # Extract from docstring (YAML/JSON format)
            if module.__doc__:
                import yaml
                try:
                    meta = yaml.safe_load(module.__doc__.strip())
                    if isinstance(meta, dict):
                        spec_dict.update({k: v for k, v in meta.items() if k in spec_dict or k in DescriptorPluginSpec.__dataclass_fields__})
                except:
                    pass
            
            return DescriptorPluginSpec(**spec_dict)
        except Exception as e:
            logger.debug(f"Could not extract metadata from {file_path}: {e}")
            # Return minimal spec for manual registration
            return DescriptorPluginSpec(**spec_dict)
    
    def _parse_init_file(self, init_file: Path) -> List[DescriptorPluginSpec]:
        """Parse PLUGIN_SPECS list from __init__.py"""
        specs = []
        try:
            module = importlib.import_module(
                str(init_file.relative_to(settings.BASE_DIR).with_suffix('')).replace('/', '.').replace('\\', '.')
            )
            if hasattr(module, 'PLUGIN_SPECS') and isinstance(module.PLUGIN_SPECS, list):
                for spec_dict in module.PLUGIN_SPECS:
                    if isinstance(spec_dict, dict):
                        specs.append(DescriptorPluginSpec(**spec_dict))
        except Exception as e:
            logger.warning(f"Failed to parse PLUGIN_SPECS from {init_file}: {e}")
        return specs
    
    def register(self, spec: DescriptorPluginSpec):
        """Register a plugin specification"""
        self._plugins[spec.name] = spec
        logger.debug(f"Registered plugin: {spec.name} ({spec.category})")
    
    def unregister(self, name: str):
        """Unregister a plugin"""
        if name in self._plugins:
            del self._plugins[name]
            self._loaded_functions.pop(name, None)
            logger.info(f"Unregistered plugin: {name}")
    
    def get(self, name: str) -> Optional[DescriptorPluginSpec]:
        """Get plugin spec by name"""
        return self._plugins.get(name)
    
    def list(self, category: str = None, available_only: bool = True, 
             min_cost: str = None, max_cost: str = None) -> List[DescriptorPluginSpec]:
        """List plugins with optional filtering"""
        plugins = list(self._plugins.values())
        
        if category:
            plugins = [p for p in plugins if p.category == category]
        
        if available_only:
            plugins = [p for p in plugins if p.load_function() is not None]
        
        if min_cost or max_cost:
            cost_order = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
            min_val = cost_order.get(min_cost, 0) if min_cost else 0
            max_val = cost_order.get(max_cost, 3) if max_cost else 3
            plugins = [p for p in plugins if min_val <= cost_order.get(p.compute_cost, 1) <= max_val]
        
        return plugins
    
    def get_recommended(self, target_property: str = None, task_type: str = None,
                       max_cost: str = 'high', requires_3d: bool = False) -> List[DescriptorPluginSpec]:
        """Get plugins recommended for a specific use case"""
        candidates = self.list(available_only=True, max_cost=max_cost)
        
        if target_property:
            candidates = [p for p in candidates if target_property.lower() in [t.lower() for t in p.recommended_for_properties]]
        
        if task_type:
            candidates = [p for p in candidates if task_type.lower() in [t.lower() for t in p.recommended_for_tasks]]
        
        if not requires_3d:
            candidates = [p for p in candidates if not p.requires_3d]
        
        # Sort by priority: cost first, then prior importance
        cost_order = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
        return sorted(candidates, key=lambda p: (
            cost_order.get(p.compute_cost, 3),
            -p.feature_importance_prior.get(target_property, 0) if target_property else 0
        ))
    
    def calculate(
        self,
        plugin_name: str,
        smiles_list: List[str],
        params: Dict[str, Any] = None,
        use_cache: bool = True,
        validate_smiles: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Calculate descriptors using specified plugin
        
        Args:
            plugin_name: Name of registered plugin
            smiles_list: List of SMILES strings
            params: Plugin-specific parameters
            use_cache: Use result cache if available
            validate_smiles: Validate SMILES before calculation
        
        Returns:
            DataFrame with descriptors or None if failed
        """
        spec = self.get(plugin_name)
        if not spec:
            logger.error(f"Plugin not found: {plugin_name}")
            return None
        
        # Validate input
        if len(smiles_list) < spec.min_smiles:
            logger.error(f"Plugin {plugin_name} requires at least {spec.min_smiles} SMILES")
            return None
        if spec.max_smiles and len(smiles_list) > spec.max_smiles:
            logger.error(f"Plugin {plugin_name} max {spec.max_smiles} SMILES, got {len(smiles_list)}")
            return None
        
        # Validate SMILES
        if validate_smiles:
            valid_mask = validate_smiles_batch(smiles_list)
            if not any(valid_mask):
                logger.warning("No valid SMILES in input")
                return None
            if not all(valid_mask):
                logger.warning(f"{sum(~valid_mask)}/{len(valid_mask)} SMILES invalid, filtering")
                smiles_list = [s for s, v in zip(smiles_list, valid_mask) if v]
        
        # Check cache
        if use_cache:
            cache_key = self._make_cache_key(plugin_name, smiles_list, params or {})
            if cache_key in self._cache:
                logger.debug(f"Cache hit for {plugin_name}")
                cached = self._cache[cache_key]
                if isinstance(cached, pd.DataFrame):
                    return cached.copy()
        
        # Load and execute function
        func = spec.load_function()
        if not func:
            logger.error(f"Failed to load function for plugin: {plugin_name}")
            return None
        
        try:
            # Merge default params
            calc_params = {**spec.default_params, **(params or {})}
            
            # Execute calculation
            result = func(smiles_list=smiles_list, **calc_params)
            
            # Validate and normalize output
            df = self._normalize_output(result, spec, smiles_list)
            
            # Add metadata
            df.attrs['plugin'] = spec.name
            df.attrs['params'] = calc_params
            df.attrs['n_input'] = len(smiles_list)
            df.attrs['n_output'] = len(df)
            
            # Cache result
            if use_cache:
                cache_key = self._make_cache_key(plugin_name, smiles_list, calc_params)
                self._cache[cache_key] = df.copy()
                # Simple cache size limit
                if len(self._cache) > 100:
                    # Remove oldest
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
            
            return df
            
        except Exception as e:
            logger.error(f"Descriptor calculation failed for {plugin_name}: {e}", exc_info=True)
            warnings.warn(f"Plugin {plugin_name} failed: {e}")
            return None
    
    def _normalize_output(self, result: Any, spec: DescriptorPluginSpec, 
                         smiles_list: List[str]) -> pd.DataFrame:
        """Normalize plugin output to standard DataFrame format"""
        if isinstance(result, pd.DataFrame):
            df = result
        elif isinstance(result, np.ndarray):
            if result.ndim == 1:
                df = pd.DataFrame(result, columns=[f"{spec.output_prefix}{spec.name}"])
            elif result.ndim == 2:
                df = pd.DataFrame(result)
                if spec.output_prefix:
                    df.columns = [f"{spec.output_prefix}{c}" for c in df.columns]
            else:
                raise ValueError(f"Unexpected array dimensions: {result.ndim}")
        elif isinstance(result, (list, tuple)):
            if result and isinstance(result[0], (list, tuple, np.ndarray)):
                df = pd.DataFrame(result)
            else:
                df = pd.DataFrame(result, columns=[f"{spec.output_prefix}{spec.name}"])
        else:
            raise ValueError(f"Unexpected return type: {type(result)}")
        
        # Ensure index matches input
        if len(df) == len(smiles_list):
            df.index = pd.RangeIndex(len(smiles_list))
        elif len(df) < len(smiles_list):
            # Pad with NaN for failed calculations
            df = df.reindex(range(len(smiles_list)))
        
        return df
    
    def _make_cache_key(self, plugin_name: str, smiles_list: List[str], 
                       params: Dict[str, Any]) -> str:
        """Generate cache key from inputs"""
        # Hash SMILES list (order-independent for caching)
        smiles_hash = hashlib.md5(json.dumps(sorted(smiles_list)).encode()).hexdigest()[:12]
        # Hash params
        params_hash = hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return f"{plugin_name}:{smiles_hash}:{params_hash}"
    
    def clear_cache(self, plugin_name: str = None):
        """Clear descriptor calculation cache"""
        if plugin_name:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{plugin_name}:")]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared cache for plugin: {plugin_name}")
        else:
            self._cache.clear()
            logger.info("Cleared all descriptor cache")


# Global registry instance
descriptor_registry = DescriptorPluginRegistry()
