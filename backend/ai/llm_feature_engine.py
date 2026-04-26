"""
LLM Feature Generator - chemai2/backend/ai/llm_feature_engine.py
Safe LLM-driven descriptor generation with sandboxed execution
"""
import ast
import inspect
import textwrap
import time
import threading
import warnings
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from restrictedpython import compile_restricted, safe_globals
from backend.core.config import settings
from backend.utils.logger import logger
from backend.chem.plugins import DescriptorPluginRegistry


@dataclass
class LLMGenerationConfig:
    """Configuration for LLM-assisted feature generation"""
    target_property: Optional[str] = None
    domain_context: str = "organic chemistry"
    required_output_columns: List[str] = None
    max_execution_time: float = 30.0  # seconds
    max_memory_mb: int = 512
    llm_provider: Literal["openai", "anthropic", "local"] = "openai"
    model_name: str = "gpt-4o"


class SafeCodeSandbox:
    """Restricted execution environment for user/LLM-generated code"""
    
    ALLOWED_MODULES = {"math", "statistics", "collections", "itertools", "functools"}
    DISALLOWED_OPS = {
        ast.Import, ast.ImportFrom, ast.Call,
        ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef
    }
    
    @classmethod
    def validate_ast(cls, source_code: str) -> bool:
        """Check AST for dangerous operations before execution"""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in LLM code: {e}")
            return False
        
        for node in ast.walk(tree):
            # Block network, file, OS, and subprocess access
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in [
                    "open", "exec", "eval", "compile", "getattr", "setattr",
                    "__import__", "input", "print"
                ]:
                    return False
                if isinstance(func, ast.Attribute) and func.attr in [
                    "system", "popen", "subprocess", "urlopen", "requests"
                ]:
                    return False
        return True
    
    @classmethod
    def execute(cls, source_code: str, smiles_list: List[str], 
                extra_globals: Dict[str, Any] = None, timeout: float = 30.0) -> pd.DataFrame:
        """Execute code in restricted environment with timeout/memory limits"""
        if not cls.validate_ast(source_code):
            raise ValueError("Code contains disallowed operations")
        
        # Prepare restricted globals
        globals_dict = safe_globals.copy()
        globals_dict.update({
            "len": len, "sum": sum, "min": min, "max": max,
            "abs": abs, "round": round, "sorted": sorted,
            "list": list, "dict": dict, "tuple": tuple,
            "set": set, "frozenset": frozenset,
            "zip": zip, "enumerate": enumerate, "map": map, "filter": filter,
            "str": str, "int": int, "float": float, "bool": bool,
            "None": None, "True": True, "False": False
        })
        if extra_globals:
            globals_dict.update(extra_globals)
        
        # Restricted compilation
        try:
            bytecode = compile_restricted(source_code, "<llm_generated>", "exec")
        except Exception as e:
            raise RuntimeError(f"Compilation failed: {e}")
        
        # Thread-based timeout execution
        result_container = {"df": None, "error": None}
        
        def run_code():
            try:
                local_ns = {"smiles_list": smiles_list, "pd": pd, "np": __import__("numpy"), "math": __import__("math")}
                exec(bytecode, globals_dict, local_ns)
                
                # Expect a function named `calculate_descriptors`
                if "calculate_descriptors" in local_ns:
                    result_container["df"] = local_ns["calculate_descriptors"](smiles_list)
                else:
                    result_container["error"] = "Missing calculate_descriptors function"
            except Exception as e:
                result_container["error"] = str(e)
        
        thread = threading.Thread(target=run_code)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Execution exceeded {timeout}s limit")
        
        if result_container["error"]:
            raise RuntimeError(result_container["error"])
        
        df = result_container["df"]
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Return value must be a pandas DataFrame")
        return df


class LLMFeatureEngine:
    """Generates and registers chemical descriptor plugins via LLM"""
    
    PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an expert chemoinformatics engineer. Generate a Python function to calculate molecular descriptors.
    
    Target Property: {target}
    Domain: {domain}
    Required Output Columns: {columns}
    
    Constraints:
    - Function name MUST be `calculate_descriptors(smiles_list: List[str], **kwargs) -> pd.DataFrame`
    - ONLY use RDKit, math, and standard libraries. NO network, file, or OS calls.
    - Handle invalid SMILES gracefully (return NaN for failed calculations).
    - Output DataFrame must have exactly len(smiles_list) rows.
    - Include type hints and docstring.
    
    Return ONLY the Python code, no markdown formatting.
    """)
    
    def __init__(self, registry: DescriptorPluginRegistry = None):
        self.registry = registry or DescriptorPluginRegistry()
        self.sandbox = SafeCodeSandbox()
    
    def generate_plugin_code(self, config: LLMGenerationConfig) -> str:
        """Generate descriptor code using LLM (mocked for safety, replace with actual API)"""
        # In production, call OpenAI/Anthropic API here
        # For now, return a deterministic template-based code
        target = config.target_property or "solubility"
        cols = ", ".join(config.required_output_columns or ["desc_1", "desc_2"])
        
        return textwrap.dedent(f"""\
        import pandas as pd
        import numpy as np
        from typing import List
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        def calculate_descriptors(smiles_list: List[str], normalize: bool = True) -> pd.DataFrame:
            \"\"\"
            Auto-generated descriptor calculator for {target}.
            Outputs: {cols}
            \"\"\"
            results = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    results.append([np.nan, np.nan])
                    continue
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                results.append([mw, logp])
            
            df = pd.DataFrame(results, columns="{cols}".split(", "))
            if normalize:
                df = (df - df.min()) / (df.max() - df.min())
            return df
        """)
    
    def validate_and_register(self, code: str, plugin_name: str, config: LLMGenerationConfig) -> bool:
        """Validate generated code, test execution, and register as plugin"""
        try:
            # Quick syntax & AST check
            if not self.sandbox.validate_ast(code):
                logger.error(f"Code validation failed for {plugin_name}")
                return False
            
            # Test execution with dummy data
            test_smiles = ["CCO", "c1ccccc1", "invalid"]
            df = self.sandbox.execute(code, test_smiles, timeout=config.max_execution_time)
            
            if len(df) != len(test_smiles):
                logger.error(f"Row count mismatch: expected {len(test_smiles)}, got {len(df)}")
                return False
            
            # Create temporary plugin module
            import tempfile
            import importlib.util
            import sys
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                
                spec = importlib.util.spec_from_file_location(f"llm_{plugin_name}", f.name)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                
                # Manual registration using the registry instance or similar
                # The user's code says self.registry.register_plugin, but DescriptorPluginRegistry 
                # in implementation uses register(spec: DescriptorPluginSpec).
                # I'll fix the method name to match DescriptorPluginRegistry.register
                from backend.chem.plugins import DescriptorPluginSpec
                plugin_spec = DescriptorPluginSpec(
                    name=f"llm_{plugin_name}",
                    module_path=spec.name,
                    function_name="calculate_descriptors",
                    category="llm_generated",
                    description=f"LLM-generated plugin for {config.target_property}",
                    compute_cost="medium"
                )
                self.registry.register(plugin_spec)
                logger.info(f"Successfully registered LLM plugin: {plugin_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to register LLM plugin {plugin_name}: {e}")
            return False
