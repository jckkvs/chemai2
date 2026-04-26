"""
Safe Code Execution Sandbox - chemai2/backend/ai/sandbox.py
AST-based security validation and restricted execution for LLM-generated code
"""
import ast
import sys
import threading
import traceback
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
import warnings

from restrictedpython import compile_restricted, safe_globals, utility_builtins
from restrictedpython.Guards import guarded_iter_unpack_sequence, guarded_unpack_sequence
from backend.utils.logger import logger


@dataclass
class SandboxConfig:
    """Configuration for code execution sandbox"""
    max_execution_time: float = 30.0  # seconds
    max_memory_mb: int = 512
    allowed_imports: Set[str] = field(default_factory=lambda: {
        'math', 'statistics', 'collections', 'itertools', 'functools',
        'operator', 're', 'json', 'datetime', 'decimal', 'fractions'
    })
    allowed_builtins: Set[str] = field(default_factory=lambda: {
        'len', 'sum', 'min', 'max', 'abs', 'round', 'sorted', 'enumerate',
        'zip', 'map', 'filter', 'range', 'list', 'tuple', 'dict', 'set',
        'str', 'int', 'float', 'bool', 'complex', 'bytes', 'bytearray',
        'None', 'True', 'False', 'Ellipsis', 'NotImplemented'
    })
    disallowed_ast_nodes: Set[type] = field(default_factory=lambda: {
        ast.Import, ast.ImportFrom, ast.Exec, ast.Call,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.Lambda, ast.GeneratorExp, ast.DictComp, ast.SetComp,
        ast.Await, ast.AsyncFor, ast.AsyncWith
    })
    disallowed_names: Set[str] = field(default_factory=lambda: {
        '__import__', 'eval', 'exec', 'compile', 'open', 'input', 'print',
        'getattr', 'setattr', 'delattr', 'vars', 'dir', 'help',
        'globals', 'locals', 'breakpoint', '__build_class__'
    })
    disallowed_attributes: Set[str] = field(default_factory=lambda: {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__code__', '__globals__', '__builtins__', '__import__',
        'system', 'popen', 'subprocess', 'urlopen', 'requests'
    })


class ASTSecurityValidator:
    """Validates Python AST for security violations before execution"""
    
    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
    
    def validate(self, source_code: str) -> tuple[bool, Optional[str]]:
        """Validate source code AST"""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            if type(node) in self.config.disallowed_ast_nodes:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue # Handled by import check
                return False, f"Disallowed AST node: {type(node).__name__}"
            
            if isinstance(node, ast.Name) and node.id in self.config.disallowed_names:
                return False, f"Disallowed name: {node.id}"
            
            if isinstance(node, ast.Attribute) and node.attr in self.config.disallowed_attributes:
                return False, f"Disallowed attribute: {node.attr}"
            
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.config.allowed_imports:
                        return False, f"Import not allowed: {alias.name}"
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module not in self.config.allowed_imports:
                    return False, f"Import from not allowed: {node.module}"
        
        return True, None


class SafeCodeExecutor:
    """Executes code in a sandboxed environment with resource limits"""
    
    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self.validator = ASTSecurityValidator(self.config)
    
    def execute(self, source_code: str, local_vars: Dict[str, Any] = None, 
                extra_globals: Dict[str, Any] = None, timeout: float = None) -> Dict[str, Any]:
        """Execute code safely with resource limits"""
        timeout = timeout or self.config.max_execution_time
        
        is_valid, error = self.validator.validate(source_code)
        if not is_valid:
            return {'success': False, 'error': f"Security validation failed: {error}"}
        
        try:
            byte_code = compile_restricted(source_code, filename='<llm_generated>', mode='exec')
        except Exception as e:
            return {'success': False, 'error': f"Compilation failed: {e}"}
        
        global_ns = safe_globals.copy()
        global_ns.update(utility_builtins)
        global_ns.update({
            '__iter_unpack_sequence__': guarded_iter_unpack_sequence,
            '__unpack_sequence__': guarded_unpack_sequence,
        })
        
        for name in self.config.allowed_builtins:
            if name in __builtins__:
                global_ns[name] = __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
        
        for module in self.config.allowed_imports:
            try:
                global_ns[module] = __import__(module)
            except ImportError: pass
            
        if extra_globals:
            global_ns.update(extra_globals)
            
        local_ns = local_vars.copy() if local_vars else {}
        result_container = {'result': None, 'error': None}
        
        def run_with_limits():
            try:
                exec(byte_code, global_ns, local_ns)
                result_container['result'] = local_ns.get('calculate_descriptors')
            except Exception as e:
                result_container['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        
        thread = threading.Thread(target=run_with_limits)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            return {'success': False, 'error': f"Execution timeout: exceeded {timeout}s limit"}
        
        if result_container['error']:
            return {'success': False, 'error': result_container['error']}
        
        if result_container['result'] is None:
            return {'success': False, 'error': "Expected function 'calculate_descriptors' not found"}
        
        return {'success': True, 'result': result_container['result']}

    def test_function(self, func: Callable, test_smiles: List[str], expected_columns: List[str] = None) -> Dict[str, Any]:
        """Test the generated descriptor function with sample data"""
        import pandas as pd
        try:
            result = func(smiles_list=test_smiles)
            if not isinstance(result, pd.DataFrame):
                return {'valid': False, 'error': "Return type must be pandas.DataFrame"}
            if len(result) != len(test_smiles):
                return {'valid': False, 'error': f"Row count mismatch: {len(result)} vs {len(test_smiles)}"}
            return {'valid': True, 'columns': list(result.columns)}
        except Exception as e:
            return {'valid': False, 'error': f"Test execution failed: {e}"}
