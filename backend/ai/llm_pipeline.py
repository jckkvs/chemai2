"""
LLM Feature Generation Pipeline - chemai2/backend/ai/llm_pipeline.py
End-to-end pipeline for LLM-assisted descriptor plugin generation
"""
import json
import re
import textwrap
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
from backend.chem.plugins import DescriptorPluginRegistry, DescriptorPluginSpec
from backend.ai.sandbox import SafeCodeExecutor, SandboxConfig, ASTSecurityValidator
from backend.utils.logger import logger


@dataclass
class LLMGenerationRequest:
    """Request specification for LLM-based descriptor generation"""
    target_property: str
    domain_context: str = "organic chemistry"
    output_columns: List[str] = field(default_factory=list)
    preferred_libraries: List[str] = field(default_factory=lambda: ["rdkit"])
    complexity: Literal['simple', 'moderate', 'advanced'] = 'moderate'
    include_docstring: bool = True
    include_type_hints: bool = True
    test_smiles: List[str] = field(default_factory=lambda: ["CCO", "c1ccccc1", "CC(=O)O"])
    
    def to_prompt_context(self) -> str:
        """Convert request to prompt context section"""
        context = []
        context.append(f"Target Property: {self.target_property}")
        context.append(f"Domain: {self.domain_context}")
        if self.output_columns:
            context.append(f"Required Output Columns: {', '.join(self.output_columns)}")
        if self.preferred_libraries:
            context.append(f"Preferred Libraries: {', '.join(self.preferred_libraries)}")
        context.append(f"Complexity Level: {self.complexity}")
        return "\n".join(context)


class LLMPromptTemplates:
    """Collection of prompt templates for descriptor generation"""
    
    BASE_TEMPLATE = textwrap.dedent("""\
    You are an expert chemoinformatics engineer specializing in molecular descriptor calculation.
    
    Task: Generate a Python function to calculate molecular descriptors for predicting {target_property}.
    
    {context}
    
    Function Requirements:
    1. Name: `calculate_descriptors(smiles_list: List[str], **kwargs) -> pd.DataFrame`
    2. Input: List of SMILES strings
    3. Output: pandas DataFrame with len(smiles_list) rows
    4. Handle invalid SMILES gracefully: return NaN for failed calculations
    5. Use only allowed libraries: RDKit, math, statistics, collections (NO network/file/OS calls)
    6. Include type hints and comprehensive docstring
    7. Add parameter validation and error handling
    
    Example output format:
    ```python
    import pandas as pd
    import numpy as np
    from typing import List
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    def calculate_descriptors(smiles_list: List[str], normalize: bool = True) -> pd.DataFrame:
        \"\"\"
        Calculate molecular descriptors for {target_property} prediction.
        
        Args:
            smiles_list: List of SMILES strings
            normalize: Whether to normalize output to [0, 1] range
        
        Returns:
            pd.DataFrame with descriptor columns
        \"\"\"
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append([np.nan] * 2)  # Adjust for number of descriptors
                continue
            # Calculate descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            results.append([mw, logp])
        
        df = pd.DataFrame(results, columns=["mol_weight", "logp"])
        if normalize:
            for col in df.columns:
                min_val, max_val = df[col].min(), df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        return df
    ```
    
    Return ONLY the Python code, no markdown formatting, no explanations.
    """)
    
    REFINEMENT_TEMPLATE = textwrap.dedent("""\
    The previous descriptor function had issues. Please fix and improve it.
    
    Original Code:
    {original_code}
    
    Issues Found:
    {issues}
    
    Please provide a corrected version following the same requirements as before.
    Return ONLY the corrected Python code.
    """)
    
    OPTIMIZATION_TEMPLATE = textwrap.dedent("""\
    Optimize the following descriptor function for better performance and accuracy.
    
    Current Code:
    {current_code}
    
    Optimization Goals:
    - Reduce computation time for large SMILES lists
    - Improve numerical stability
    - Add caching for repeated calculations if beneficial
    - Maintain the same output format and column names
    
    Return ONLY the optimized Python code.
    """)
    
    @classmethod
    def generate_prompt(cls, request: LLMGenerationRequest, 
                       template_type: Literal['base', 'refinement', 'optimization'] = 'base',
                       **kwargs) -> str:
        """Generate complete prompt for LLM"""
        context = request.to_prompt_context()
        
        if template_type == 'base':
            return cls.BASE_TEMPLATE.format(
                target_property=request.target_property,
                context=context
            )
        elif template_type == 'refinement':
            return cls.REFINEMENT_TEMPLATE.format(
                original_code=kwargs.get('original_code', ''),
                issues=kwargs.get('issues', '')
            )
        elif template_type == 'optimization':
            return cls.OPTIMIZATION_TEMPLATE.format(
                current_code=kwargs.get('current_code', '')
            )
        
        return ""


class LLMFeatureGenerator:
    """
    Orchestrates LLM-based descriptor plugin generation, validation, and registration
    """
    
    def __init__(
        self,
        plugin_registry: DescriptorPluginRegistry = None,
        sandbox_config: SandboxConfig = None,
        llm_client: Any = None  # OpenAI/Anthropic client or mock
    ):
        self.registry = plugin_registry or DescriptorPluginRegistry()
        self.sandbox = SafeCodeExecutor(sandbox_config)
        self.llm_client = llm_client
        self._generation_history: List[Dict[str, Any]] = []
    
    def generate_code(
        self,
        request: LLMGenerationRequest,
        max_attempts: int = 3,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generate descriptor code using LLM
        
        Returns generation result with metadata
        """
        if not self.llm_client:
            # Mock generation for testing
            return self._mock_generate(request)
        
        prompt = LLMPromptTemplates.generate_prompt(request)
        
        for attempt in range(max_attempts):
            try:
                # Call LLM API (implementation depends on client)
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2000
                )
                
                code = response.choices[0].message.content.strip()
                # Remove markdown code fences if present
                code = re.sub(r'^```python\s*', '', code)
                code = re.sub(r'\s*```$', '', code)
                
                # Validate and test
                exec_result = self.sandbox.execute(code)
                if not exec_result['success']:
                    if attempt < max_attempts - 1:
                        # Refine prompt with error feedback
                        prompt = LLMPromptTemplates.generate_prompt(
                            request, 'refinement',
                            original_code=code,
                            issues=exec_result['error']
                        )
                        continue
                    return {
                        'success': False,
                        'error': f"Code execution failed: {exec_result['error']}",
                        'code': code,
                        'attempt': attempt + 1
                    }
                
                # Test the function
                func = exec_result['result']
                test_result = self.sandbox.test_function(
                    func, request.test_smiles, request.output_columns
                )
                
                if not test_result['valid']:
                    if attempt < max_attempts - 1:
                        prompt = LLMPromptTemplates.generate_prompt(
                            request, 'refinement',
                            original_code=code,
                            issues=test_result['error']
                        )
                        continue
                    return {
                        'success': False,
                        'error': f"Function test failed: {test_result['error']}",
                        'code': code,
                        'attempt': attempt + 1
                    }
                
                # Success
                return {
                    'success': True,
                    'code': code,
                    'function': func,
                    'test_result': test_result,
                    'attempt': attempt + 1,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"LLM generation attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    return {
                        'success': False,
                        'error': f"LLM API error: {e}",
                        'attempt': attempt + 1
                    }
        
        return {'success': False, 'error': 'Max attempts reached'}
    
    def _mock_generate(self, request: LLMGenerationRequest) -> Dict[str, Any]:
        """Mock generation for testing without LLM API"""
        # Generate deterministic template-based code
        target = request.target_property
        cols = request.output_columns or ["desc_1", "desc_2"]
        col_defs = ", ".join(f'"{c}"' for c in cols)
        
        code = textwrap.dedent(f"""\
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
                    results.append([np.nan] * {len(cols)})
                    continue
                # Example descriptors - replace with property-specific logic
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                results.append([mw, logp, tpsa][:{len(cols)}])
            
            df = pd.DataFrame(results, columns=[{col_defs}])
            if normalize:
                for col in df.columns:
                    min_val, max_val = df[col].min(), df[col].max()
                    if max_val > min_val and not np.isnan(min_val):
                        df[col] = (df[col] - min_val) / (max_val - min_val)
            return df
        """)
        
        # Execute and test mock code
        exec_result = self.sandbox.execute(code)
        if exec_result['success']:
            func = exec_result['result']
            test_result = self.sandbox.test_function(func, request.test_smiles, cols)
            return {
                'success': test_result['valid'],
                'code': code,
                'function': func if test_result['valid'] else None,
                'test_result': test_result,
                'attempt': 1,
                'timestamp': datetime.now().isoformat(),
                'mock': True
            }
        
        return {'success': False, 'error': exec_result['error'], 'mock': True}
    
    def register_generated_plugin(
        self,
        code: str,
        plugin_name: str,
        request: LLMGenerationRequest,
        generation_metadata: Dict[str, Any]
    ) -> Optional[DescriptorPluginSpec]:
        """
        Register successfully generated code as a plugin
        
        Returns the registered plugin spec or None if registration failed
        """
        import tempfile
        import importlib.util
        import sys
        
        try:
            # Create unique module name
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
            module_name = f"llm_plugin_{plugin_name}_{code_hash}"
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = Path(f.name)
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Verify function exists
            if not hasattr(module, 'calculate_descriptors'):
                raise AttributeError("Generated code missing calculate_descriptors function")
            
            # Create plugin spec
            plugin_spec = DescriptorPluginSpec(
                name=f"llm_{plugin_name}",
                module_path=module_name,
                function_name='calculate_descriptors',
                display_name=f"LLM: {plugin_name}",
                description=f"Auto-generated for {request.target_property}",
                category='llm_generated',
                tags=[request.target_property, 'llm', 'auto'],
                param_schema={
                    'normalize': {
                        'type': 'boolean',
                        'default': True,
                        'description': 'Normalize output to [0,1]'
                    }
                },
                default_params={'normalize': True},
                compute_cost='medium',
                recommended_for_properties=[request.target_property],
                output_prefix=f"llm_{plugin_name}_",
                sandboxed=True,
                # Metadata for tracking
                llm_metadata={
                    'generated_at': generation_metadata.get('timestamp'),
                    'attempts': generation_metadata.get('attempt'),
                    'request': asdict(request)
                }
            )
            
            # Register
            self.registry.register(plugin_spec)
            
            # Record in history
            self._generation_history.append({
                'plugin_name': plugin_spec.name,
                'request': asdict(request),
                'metadata': generation_metadata,
                'registered_at': datetime.now().isoformat()
            })
            
            logger.info(f"Successfully registered LLM plugin: {plugin_spec.name}")
            return plugin_spec
            
        except Exception as e:
            logger.error(f"Failed to register LLM plugin {plugin_name}: {e}")
            return None
        finally:
            # Cleanup temp file
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
    
    def get_generation_history(self, plugin_name: str = None) -> List[Dict[str, Any]]:
        """Get history of LLM generations"""
        if plugin_name:
            return [h for h in self._generation_history if h['plugin_name'] == plugin_name]
        return self._generation_history.copy()
