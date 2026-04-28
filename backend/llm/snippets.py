# backend/llm/snippets.py

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GoldenSnippet:
    id: str
    name: str
    description: str
    code_template: str
    category: str # 'data', 'feature', 'model', 'viz'
    inputs: List[str]
    outputs: List[str]

class SnippetLibrary:
    """Library of validated MI snippets for LLM Agent usage"""
    
    def __init__(self):
        self.snippets: Dict[str, GoldenSnippet] = {}
        self._load_default_snippets()
    
    def _load_default_snippets(self):
        # Example Snippets
        self.add_snippet(GoldenSnippet(
            id="calc_rdkit_desc",
            name="RDKit Descriptor Calculation",
            description="Calculates standard RDKit descriptors for a list of SMILES strings.",
            code_template="""
from backend.chem.descriptors import RDKitDescriptorCalculator
calc = RDKitDescriptorCalculator()
df_features = calc.compute(df['smiles_column'])
""",
            category="feature",
            inputs=["df", "smiles_column"],
            outputs=["df_features"]
        ))
        
        self.add_snippet(GoldenSnippet(
            id="train_linear_tree",
            name="Linear Tree Model Training",
            description="Trains a Linear Tree model for regression tasks (interpretable).",
            code_template="""
from backend.models.linear_tree import LinearTreeRegressor
model = LinearTreeRegressor()
model.fit(X_train, y_train)
results = model.evaluate(X_test, y_test)
""",
            category="model",
            inputs=["X_train", "y_train", "X_test", "y_test"],
            outputs=["model", "results"]
        ))

    def add_snippet(self, snippet: GoldenSnippet):
        self.snippets[snippet.id] = snippet
        
    def get_snippet(self, snippet_id: str) -> Optional[GoldenSnippet]:
        return self.snippets.get(snippet_id)

    def get_all_summaries(self) -> List[Dict[str, str]]:
        """Return summaries for LLM prompt context"""
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "inputs": s.inputs,
                "outputs": s.outputs
            }
            for s in self.snippets.values()
        ]

class SnippetAssembler:
    """Assembles multiple snippets into a full execution plan"""
    def __init__(self, library: SnippetLibrary):
        self.library = library
        
    def assemble(self, snippet_ids: List[str]) -> str:
        """Combine selected snippets into a single runnable script"""
        full_code = "# Auto-generated MI Analysis Script\n"
        for sid in snippet_ids:
            snippet = self.library.get_snippet(sid)
            if snippet:
                full_code += f"\n# --- {snippet.name} ---\n"
                full_code += snippet.code_template
        return full_code
