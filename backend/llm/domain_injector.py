# backend/llm/domain_injector.py

"""
Domain Knowledge Injector for LLM Prompts

Enriches prompts with chemical domain information such as RDKit properties,
SMILES standardization results, and molecular descriptors.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

class DomainInjector:
    """Injects chemical context into LLM prompts"""
    
    def inject_molecular_context(self, smiles: str) -> str:
        """Generate a description of molecular properties for a given SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return f"Invalid SMILES: {smiles}"
            
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        context = (
            f"Molecular Context for SMILES: {smiles}\n"
            f"- Molecular Weight: {mw:.2f}\n"
            f"- LogP: {logp:.2f}\n"
            f"- Hydrogen Bond Donors: {hbd}\n"
            f"- Hydrogen Bond Acceptors: {hba}\n"
        )
        return context

    def enrich_analysis_prompt(self, base_prompt: str, data: Dict[str, Any]) -> str:
        """Enrich a general analysis prompt with specific chemical metadata"""
        # Logic to extract chemical insights from 'data' and append to prompt
        chemical_context = "Machine Learning Analysis Context:\n"
        for key, val in data.items():
            if "smiles" in key.lower() or "mol" in key.lower():
                chemical_context += f"- {key}: {val}\n"
                
        return f"{base_prompt}\n\n{chemical_context}"

# Global injector instance
domain_injector = DomainInjector()
