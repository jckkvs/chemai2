"""
backend/api/params.py
Model and Adapter parameter introspection API
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from backend.models.factory import list_models
from backend.ui.param_schema import introspect_params, ParamSpec
import importlib

router = APIRouter(prefix="/api/params", tags=["params"])

# SMILES Adapters definition (Synced with data_tab.py)
ADAPTERS = [
    {"key": "RDKit", "module": "backend.chem.rdkit_adapter", "class": "RDKitAdapter"},
    {"key": "Mordred", "module": "backend.chem.mordred_adapter", "class": "MordredAdapter"},
    {"key": "GroupContrib", "module": "backend.chem.group_contrib_adapter", "class": "GroupContribAdapter"},
    {"key": "DescriptaStorus", "module": "backend.chem.descriptastorus_adapter", "class": "DescriptaStorusAdapter"},
    {"key": "MolAI", "module": "backend.chem.molai_adapter", "class": "MolAIAdapter"},
    {"key": "scikit-FP", "module": "backend.chem.skfp_adapter", "class": "SkfpAdapter"},
    {"key": "UMA", "module": "backend.chem.uma_adapter", "class": "UMAAdapter"},
    {"key": "Mol2Vec", "module": "backend.chem.mol2vec_adapter", "class": "Mol2VecAdapter"},
    {"key": "PaDEL", "module": "backend.chem.padel_adapter", "class": "PaDELAdapter"},
    {"key": "Molfeat", "module": "backend.chem.molfeat_adapter", "class": "MolfeatAdapter"},
    {"key": "XTB", "module": "backend.chem.xtb_adapter", "class": "XTBAdapter"},
    {"key": "UniPKa", "module": "backend.chem.unipka_adapter", "class": "UniPkaAdapter"},
    {"key": "COSMO-RS", "module": "backend.chem.cosmo_adapter", "class": "CosmoAdapter"},
    {"key": "Chemprop", "module": "backend.chem.chemprop_adapter", "class": "ChempropAdapter"},
]

@router.get("/models")
async def get_models(task: str = "regression"):
    """List available models for a given task"""
    models = list_models(task=task, available_only=True)
    return models

@router.get("/models/{model_key}/schema")
async def get_model_schema(model_key: str, task: str = "regression"):
    """Get dynamic parameter schema for a specific model"""
    models = list_models(task=task, available_only=False)
    model_info = next((m for m in models if m["key"] == model_key), None)
    
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    
    model_cls = model_info.get("class")
    if not model_cls:
        # Some models use factories
        factory = model_info.get("factory")
        if factory:
            # We can't easily introspect a factory function's result class without calling it
            # But usually we can infer from the factory or just use a default instance
            try:
                # Try to get the class from factory if possible, or just call it with no args
                instance = factory()
                model_cls = type(instance)
            except:
                raise HTTPException(status_code=400, detail="Could not introspect factory-based model")
        else:
            raise HTTPException(status_code=400, detail="Model info missing class or factory")
            
    specs = introspect_params(model_cls)
    return [spec.to_dict() for spec in specs]

@router.get("/adapters")
async def get_adapters():
    """List available SMILES feature adapters"""
    return ADAPTERS

@router.get("/adapters/{adapter_key}/schema")
async def get_adapter_schema(adapter_key: str):
    """Get dynamic parameter schema for a specific SMILES adapter"""
    adapter_info = next((a for a in ADAPTERS if a["key"] == adapter_key), None)
    
    if not adapter_info:
        raise HTTPException(status_code=404, detail=f"Adapter {adapter_key} not found")
    
    try:
        module = importlib.import_module(adapter_info["module"])
        adapter_cls = getattr(module, adapter_info["class"])
        specs = introspect_params(adapter_cls)
        return [spec.to_dict() for spec in specs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to introspect adapter: {str(e)}")
