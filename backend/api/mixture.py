"""
backend/api/mixture.py
Mixture feature extraction API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/mixture", tags=["mixture"])

class MixtureComponent(BaseModel):
    smiles: str
    compound_name: Optional[str] = None
    ratio_value: float
    ratio_unit: str  # "weight", "mole", "other"

class MixtureCalculationRequest(BaseModel):
    components: List[MixtureComponent]

@router.post("/calculate")
async def calculate_mixture_features(req: MixtureCalculationRequest):
    """Calculate weighted average descriptors for a mixture of compounds"""
    if len(req.components) < 2:
        raise HTTPException(status_code=400, detail="At least 2 components are required")
    
    try:
        extractor = MixtureFeatureExtractor()
        # Convert Pydantic models to dicts for the extractor
        components_dict = [c.model_dump() for c in req.components]
        
        result = extractor.extract(components_dict)
        
        # Format result for JSON response
        return {
            "mixture_features": result.mixture_features,
            "conversion_info": result.conversion_info,
            "warnings": result.warnings
        }
    except Exception as e:
        logger.error(f"Mixture calculation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")
