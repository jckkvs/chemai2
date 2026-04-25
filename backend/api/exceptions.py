"""Custom exception classes"""
from fastapi import HTTPException, status

class ChemAIException(HTTPException):
    """Base exception for ChemAI API"""
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=detail)

class DataProcessingError(ChemAIException):
    """Raised when data processing fails"""
    def __init__(self, detail: str):
        super().__init__(detail, status_code=400)

class ModelExecutionError(ChemAIException):
    """Raised when model execution fails"""
    def __init__(self, detail: str):
        super().__init__(detail, status_code=500)

class ValidationError(ChemAIException):
    """Raised when validation fails"""
    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}", status_code=422)
