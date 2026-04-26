# backend/core/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ChemAI Nexus"
    API_V1_STR: str = "/api/v1"
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    EXPORT_DIR: Path = BASE_DIR / "exports"
    CACHE_DIR: Path = BASE_DIR / "cache"
    
    DEBUG: bool = True
    ALLOWED_ORIGINS: list = ["*"]
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    
    class Config:
        env_file = ".env"

settings = Settings()
