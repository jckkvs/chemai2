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
    
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_FILE_SIZE: int = 52428800
    SESSION_TTL: int = 3600
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    
    # Task Queue settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    
    model_config = {
        "env_file": ".env",
        "extra": "allow"
    }

settings = Settings()
