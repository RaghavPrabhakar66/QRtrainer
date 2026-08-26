import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL") or "sqlite:///./inference.db"
    
    # AWS S3
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION") or "us-east-1"
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME") or ""
    
    # Redis (for RQ and job status)
    REDIS_URL: str = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
    
    # OCR model base
    OCR_MODEL_NAME: str = os.getenv("OCR_MODEL_NAME", "microsoft/trocr-base-printed")
    
    # Training hyperparameters
    TRAINING_BATCH_SIZE: int = int(os.getenv("TRAINING_BATCH_SIZE", "8"))
    TRAINING_EPOCHS: int = int(os.getenv("TRAINING_EPOCHS", "3"))
    TRAINING_LEARNING_RATE: float = float(os.getenv("TRAINING_LEARNING_RATE", "2e-5"))
    MIN_IMPROVEMENT_CER: float = float(os.getenv("MIN_IMPROVEMENT_CER", "0.05"))
    
    # Model registry (shared volume or S3)
    MODEL_REGISTRY_DIR: str = os.getenv("MODEL_REGISTRY_DIR") or "./models"
    PRODUCTION_MODEL_PATH: str = os.getenv("PRODUCTION_MODEL_PATH") or "./models/production"
    
settings = Settings()