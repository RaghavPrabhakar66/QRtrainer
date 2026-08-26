import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL") or "sqlite:///./inference.db"
    
    # AWS S3
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME") or ""
    
    # Model (TrOCR)
    OCR_MODEL_NAME: str = os.getenv("OCR_MODEL_NAME") or "microsoft/trocr-base-printed"
    
settings = Settings()
if not settings.DATABASE_URL:
    settings.DATABASE_URL = "sqlite:///./inference.db"