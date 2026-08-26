from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Enum, Integer
from sqlalchemy.dialects.postgresql import UUID
from .config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



class Scan(Base):
    __tablename__ = "scans"
    id = Column(
        UUID(as_uuid=True) if not settings.DATABASE_URL.startswith("sqlite") else String(36),
        primary_key=True,
        default=uuid.uuid4 if not settings.DATABASE_URL.startswith("sqlite") else lambda: str(uuid.uuid4()),
    )
    image_key = Column(String, nullable=False)
    client_id = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    detection_success = Column(Boolean, nullable=False)
    ocr_success = Column(Boolean, nullable=True)
    predicted_text = Column(String, nullable=True)
    ground_truth = Column(String, nullable=True)
    is_corrected = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    used_for_training = Column(Boolean, default=False)
    split = Column(Enum("train", "test", name="split_enum"), nullable=True)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String, unique=True, nullable=False)
    model_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    cer = Column(Float, nullable=True)
    wer = Column(Float, nullable=True)
    is_production = Column(Boolean, default=False)
    training_data_count = Column(Integer, nullable=True)
    description = Column(String, nullable=True)