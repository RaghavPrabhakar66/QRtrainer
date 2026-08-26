import uuid
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Enum
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
    image_key = Column(String, nullable=False)          # S3 key
    client_id = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    detection_success = Column(Boolean, nullable=False)  # QR decode succeeded?
    ocr_success = Column(Boolean, nullable=True)        # OCR gave a result?
    predicted_text = Column(String, nullable=True)      # text from OCR or QR if detected
    ground_truth = Column(String, nullable=True)        # final correct text (from QR or manual label)

    is_corrected = Column(Boolean, default=False)       # manually labelled
    created_at = Column(DateTime, default=datetime.utcnow)

    used_for_training = Column(Boolean, default=False)
    split = Column(Enum("train", "test", name="split_enum"), nullable=True)