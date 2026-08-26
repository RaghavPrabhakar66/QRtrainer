import os
import shutil
from typing import Optional
from sqlalchemy.orm import Session
from .database import ModelVersion
from .config import settings
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from app.inference_service.logger import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    @staticmethod
    def get_latest_version(db: Session) -> Optional[ModelVersion]:
        return db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()

    @staticmethod
    def get_production_model(db: Session) -> Optional[ModelVersion]:
        return db.query(ModelVersion).filter(ModelVersion.is_production == True).first()

    @staticmethod
    def save_model(
        db: Session,
        model: VisionEncoderDecoderModel,
        processor: TrOCRProcessor,
        cer: float,
        wer: float,
        training_count: int,
        description: str = ""
    ) -> ModelVersion:
        latest = ModelRegistry.get_latest_version(db)
        if latest:
            version_num = int(latest.version[1:]) + 1
        else:
            version_num = 1
        version = f"v{version_num}"

        model_dir = os.path.join(settings.MODEL_REGISTRY_DIR, version)
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        processor.save_pretrained(model_dir)

        mv = ModelVersion(
            version=version,
            model_path=model_dir,
            cer=cer,
            wer=wer,
            is_production=False,
            training_data_count=training_count,
            description=description,
        )
        db.add(mv)
        db.commit()
        db.refresh(mv)
        logger.info(f"Saved model version {version} at {model_dir}")
        return mv

    @staticmethod
    def promote_model(db: Session, version_id: int):
        current_prod = ModelRegistry.get_production_model(db)
        if current_prod:
            current_prod.is_production = False
            db.commit()

        new_prod = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not new_prod:
            raise ValueError("Model version not found")
        new_prod.is_production = True
        db.commit()

        # Update production symlink/folder
        prod_path = settings.PRODUCTION_MODEL_PATH
        if os.path.exists(prod_path):
            shutil.rmtree(prod_path)
        shutil.copytree(new_prod.model_path, prod_path)
        logger.info(f"Promoted model {new_prod.version} to production")