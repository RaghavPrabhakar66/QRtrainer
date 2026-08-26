import torch
import random
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from datasets import Dataset, load_metric
from sqlalchemy.orm import Session
from typing import List, Tuple
from .database import SessionLocal, Scan
from .storage import download_image_from_s3
from .config import settings
from .model_registry import ModelRegistry
from .jobs import update_job_status, set_job_status, JobStatus
from app.inference_service.logger import get_logger

logger = get_logger(__name__)

def ocr_data_collator(features, pad_token_id):
    pixel_values = torch.stack([torch.as_tensor(feature["pixel_values"]) for feature in features])
    labels = [torch.as_tensor(feature["labels"], dtype=torch.long) for feature in features]
    max_length = max(label.size(0) for label in labels)
    padded_labels = torch.full(
        (len(labels), max_length),
        pad_token_id,
        dtype=torch.long,
    )
    for index, label in enumerate(labels):
        padded_labels[index, :label.size(0)] = label
    padded_labels[padded_labels == pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": padded_labels}

class TrainingPipeline:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.processor = TrOCRProcessor.from_pretrained(settings.OCR_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cer_metric = load_metric("cer")
        self.wer_metric = load_metric("wer")

    def _update_status(self, stage: str, progress: float, message: str = ""):
        update_job_status(self.job_id, stage, progress, message)

    def fetch_new_data(self, db: Session, limit: int = 10000) -> List[Scan]:
        scans = db.query(Scan).filter(
            Scan.ground_truth.isnot(None),
            Scan.ground_truth != "",
            Scan.used_for_training == False,
        ).order_by(Scan.created_at.asc()).limit(limit).all()
        logger.info(f"Fetched {len(scans)} new scans for training")
        return scans

    def prepare_dataset(self, scans: List[Scan]) -> Tuple[Dataset, Dataset]:
        self._update_status("downloading", 10, f"Downloading {len(scans)} images")
        images = []
        texts = []
        for i, scan in enumerate(scans):
            try:
                img = download_image_from_s3(scan.image_key)
                images.append(img)
                texts.append(scan.ground_truth)
            except Exception as e:
                logger.error(f"Failed to download image {scan.image_key}: {e}")
            if i % 100 == 0:
                self._update_status("downloading", 10 + 20 * (i/len(scans)), f"Downloaded {i+1}/{len(scans)}")
        if not images:
            raise ValueError("No valid images downloaded")

        indices = list(range(len(images)))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices = indices[:split]
        test_indices = indices[split:]

        def create_hf_dataset(idx_list):
            img_batch = [images[i] for i in idx_list]
            text_batch = [texts[i] for i in idx_list]
            pixel_values = self.processor(images=img_batch, return_tensors="pt").pixel_values
            labels = self.processor.tokenizer(text_batch, padding=True, return_tensors="pt").input_ids
            return Dataset.from_dict({"pixel_values": pixel_values, "labels": labels})

        train_ds = create_hf_dataset(train_indices)
        test_ds = create_hf_dataset(test_indices)
        self._update_status("prepared", 40, f"Dataset ready: {len(train_ds)} train, {len(test_ds)} test")
        return train_ds, test_ds

    def train_and_evaluate(self, train_ds: Dataset, eval_ds: Dataset) -> Tuple[VisionEncoderDecoderModel, float, float]:
        self._update_status("training", 50, "Initializing model...")
        model = VisionEncoderDecoderModel.from_pretrained(settings.OCR_MODEL_NAME)
        model.config.decoder_start_token_id = self.processor.tokenizer.eos_token_id
        model.to(self.device)
        model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        model.config.max_length = 64

        training_args = Seq2SeqTrainingArguments(
            output_dir="./training_output",
            per_device_train_batch_size=settings.TRAINING_BATCH_SIZE,
            per_device_eval_batch_size=settings.TRAINING_BATCH_SIZE,
            num_train_epochs=settings.TRAINING_EPOCHS,
            learning_rate=settings.TRAINING_LEARNING_RATE,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            predict_with_generate=True,
            generation_max_length=64,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
        )

        def compute_metrics(pred):
            labels_ids = pred.label_ids
            pred_ids = pred.predictions
            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            labels_ids = labels_ids.copy()
            labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
            labels_str = self.processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
            cer = self.cer_metric.compute(predictions=pred_str, references=labels_str)
            wer = self.wer_metric.compute(predictions=pred_str, references=labels_str)
            return {"cer": cer, "wer": wer}

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.processor.tokenizer,
            data_collator=lambda features: ocr_data_collator(
                features,
                self.processor.tokenizer.pad_token_id,
            ),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()
        self._update_status("evaluating", 90, "Evaluating best model...")
        eval_result = trainer.evaluate()
        cer = eval_result["eval_cer"]
        wer = eval_result["eval_wer"]
        logger.info(f"Training finished - CER: {cer:.4f}, WER: {wer:.4f}")
        return trainer.model, cer, wer

    def should_promote(self, new_cer: float, db: Session) -> bool:
        prod_model = ModelRegistry.get_production_model(db)
        if not prod_model:
            return True
        if prod_model.cer is None:
            return True
        improvement = (prod_model.cer - new_cer) / prod_model.cer
        logger.info(f"Current production CER: {prod_model.cer:.4f}, new CER: {new_cer:.4f}, improvement: {improvement:.2%}")
        return improvement >= settings.MIN_IMPROVEMENT_CER

    def run(self):
        db = SessionLocal()
        try:
            self._update_status("starting", 0, "Fetching new data...")
            scans = self.fetch_new_data(db)
            if len(scans) < 10:
                self._update_status("failed", 0, "Not enough data (need ≥10)")
                logger.warning("Not enough data to train. Skipping.")
                set_job_status(self.job_id, JobStatus.FAILED)
                return

            train_ds, eval_ds = self.prepare_dataset(scans)
            model, cer, wer = self.train_and_evaluate(train_ds, eval_ds)

            mv = ModelRegistry.save_model(
                db,
                model,
                self.processor,
                cer,
                wer,
                len(scans),
                f"Trained on {len(scans)} scans, CER={cer:.4f}, WER={wer:.4f}"
            )
            self._update_status("saving", 95, f"Model {mv.version} saved.")

            if self.should_promote(cer, db):
                ModelRegistry.promote_model(db, mv.id)
                self._update_status("promoted", 100, f"Model {mv.version} promoted to production.")
            else:
                self._update_status("completed", 100, f"Model {mv.version} not promoted (insufficient improvement).")

            for scan in scans:
                scan.used_for_training = True
            db.commit()
            logger.info(f"Marked {len(scans)} scans as used.")
            set_job_status(self.job_id, JobStatus.COMPLETED)
        except Exception as e:
            logger.exception("Training pipeline failed")
            self._update_status("failed", 0, str(e))
            set_job_status(self.job_id, JobStatus.FAILED)
            db.rollback()
            raise
        finally:
            db.close()