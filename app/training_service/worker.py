from .training import TrainingPipeline
from .jobs import set_job_status, JobStatus
from app.inference_service.logger import get_logger
import traceback

logger = get_logger(__name__)

def run_training_job(job_id: str):
    """
    RQ job function. It instantiates the pipeline and runs it.
    """
    logger.info(f"Starting training job {job_id}")
    set_job_status(job_id, JobStatus.RUNNING)
    try:
        pipeline = TrainingPipeline(job_id)
        pipeline.run()
        # The pipeline will set status to COMPLETED or FAILED internally
    except Exception as e:
        logger.exception(f"Training job {job_id} failed with exception")
        set_job_status(job_id, JobStatus.FAILED)
        # Re-raise to let RQ know the job failed (it will retry if configured)
        raise