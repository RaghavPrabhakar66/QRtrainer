from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from redis import Redis
from rq import Queue

from .config import settings
from .database import Base, ModelVersion, engine, get_db
from .jobs import create_job, get_job
from app.inference_service.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Redis connection and RQ queue
redis_conn = Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1)
q = Queue(connection=redis_conn)

app = FastAPI(title="Training Service", version="1.0")

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    logger.info("Training service started successfully")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
async def start_training():
    """
    Enqueue a training job.
    Returns a job_id for tracking.
    """
    try:
        job_id = create_job()
        from .worker import run_training_job
        rq_job = q.enqueue(run_training_job, job_id, job_timeout=36000)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis is unavailable: {exc}") from exc
    # Store RQ job ID in Redis as well (optional)
    redis_conn.set(f"rq_job:{job_id}", rq_job.id)
    return {"job_id": job_id, "rq_job_id": rq_job.id}

@app.get("/train/status/{job_id}")
def get_training_status(job_id: str):
    try:
        job = get_job(job_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis is unavailable: {exc}") from exc
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/models")
def list_models(db: Session = Depends(get_db)):
    models = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [
        {
            "version": m.version,
            "created_at": m.created_at,
            "cer": m.cer,
            "wer": m.wer,
            "is_production": m.is_production,
            "training_data_count": m.training_data_count,
            "description": m.description,
        }
        for m in models
    ]

@app.post("/models/promote/{version_id}")
def promote_model(version_id: int, db: Session = Depends(get_db)):
    from .model_registry import ModelRegistry

    try:
        ModelRegistry.promote_model(db, version_id)
        return {"status": "promoted", "version_id": version_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))