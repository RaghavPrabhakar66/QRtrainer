import json
import uuid
from datetime import datetime
from redis import Redis
from .config import settings

redis_client = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)

class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

def create_job() -> str:
    job_id = str(uuid.uuid4())
    data = {
        "status": JobStatus.PENDING,
        "stage": "initializing",
        "progress": 0.0,
        "message": "Job created",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    redis_client.set(f"job:{job_id}", json.dumps(data))
    return job_id

def update_job_status(job_id: str, stage: str, progress: float, message: str):
    key = f"job:{job_id}"
    data = redis_client.get(key)
    if data:
        job_data = json.loads(data)
        job_data.update({
            "stage": stage,
            "progress": progress,
            "message": message,
            "updated_at": datetime.utcnow().isoformat(),
        })
        redis_client.set(key, json.dumps(job_data))

def set_job_status(job_id: str, status: str):
    key = f"job:{job_id}"
    data = redis_client.get(key)
    if data:
        job_data = json.loads(data)
        job_data["status"] = status
        job_data["updated_at"] = datetime.utcnow().isoformat()
        redis_client.set(key, json.dumps(job_data))

def get_job(job_id: str):
    data = redis_client.get(f"job:{job_id}")
    if data:
        return json.loads(data)
    return None