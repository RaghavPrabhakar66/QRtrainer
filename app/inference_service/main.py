from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from PIL import Image
import uuid
import time

from .database import Base, Scan, engine, get_db
from .schemas import ScanResponse, LabelRequest
from .qr_decoder import decode_qr
from .ocr import get_ocr_model
from .storage import upload_image_to_s3
from .logger import setup_logging, get_logger

# Setup logging early
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Tap Electric Scan Service", version="1.0")

# Middleware to add request_id and log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    # Attach request_id to request.state so endpoints can access it
    request.state.request_id = request_id
    start_time = time.time()

    # Log request
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={"request_id": request_id, "method": request.method, "path": request.url.path}
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        # Add request_id to response headers
        response.headers["X-Request-ID"] = request_id
        logger.info(
            f"Request finished: status={response.status_code} time={process_time:.3f}s",
            extra={"request_id": request_id, "status_code": response.status_code, "duration": process_time}
        )
        return response
    except Exception as e:
        logger.exception(f"Request failed: {str(e)}", extra={"request_id": request_id})
        raise

@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    logger.info("Service started successfully")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/scan", response_model=ScanResponse)
async def scan_charger(
    request: Request,
    image: UploadFile = File(...),
    client_id: str = Form(None),
    latitude: float = Form(None),
    longitude: float = Form(None),
    db: Session = Depends(get_db),
):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info("Scan request received", extra={"request_id": request_id, "client_id": client_id})

    try:
        # 1. Read and validate image
        img = Image.open(image.file)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception as e:
        logger.error(f"Invalid image: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # 2. Try QR decoding
    qr_text = decode_qr(img)
    detection_success = qr_text is not None
    logger.info(f"QR detection: success={detection_success}", extra={"request_id": request_id})

    if detection_success:
        predicted_text = qr_text
        ground_truth = qr_text
        ocr_success = None
    else:
        # QR failed → run OCR
        ocr = get_ocr_model()
        predicted_text = ocr.extract_text(img)
        if predicted_text:
            ocr_success = True
            ground_truth = predicted_text
            logger.info(f"OCR succeeded, text length={len(predicted_text)}", extra={"request_id": request_id})
        else:
            ocr_success = False
            ground_truth = None
            logger.warning("OCR failed to extract any text", extra={"request_id": request_id})

    # 3. Upload image to S3
    s3_key = f"scans/{uuid.uuid4()}.jpg"
    try:
        upload_image_to_s3(img, s3_key)
        logger.info(f"Image uploaded to S3: {s3_key}", extra={"request_id": request_id})
    except Exception as e:
        logger.error(f"S3 upload failed: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Failed to store image")

    # 4. Save scan record
    scan = Scan(
        image_key=s3_key,
        client_id=client_id,
        latitude=latitude,
        longitude=longitude,
        detection_success=detection_success,
        ocr_success=ocr_success,
        predicted_text=predicted_text,
        ground_truth=ground_truth,
        is_corrected=False,
        used_for_training=False,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    logger.info(f"Scan saved with ID: {scan.id}", extra={"request_id": request_id})

    return ScanResponse(
        scan_id=scan.id,
        detection_success=detection_success,
        ocr_success=ocr_success,
        extracted_text=predicted_text,
    )

@app.post("/label/{scan_id}")
def label_scan(
    scan_id: uuid.UUID,
    payload: LabelRequest,
    db: Session = Depends(get_db),
):
    # (Existing code) – we could also add logging here if desired
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    scan.ground_truth = payload.corrected_text
    scan.is_corrected = True
    db.commit()
    logger.info(f"Scan {scan_id} manually labelled with corrected text")
    return {"status": "updated", "scan_id": str(scan_id)}