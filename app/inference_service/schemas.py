from pydantic import BaseModel, Field
from typing import Optional
import uuid

class ScanRequest(BaseModel):
    client_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class ScanResponse(BaseModel):
    scan_id: uuid.UUID
    detection_success: bool
    ocr_success: Optional[bool] = None
    extracted_text: Optional[str] = None

class LabelRequest(BaseModel):
    corrected_text: str