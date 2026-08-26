import io
from pathlib import Path
import boto3
from PIL import Image
from .config import settings

s3_client = None

def get_s3_client():
    global s3_client
    if s3_client is None:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION or "us-east-1",
        )
    return s3_client

def upload_image_to_s3(image: Image.Image, key: str) -> str:
    """Uploads a PIL image to S3 and returns the S3 key."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    if not settings.S3_BUCKET_NAME:
        local_path = Path("uploads") / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(buffer.getvalue())
        return key

    get_s3_client().upload_fileobj(buffer, settings.S3_BUCKET_NAME, key, ExtraArgs={"ContentType": "image/jpeg"})
    return key

def download_image_from_s3(key: str) -> Image.Image:
    """Downloads an image from S3 and returns a PIL Image."""
    response = get_s3_client().get_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
    image_data = response['Body'].read()
    return Image.open(io.BytesIO(image_data))