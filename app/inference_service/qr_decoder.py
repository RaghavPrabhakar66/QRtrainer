from PIL import Image

def decode_qr(image: Image.Image) -> str | None:
    """Attempts to decode a QR code from a PIL image. Returns the decoded string or None."""
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None

        image_array = np.array(image)
        value, _, _ = cv2.QRCodeDetector().detectAndDecode(image_array)
        return value or None

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    decoded_objects = decode(image)
    for obj in decoded_objects:
        if obj.type == "QRCODE":
            return obj.data.decode("utf-8")
    return None