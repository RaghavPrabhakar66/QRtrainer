import torch
from PIL import Image
from .config import settings

class TrOCRInference:
    def __init__(self, model_name: str):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        # Set generation parameters (optional)
        self.model.config.decoder_start_token_id = self.processor.tokenizer.eos_token_id

    def extract_text(self, image: Image.Image) -> str:
        """Extract text from a PIL image using TrOCR."""
        # Preprocess image
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, max_length=64)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

# Global instance (initialized on startup)
ocr_model = None

def load_ocr_model():
    global ocr_model
    ocr_model = TrOCRInference(settings.OCR_MODEL_NAME)

def get_ocr_model():
    if ocr_model is None:
        load_ocr_model()
    return ocr_model