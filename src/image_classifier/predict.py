import time
import torch
from PIL import Image

from src.image_classifier.model import load_model
from src.image_classifier.preprocess import preprocess_for_inference
from src.utils.config import IMAGE_CLASSIFIER_MODEL_PATH, IMAGE_CLASSES, DEVICE

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model(IMAGE_CLASSIFIER_MODEL_PATH, num_classes=len(IMAGE_CLASSES), device=DEVICE)
    return _model


def predict(image: Image.Image) -> dict:
    """
    Args:
        image: PIL.Image en cualquier modo (se convierte a RGB internamente).

    Returns:
        {
            "top_prediction": str,
            "top_3": [{"class": str, "confidence": float}, ...],
            "inference_time_ms": float
        }
    """
    model = _get_model()
    device = torch.device(DEVICE)

    tensor = preprocess_for_inference(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    probs = torch.softmax(logits, dim=1).squeeze()
    top3_idx = probs.topk(min(3, len(IMAGE_CLASSES))).indices.tolist()

    top3 = [{"class": IMAGE_CLASSES[i], "confidence": round(probs[i].item(), 4)} for i in top3_idx]

    return {
        "top_prediction": top3[0]["class"],
        "top_3": top3,
        "inference_time_ms": round(elapsed_ms, 2),
    }
