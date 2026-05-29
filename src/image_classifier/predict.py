import time
import numpy as np
from PIL import Image

from src.utils.config import IMAGE_CLASSES, IMAGE_SIZE, MODELS_DIR

_MODEL_PATHS = {
    "efficientnet": MODELS_DIR / "image_classifier" / "efficientnet_b0.keras",
    "mobilenetv2": MODELS_DIR / "image_classifier" / "mobilenetv2.keras",
}

_loaded_models: dict = {}


def _get_model(model_name: str):
    if model_name not in _loaded_models:
        import tensorflow as tf
        path = _MODEL_PATHS[model_name]
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo '{model_name}' no encontrado en {path}. "
                "Entrenalo en Colab y copiá el .keras a models/image_classifier/"
            )
        _loaded_models[model_name] = tf.keras.models.load_model(str(path))
    return _loaded_models[model_name]


def predict(image: Image.Image, model_name: str = "efficientnet") -> dict:
    """
    Args:
        image: PIL.Image en cualquier modo (se convierte a RGB internamente).
        model_name: "efficientnet" | "mobilenetv2"

    Returns:
        {
            "model_used": str,
            "top_prediction": str,
            "top_3": [{"class": str, "confidence": float}, ...],
            "inference_time_ms": float
        }
    """
    if model_name not in _MODEL_PATHS:
        raise ValueError(f"model_name debe ser 'efficientnet' o 'mobilenetv2', no '{model_name}'")

    model = _get_model(model_name)

    img = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    start = time.perf_counter()
    probs = model.predict(arr, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    top3_idx = np.argsort(probs)[::-1][:min(3, len(IMAGE_CLASSES))]
    top3 = [{"class": IMAGE_CLASSES[i], "confidence": round(float(probs[i]), 4)} for i in top3_idx]

    return {
        "model_used": model_name,
        "top_prediction": top3[0]["class"],
        "top_3": top3,
        "inference_time_ms": round(elapsed_ms, 2),
    }
