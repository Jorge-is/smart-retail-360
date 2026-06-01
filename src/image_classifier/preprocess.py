import numpy as np
from PIL import Image

from src.utils.config import IMAGE_SIZE


def preprocess_for_training(image: Image.Image) -> np.ndarray:
    """Devuelve ndarray (H, W, 3) float32 con augmentation ligera."""
    img = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    if np.random.random() > 0.5:
        arr = arr[:, ::-1, :]  # horizontal flip
    return arr


def preprocess_for_inference(image: Image.Image) -> np.ndarray:
    """Devuelve ndarray (H, W, 3) float32 listo para inferencia."""
    img = image.convert("RGB").resize(IMAGE_SIZE)
    return np.array(img, dtype=np.float32)
