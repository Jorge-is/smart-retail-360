"""EfficientNet-B0 — wrapper TF/Keras (delega a model_efficientnet)."""
from tensorflow import keras

from src.utils.config import IMAGE_CLASSES


def build_model(num_classes: int = len(IMAGE_CLASSES), freeze_backbone: bool = True) -> keras.Model:
    from src.image_classifier.model_efficientnet import build_model as _build
    return _build(num_classes=num_classes, freeze_backbone=freeze_backbone)


def load_model(path: str) -> keras.Model:
    return keras.models.load_model(path)
