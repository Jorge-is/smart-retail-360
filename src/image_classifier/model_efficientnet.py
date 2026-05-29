"""EfficientNetB0 — transfer learning con TensorFlow/Keras."""
from tensorflow import keras

from src.utils.config import IMAGE_CLASSES, IMAGE_SIZE


def build_model(num_classes: int = len(IMAGE_CLASSES), freeze_backbone: bool = True) -> keras.Model:
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base.trainable = not freeze_backbone

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def load_model(path: str) -> keras.Model:
    return keras.models.load_model(path)
