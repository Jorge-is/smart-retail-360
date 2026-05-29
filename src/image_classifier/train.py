"""
Script de entrenamiento — diseñado para correr en Google Colab con GPU T4.
Uso:
    python -m src.image_classifier.train --model efficientnet
    python -m src.image_classifier.train --model mobilenetv2
"""
import argparse
import tensorflow as tf

from src.utils.config import DATA_DIR, MODELS_DIR, IMAGE_SIZE, SEED, IMAGE_CLASSES
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

MODEL_SAVE_PATHS = {
    "efficientnet": MODELS_DIR / "image_classifier" / "efficientnet_b0.keras",
    "mobilenetv2": MODELS_DIR / "image_classifier" / "mobilenetv2.keras",
}


def _build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomBrightness(0.2),
    ], name="augmentation")


def train(
    model_name: str = "efficientnet",
    data_dir=DATA_DIR / "processed" / "fashion_products",
    epochs_head: int = 10,
    epochs_full: int = 10,
    lr_head: float = 1e-3,
    lr_full: float = 1e-4,
    batch_size: int = 32,
) -> None:
    if model_name == "efficientnet":
        from src.image_classifier.model_efficientnet import build_model
    elif model_name == "mobilenetv2":
        from src.image_classifier.model_mobilenetv2 import build_model
    else:
        raise ValueError(f"model_name debe ser 'efficientnet' o 'mobilenetv2', no '{model_name}'")

    tf.random.set_seed(SEED)
    num_classes = len(IMAGE_CLASSES)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir / "train"),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir / "val"),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        seed=SEED,
    )

    augment = _build_augmentation()
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Fase 1 — solo la cabeza
    logger.info("Fase 1: entrenando cabeza clasificadora (%d épocas)", epochs_head)
    model = build_model(num_classes=num_classes, freeze_backbone=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_head),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_head)

    # Fase 2 — fine-tuning completo
    logger.info("Fase 2: fine-tuning completo (%d épocas)", epochs_full)
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_full),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_full)

    save_path = MODEL_SAVE_PATHS[model_name]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info("Modelo '%s' guardado en %s", model_name, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["efficientnet", "mobilenetv2"], default="efficientnet",
                        help="Arquitectura a entrenar")
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-full", type=int, default=10)
    args = parser.parse_args()
    train(model_name=args.model, epochs_head=args.epochs_head, epochs_full=args.epochs_full)
