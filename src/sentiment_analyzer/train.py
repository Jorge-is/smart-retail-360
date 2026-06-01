"""
Entrenamiento de ambos modelos de sentimiento.
Diseñado para correr en Google Colab (GPU T4).
"""
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, classification_report

from src.sentiment_analyzer.preprocess import clean_text, rating_to_label
from src.sentiment_analyzer.baseline_model import build_pipeline, save_pipeline
from src.sentiment_analyzer.beto_model import build_model_for_training
from src.utils.config import SENTIMENT_BETO_MODEL_PATH, SEED
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# amazon_reviews_multi usa un script legacy incompatible con datasets >= 4.0.
# mteb/amazon_reviews_multi es el mismo dataset en formato parquet nativo.
# Renombramos aquí para que el resto del script no cambie.
def _load_dataset():
    raw = load_dataset("mteb/amazon_reviews_multi", "es")
    return raw.rename_columns({"text": "review_body"}).map(
        lambda batch: {"stars": [l + 1 for l in batch["label"]]},
        batched=True,
        remove_columns=["label", "id"],
    )


def train_baseline() -> None:
    logger.info("Cargando amazon_reviews_multi (es) via parquet...")
    ds = _load_dataset()

    texts_train = [clean_text(r["review_body"]) for r in ds["train"]]
    labels_train = [rating_to_label(r["stars"]) for r in ds["train"]]
    texts_test = [clean_text(r["review_body"]) for r in ds["test"]]
    labels_test = [rating_to_label(r["stars"]) for r in ds["test"]]

    pipeline = build_pipeline()
    pipeline.fit(texts_train, labels_train)

    preds = pipeline.predict(texts_test)
    logger.info("TF-IDF F1 macro: %.4f", f1_score(labels_test, preds, average="macro"))
    logger.info(classification_report(labels_test, preds, target_names=["negative", "neutral", "positive"]))

    save_pipeline(pipeline)
    logger.info("Pipeline guardado en %s", SENTIMENT_BETO_MODEL_PATH.parent / "tfidf_baseline.pkl")


def train_beto(epochs: int = 3, batch_size: int = 16) -> None:
    import tensorflow as tf
    from transformers import DataCollatorWithPadding

    logger.info("Cargando amazon_reviews_multi (es) via parquet para BETO...")
    ds = _load_dataset()

    model, tokenizer = build_model_for_training(num_labels=3)

    def tokenize(batch):
        return tokenizer(
            [clean_text(t) for t in batch["review_body"]],
            truncation=True, max_length=256,
        )

    def add_labels(batch):
        batch["labels"] = [rating_to_label(s) for s in batch["stars"]]
        return batch

    ds = ds.map(tokenize, batched=True).map(add_labels, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    tf_train = ds["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    tf_val = ds["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"],
        shuffle=False,
        batch_size=batch_size * 2,
        collate_fn=data_collator,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(tf_train, validation_data=tf_val, epochs=epochs)

    model.save_pretrained(str(SENTIMENT_BETO_MODEL_PATH))
    tokenizer.save_pretrained(str(SENTIMENT_BETO_MODEL_PATH))
    logger.info("BETO guardado en %s", SENTIMENT_BETO_MODEL_PATH)


if __name__ == "__main__":
    train_baseline()
    train_beto()
