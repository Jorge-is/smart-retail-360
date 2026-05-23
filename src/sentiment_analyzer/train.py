"""
Entrenamiento de ambos modelos de sentimiento.
Diseñado para correr en Google Colab (GPU T4).
"""
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score, classification_report

from src.sentiment_analyzer.preprocess import clean_text, rating_to_label
from src.sentiment_analyzer.baseline_model import build_pipeline, save_pipeline
from src.sentiment_analyzer.beto_model import build_model_for_training
from src.utils.config import SENTIMENT_BETO_MODEL_PATH, SEED
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def train_baseline() -> None:
    logger.info("Cargando amazon_reviews_multi (es)...")
    ds = load_dataset("amazon_reviews_multi", "es")

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
    import torch
    from transformers import DataCollatorWithPadding

    logger.info("Cargando amazon_reviews_multi (es) para BETO...")
    ds = load_dataset("amazon_reviews_multi", "es")

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

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"f1_macro": f1_score(labels, preds, average="macro")}

    args = TrainingArguments(
        output_dir=str(SENTIMENT_BETO_MODEL_PATH),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=SEED,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(SENTIMENT_BETO_MODEL_PATH))
    tokenizer.save_pretrained(str(SENTIMENT_BETO_MODEL_PATH))
    logger.info("BETO guardado en %s", SENTIMENT_BETO_MODEL_PATH)


if __name__ == "__main__":
    train_baseline()
    train_beto()
