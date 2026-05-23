"""Wrapper para BETO fine-tuneado en clasificación de sentimiento (3 clases)."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from src.utils.config import SENTIMENT_BETO_MODEL_PATH, BETO_MODEL_NAME, DEVICE

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        path = str(SENTIMENT_BETO_MODEL_PATH)
        _tokenizer = AutoTokenizer.from_pretrained(path)
        _model = AutoModelForSequenceClassification.from_pretrained(path)
        _model.eval()
        _model.to(torch.device(DEVICE))


def get_model_and_tokenizer():
    _load()
    return _model, _tokenizer


def build_model_for_training(num_labels: int = 3):
    """Carga BETO base listo para fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(BETO_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BETO_MODEL_NAME, num_labels=num_labels)
    return model, tokenizer
