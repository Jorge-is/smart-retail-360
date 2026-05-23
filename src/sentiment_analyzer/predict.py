import torch
from src.sentiment_analyzer.preprocess import clean_text
from src.utils.config import SENTIMENT_LABELS, DEVICE

_mode = "beto"  # "beto" | "tfidf"
_beto_model = None
_beto_tokenizer = None
_tfidf_pipeline = None


def set_mode(mode: str) -> None:
    """Cambia entre 'beto' y 'tfidf' sin recargar ambos modelos."""
    global _mode
    assert mode in ("beto", "tfidf"), "mode debe ser 'beto' o 'tfidf'"
    _mode = mode


def _get_beto():
    global _beto_model, _beto_tokenizer
    if _beto_model is None:
        from src.sentiment_analyzer.beto_model import get_model_and_tokenizer
        _beto_model, _beto_tokenizer = get_model_and_tokenizer()
    return _beto_model, _beto_tokenizer


def _get_tfidf():
    global _tfidf_pipeline
    if _tfidf_pipeline is None:
        from src.sentiment_analyzer.baseline_model import load_pipeline
        _tfidf_pipeline = load_pipeline()
    return _tfidf_pipeline


def predict(text: str) -> dict:
    """
    Args:
        text: Reseña en texto libre (cualquier longitud).

    Returns:
        {
            "sentiment": "positive" | "neutral" | "negative",
            "confidence": float,
            "scores": {"positive": float, "neutral": float, "negative": float}
        }
    """
    cleaned = clean_text(text)

    if _mode == "beto":
        return _predict_beto(cleaned)
    return _predict_tfidf(cleaned)


def _predict_beto(text: str) -> dict:
    model, tokenizer = _get_beto()
    device = torch.device(DEVICE)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).squeeze()
    label_idx = probs.argmax().item()

    scores = {SENTIMENT_LABELS[i]: round(probs[i].item(), 4) for i in range(3)}
    return {
        "sentiment": SENTIMENT_LABELS[label_idx],
        "confidence": round(probs[label_idx].item(), 4),
        "scores": scores,
    }


def _predict_tfidf(text: str) -> dict:
    pipeline = _get_tfidf()
    probs = pipeline.predict_proba([text])[0]
    label_idx = probs.argmax()
    scores = {SENTIMENT_LABELS[i]: round(float(probs[i]), 4) for i in range(3)}
    return {
        "sentiment": SENTIMENT_LABELS[label_idx],
        "confidence": round(float(probs[label_idx]), 4),
        "scores": scores,
    }
