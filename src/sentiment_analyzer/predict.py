from src.sentiment_analyzer.preprocess import clean_text
from src.utils.config import SENTIMENT_LABELS, DEVICE

_mode = "beto"  # default — se puede cambiar con set_mode()
_beto_model = None
_beto_tokenizer = None
_rf_pipeline = None


def set_mode(mode: str) -> None:
    """Cambia el modelo activo sin recargar ambos. Compatibilidad con código previo."""
    global _mode
    # Acepta "tfidf" como alias legacy de "random_forest"
    if mode == "tfidf":
        mode = "random_forest"
    assert mode in ("beto", "random_forest"), "mode debe ser 'beto' o 'random_forest'"
    _mode = mode


def _get_beto():
    global _beto_model, _beto_tokenizer
    if _beto_model is None:
        from src.sentiment_analyzer.beto_model import get_model_and_tokenizer
        _beto_model, _beto_tokenizer = get_model_and_tokenizer()
    return _beto_model, _beto_tokenizer


def _get_rf():
    global _rf_pipeline
    if _rf_pipeline is None:
        from src.sentiment_analyzer.baseline_model import load_pipeline
        _rf_pipeline = load_pipeline()
    return _rf_pipeline


def predict(text: str, model_name: str = "beto") -> dict:
    """
    Args:
        text: Reseña en texto libre (cualquier longitud).
        model_name: "beto" | "random_forest"

    Returns:
        {
            "model_used": str,
            "sentiment": "positive" | "neutral" | "negative",
            "confidence": float,
            "scores": {"positive": float, "neutral": float, "negative": float}
        }
    """
    cleaned = clean_text(text)

    if model_name == "beto":
        result = _predict_beto(cleaned)
    elif model_name == "random_forest":
        result = _predict_rf(cleaned)
    else:
        raise ValueError(f"model_name debe ser 'beto' o 'random_forest', no '{model_name}'")

    result["model_used"] = model_name
    return result


def _predict_beto(text: str) -> dict:
    import torch
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


def _predict_rf(text: str) -> dict:
    pipeline = _get_rf()
    probs = pipeline.predict_proba([text])[0]
    label_idx = probs.argmax()
    scores = {SENTIMENT_LABELS[i]: round(float(probs[i]), 4) for i in range(3)}
    return {
        "sentiment": SENTIMENT_LABELS[label_idx],
        "confidence": round(float(probs[label_idx]), 4),
        "scores": scores,
    }
