import re
import unicodedata


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\sáéíóúüñ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def rating_to_label(rating: int) -> int:
    """Mapea rating 1-5 a etiqueta 0=negativo, 1=neutro, 2=positivo."""
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2
