"""TF-IDF + Logistic Regression — baseline clásico."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.config import SENTIMENT_TFIDF_MODEL_PATH


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])


def save_pipeline(pipeline: Pipeline) -> None:
    SENTIMENT_TFIDF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, SENTIMENT_TFIDF_MODEL_PATH)


def load_pipeline() -> Pipeline:
    return joblib.load(SENTIMENT_TFIDF_MODEL_PATH)
