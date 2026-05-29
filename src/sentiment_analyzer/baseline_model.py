"""TF-IDF + Random Forest — baseline clásico."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.utils.config import SENTIMENT_TFIDF_MODEL_PATH


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)),
    ])


def save_pipeline(pipeline: Pipeline) -> None:
    SENTIMENT_TFIDF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, SENTIMENT_TFIDF_MODEL_PATH)


def load_pipeline() -> Pipeline:
    return joblib.load(SENTIMENT_TFIDF_MODEL_PATH)
