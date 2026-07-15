import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")
MODELS_DIR = ROOT_DIR / os.getenv("MODELS_DIR", "models")
DB_PATH = ROOT_DIR / os.getenv("DB_PATH", "data/smartretail.db")

DEVICE = os.getenv("DEVICE", "cpu")
SEED = int(os.getenv("SEED", 42))

IMAGE_CLASSIFIER_MODEL_PATH = MODELS_DIR / "image_classifier" / "efficientnet_b0.keras"
SENTIMENT_TFIDF_MODEL_PATH = MODELS_DIR / "sentiment_analyzer" / "tfidf_baseline.pkl"
SENTIMENT_BETO_MODEL_PATH = MODELS_DIR / "sentiment_analyzer" / "beto_finetuned"

# --- Módulo 3: predicción de ventas ---
SALES_MODELS_DIR = MODELS_DIR / "sales_predictor"
SALES_METRICS_PATH = SALES_MODELS_DIR / "metrics.json"
SALES_SENTIMENT_CSV_PATH = DATA_DIR / "raw" / "sentiment_synthetic" / "reviews_synthetic.csv"

# Subconjunto de tiendas Rossmann usado para entrenar (de 1115 totales).
SALES_STORE_SUBSET = [1, 3, 7, 15, 22, 34, 45, 58, 67, 78]


def sales_prophet_path(store_id: int) -> Path:
    return SALES_MODELS_DIR / f"prophet_store_{store_id}.joblib"


# XGBoost GLOBAL: un solo modelo entrenado con las 1115 tiendas juntas
SALES_XGBOOST_GLOBAL_MODEL_PATH = SALES_MODELS_DIR / "xgboost_global.joblib"
SALES_PROPHET_MODELS_PATH = SALES_MODELS_DIR / "prophet_models.joblib"

IMAGE_SIZE = (224, 224)
IMAGE_CLASSES = ["Accessories", "Apparel", "Footwear"]

SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
BETO_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
