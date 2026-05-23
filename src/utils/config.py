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

IMAGE_CLASSIFIER_MODEL_PATH = MODELS_DIR / "image_classifier" / "efficientnet_b0.pt"
SENTIMENT_TFIDF_MODEL_PATH = MODELS_DIR / "sentiment_analyzer" / "tfidf_baseline.pkl"
SENTIMENT_BETO_MODEL_PATH = MODELS_DIR / "sentiment_analyzer" / "beto_finetuned"
SALES_PROPHET_MODEL_PATH = MODELS_DIR / "sales_predictor" / "prophet_model.joblib"

IMAGE_SIZE = (224, 224)
IMAGE_CLASSES = ["Apparel", "Accessories", "Footwear", "Personal Care", "Sporting Goods"]

SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
BETO_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
