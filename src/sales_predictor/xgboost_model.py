"""XGBoost con features temporales — modelo de comparación opcional."""
import joblib
import pandas as pd
from xgboost import XGBRegressor

from src.sales_predictor.features import add_temporal_features
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_COLS = ["day_of_week", "day_of_month", "week_of_year", "month", "is_weekend", "quarter"]


def build_model(**kwargs) -> XGBRegressor:
    defaults = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)
    return XGBRegressor(**{**defaults, **kwargs})


def prepare_features(df: pd.DataFrame, sentiment_col: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    df = add_temporal_features(df, date_col="ds")
    cols = FEATURE_COLS + (["sentiment"] if sentiment_col and "sentiment" in df.columns else [])
    return df[cols], df["y"]


def save_model(model: XGBRegressor, path) -> None:
    joblib.dump(model, path)
    logger.info("XGBoost guardado en %s", path)


def load_model(path) -> XGBRegressor:
    return joblib.load(path)
