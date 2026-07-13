"""XGBoost con features temporales — modelo de comparación frente a Prophet."""
import joblib
import pandas as pd
from xgboost import XGBRegressor

from src.sales_predictor.features import add_temporal_features
from src.utils.config import sales_xgboost_path, SALES_XGBOOST_GLOBAL_MODEL_PATH, SEED
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_COLS = ["day_of_week", "day_of_month", "week_of_year", "month", "is_weekend", "quarter"]
GLOBAL_FEATURE_COLS = FEATURE_COLS + ["store_id", "Promo"]


def build_model(**kwargs) -> XGBRegressor:
    defaults = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=SEED)
    return XGBRegressor(**{**defaults, **kwargs})


def prepare_features(df: pd.DataFrame, sentiment_col: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    df = add_temporal_features(df, date_col="ds")
    cols = FEATURE_COLS + (["sentiment"] if sentiment_col and "sentiment" in df.columns else [])
    return df[cols], df["y"]


def prepare_global_features(df: pd.DataFrame, sentiment_col: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    df = add_temporal_features(df, date_col="ds")
    cols = GLOBAL_FEATURE_COLS + (["sentiment"] if sentiment_col and "sentiment" in df.columns else [])
    return df[cols], df["y"]


def train_xgboost(train_df: pd.DataFrame, use_sentiment: bool = False, **kwargs) -> XGBRegressor:
    """Entrena un XGBRegressor por tienda sobre features temporales."""
    X_train, y_train = prepare_features(train_df, sentiment_col=use_sentiment)
    model = build_model(**kwargs)
    model.fit(X_train, y_train)
    return model


def train_xgboost_global(train_df: pd.DataFrame, use_sentiment: bool = False, **kwargs) -> XGBRegressor:
    X_train, y_train = prepare_global_features(train_df, sentiment_col=use_sentiment)
    model = build_model(**kwargs)
    model.fit(X_train, y_train)
    return model


def save_model(model: XGBRegressor, store_id: int) -> None:
    path = sales_xgboost_path(store_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("XGBoost (tienda %s) guardado en %s", store_id, path)


def load_model(store_id: int) -> XGBRegressor:
    path = sales_xgboost_path(store_id)
    if not path.exists():
        raise FileNotFoundError(f"No hay modelo XGBoost entrenado para store_id={store_id}.")
    return joblib.load(path)


def save_global_model(model: XGBRegressor) -> None:
    SALES_XGBOOST_GLOBAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SALES_XGBOOST_GLOBAL_MODEL_PATH)
    logger.info("XGBoost global guardado en %s", SALES_XGBOOST_GLOBAL_MODEL_PATH)


def load_global_model() -> XGBRegressor:
    if not SALES_XGBOOST_GLOBAL_MODEL_PATH.exists():
        raise FileNotFoundError("No hay modelo XGBoost global entrenado. Correr train.py primero.")
    return joblib.load(SALES_XGBOOST_GLOBAL_MODEL_PATH)
