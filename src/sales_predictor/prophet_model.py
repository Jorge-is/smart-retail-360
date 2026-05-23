import joblib
import pandas as pd
from prophet import Prophet

from src.utils.config import SALES_PROPHET_MODEL_PATH
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_model(use_sentiment: bool = False) -> Prophet:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    if use_sentiment:
        model.add_regressor("sentiment")
    return model


def save_model(model: Prophet) -> None:
    SALES_PROPHET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SALES_PROPHET_MODEL_PATH)
    logger.info("Prophet guardado en %s", SALES_PROPHET_MODEL_PATH)


def load_model() -> Prophet:
    return joblib.load(SALES_PROPHET_MODEL_PATH)
