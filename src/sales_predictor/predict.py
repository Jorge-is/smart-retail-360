import pandas as pd

from src.sales_predictor.prophet_model import load_model
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_models: dict[int, object] = {}


def _get_model(store_id: int):
    if store_id not in _models:
        _models[store_id] = load_model(store_id)
    return _models[store_id]


def predict(store_id: int, horizon_days: int, sentiment_score: float = None) -> dict:
    """
    Args:
        store_id: ID de la tienda (debe estar en SALES_STORE_SUBSET, ver config.py).
        horizon_days: Días a pronosticar hacia el futuro (7, 15 o 30).
        sentiment_score: Sentimiento promedio (0.0–1.0) para usar como regressor.
                         None si el modelo fue entrenado sin esta feature.

    Returns:
        {
            "forecast": [
                {"date": str, "predicted_sales": float, "lower": float, "upper": float},
                ...
            ],
            "metrics": {"mae": float, "rmse": float, "mape": float}
        }
    """
    model = _get_model(store_id)
    future = model.make_future_dataframe(periods=horizon_days)
    if sentiment_score is not None and "sentiment" in model.extra_regressors:
        future["sentiment"] = sentiment_score
    forecast = model.predict(future)
    result_df = forecast.tail(horizon_days)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast_list = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "predicted_sales": round(max(row["yhat"], 0), 2),
            "lower": round(max(row["yhat_lower"], 0), 2),
            "upper": round(max(row["yhat_upper"], 0), 2),
        }
        for _, row in result_df.iterrows()
    ]
    # Métricas vacías — se calculan durante entrenamiento y se guardan separado
    return {
        "forecast": forecast_list,
        "metrics": {"mae": None, "rmse": None, "mape": None},
    }
