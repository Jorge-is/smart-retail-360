import pandas as pd

from src.sales_predictor.prophet_model import load_all_models
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_models_cache: dict | None = None


def _get_all_models() -> dict:
    global _models_cache
    if _models_cache is None:
        _models_cache = load_all_models()
    return _models_cache


def _get_model(store_id: int):
    models = _get_all_models()
    if store_id not in models:
        raise FileNotFoundError(
            f"No hay modelo Prophet consolidado para store_id={store_id}. "
            f"Revisá que se haya corrido consolidate_models() incluyendo esa tienda."
        )
    return models[store_id]


def predict(store_id: int, horizon_days: int, sentiment_score: float = None) -> dict:
    """
    Args:
        store_id: ID de la tienda (debe estar dentro del archivo consolidado).
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
