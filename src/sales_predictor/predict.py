from datetime import date, timedelta
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


def predict(
    store_id: int,
    horizon_days: int,
    sentiment_score: float = None,
    reference_date: date | None = None,
) -> dict:
    if sentiment_score is not None:
        sentiment_score = max(0.0, min(1.0, sentiment_score))

    model = _get_model(store_id)
    future = model.make_future_dataframe(periods=horizon_days)
    if sentiment_score is not None and "sentiment" in model.extra_regressors:
        future["sentiment"] = sentiment_score
    forecast = model.predict(future)
    result_df = forecast.tail(horizon_days)[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)

    ref = reference_date or date.today()
    display_dates = [ref + timedelta(days=i + 1) for i in range(horizon_days)]

    forecast_list = []
    negative_days = 0
    for i, row in result_df.iterrows():
        if row["yhat"] < 0:
            negative_days += 1
        forecast_list.append({
            "date": display_dates[i].strftime("%Y-%m-%d"),
            "predicted_sales": round(max(row["yhat"], 0), 2),
            "lower": round(max(row["yhat_lower"], 0), 2),
            "upper": round(max(row["yhat_upper"], 0), 2),
        })

    # Métricas vacías — se calculan durante entrenamiento y se guardan separado
    result = {
        "forecast": forecast_list,
        "metrics": {"mae": None, "rmse": None, "mape": None},
    }

    if negative_days:
        result["warning"] = (
            f"El pronóstico dio valores negativos (recortados a 0) en {negative_days} de "
            f"{horizon_days} días. Esto suele pasar cuando sentiment_score "
            f"({sentiment_score if sentiment_score is not None else 'N/A'}) está lejos del "
            f"rango con el que se entrenó el regressor de sentimiento — probá con un valor "
            f"más bajo (cerca de 0.1–0.3)."
        )

    return result
