"""
Entrenamiento de Prophet y XGBoost sobre Rossmann Store Sales. 
Diseñado para correr en Google Colab (CPU).

Al final, todas las métricas quedan en models/sales_predictor/metrics.json.
"""
import json

import pandas as pd

from src.sales_predictor.features import (
    prepare_prophet_df,
    prepare_global_df,
    add_sentiment_regressor,
    build_sentiment_series_from_csv,
)
from src.sales_predictor.prophet_model import build_model as build_prophet, save_model as save_prophet
from src.sales_predictor.xgboost_model import (
    train_xgboost_global,
    save_global_model,
    prepare_global_features,
)
from src.sentiment_analyzer.predict import predict as predict_sentiment
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.viability import assess_regression_viability
from src.utils.config import DATA_DIR, SALES_STORE_SUBSET, SALES_SENTIMENT_CSV_PATH, SALES_METRICS_PATH
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_sentiment_series() -> pd.Series | None:
    if not SALES_SENTIMENT_CSV_PATH.exists():
        logger.warning(
            "No se encontró CSV de reseñas sintéticas en %s — se entrena sin regressor de sentimiento.",
            SALES_SENTIMENT_CSV_PATH,
        )
        return None
    return build_sentiment_series_from_csv(SALES_SENTIMENT_CSV_PATH, predict_sentiment)


def train_store(raw_df: pd.DataFrame, store_id: int, sentiment_series: pd.Series | None) -> dict:
    df = prepare_prophet_df(raw_df, store_id=store_id)
    use_sentiment = sentiment_series is not None
    if use_sentiment:
        df = add_sentiment_regressor(df, sentiment_series)

    cutoff = df["ds"].max() - pd.Timedelta(days=30)
    train_df = df[df["ds"] <= cutoff]
    test_df = df[df["ds"] > cutoff]

    prophet_model = build_prophet(use_sentiment=use_sentiment)
    prophet_model.fit(train_df)

    horizon_days = (test_df["ds"].max() - train_df["ds"].max()).days
    future = prophet_model.make_future_dataframe(periods=horizon_days)
    if use_sentiment:
        future["sentiment"] = future["ds"].map(sentiment_series).fillna(sentiment_series.mean())
    forecast = prophet_model.predict(future)

    merged = test_df.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    prophet_metrics = compute_regression_metrics(merged["y"].values, merged["yhat"].values)
    prophet_viability = assess_regression_viability(prophet_metrics, model_name="prophet")
    save_prophet(prophet_model, store_id)

    logger.info(
        "Tienda %s — Prophet MAPE: %.2f%% (%s)",
        store_id, prophet_metrics["mape"], prophet_viability["recommendation"],
    )

    return {
        "store_id": store_id,
        "used_sentiment_regressor": use_sentiment,
        "prophet": {"metrics": prophet_metrics, "viability": prophet_viability},
    }


def train_xgboost_all_stores(raw_df: pd.DataFrame, sentiment_series: pd.Series | None) -> dict:
    """Entrena UN SOLO XGBoost con las 1115 tiendas juntas."""
    df = prepare_global_df(raw_df)
    use_sentiment = sentiment_series is not None
    if use_sentiment:
        df = add_sentiment_regressor(df, sentiment_series)

    cutoff = df["ds"].max() - pd.Timedelta(days=30)
    train_df = df[df["ds"] <= cutoff]
    test_df = df[df["ds"] > cutoff]

    xgb_model = train_xgboost_global(train_df, use_sentiment=use_sentiment)
    X_test, y_test = prepare_global_features(test_df, sentiment_col=use_sentiment)
    xgb_pred = xgb_model.predict(X_test)

    xgb_metrics = compute_regression_metrics(y_test.values, xgb_pred)
    xgb_viability = assess_regression_viability(xgb_metrics, model_name="xgboost")
    save_global_model(xgb_model)

    logger.info(
        "XGBoost global (%s tiendas, %s filas de train) — MAPE: %.2f%% (%s)",
        df["store_id"].nunique(), len(train_df), xgb_metrics["mape"], xgb_viability["recommendation"],
    )

    return {
        "scope": "global",
        "n_stores": int(df["store_id"].nunique()),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "used_sentiment_regressor": use_sentiment,
        "xgboost": {"metrics": xgb_metrics, "viability": xgb_viability},
    }


def train_all(store_subset: list[int] | None = None) -> None:
    store_subset = store_subset or SALES_STORE_SUBSET
    raw_path = DATA_DIR / "raw" / "rossmann" / "train.csv"
    logger.info("Cargando Rossmann desde %s", raw_path)
    raw_df = pd.read_csv(raw_path, parse_dates=["Date"])

    sentiment_series = _load_sentiment_series()

    prophet_results = [train_store(raw_df, store_id, sentiment_series) for store_id in store_subset]
    xgboost_global_result = train_xgboost_all_stores(raw_df, sentiment_series)

    output = {
        "prophet_by_store": prophet_results,
        "xgboost_global": xgboost_global_result,
    }

    SALES_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SALES_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(
        "Métricas guardadas en %s — Prophet: %s tiendas | XGBoost: %s tiendas (global)",
        SALES_METRICS_PATH, len(store_subset), xgboost_global_result["n_stores"],
    )


if __name__ == "__main__":
    train_all()
