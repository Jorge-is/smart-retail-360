"""
Entrenamiento de Prophet (y opcionalmente XGBoost) sobre Rossmann Store Sales.
Diseñado para correr en Google Colab.
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from src.sales_predictor.features import prepare_prophet_df
from src.sales_predictor.prophet_model import build_model as build_prophet, save_model as save_prophet
from src.utils.config import DATA_DIR, SEED
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def train_prophet(store_id: int = 1, use_sentiment: bool = False) -> None:
    raw_path = DATA_DIR / "raw" / "rossmann" / "train.csv"
    logger.info("Cargando Rossmann desde %s", raw_path)

    raw_df = pd.read_csv(raw_path, parse_dates=["Date"])
    df = prepare_prophet_df(raw_df, store_id=store_id)

    cutoff = df["ds"].max() - pd.Timedelta(days=30)
    train_df = df[df["ds"] <= cutoff]
    test_df = df[df["ds"] > cutoff]

    model = build_prophet(use_sentiment=use_sentiment)
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)

    y_true = test_df["y"].values
    y_pred = forecast.tail(len(test_df))["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    logger.info("Prophet — MAE: %.2f | RMSE: %.2f | MAPE: %.2f%%", mae, rmse, mape)
    save_prophet(model)


if __name__ == "__main__":
    train_prophet(store_id=1)
