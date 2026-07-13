from typing import Callable

import pandas as pd


def add_temporal_features(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    """Agrega features temporales derivadas de la columna de fecha."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df[date_col].dt.quarter
    return df


def prepare_prophet_df(raw_df: pd.DataFrame, store_id: int) -> pd.DataFrame:
    """
    Filtra por tienda y devuelve DataFrame con columnas 'ds' e 'y'
    listo para Prophet.
    """
    df = raw_df[raw_df["Store"] == store_id][["Date", "Sales", "Open"]].copy()
    df = df[df["Open"] == 1].drop(columns=["Open"])
    df = df.rename(columns={"Date": "ds", "Sales": "y"})
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def prepare_global_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset COMPLETO (las 1115 tiendas juntas) para el XGBoost
    Devuelve columnas: store_id, ds, y, Promo.
    """
    df = raw_df[raw_df["Open"] == 1][["Store", "Date", "Sales", "Promo"]].copy()
    df = df.rename(columns={"Store": "store_id", "Date": "ds", "Sales": "y"})
    df = df.sort_values(["store_id", "ds"]).reset_index(drop=True)
    return df


def add_sentiment_regressor(df: pd.DataFrame, sentiment_series: pd.Series) -> pd.DataFrame:
    """
    Une el sentimiento promedio diario al DataFrame de ventas.
    sentiment_series: índice=fecha, valor=score promedio (−1 a 1 o 0 a 1).
    Los días sin reseña se rellenan con el promedio general de la serie.
    """
    df = df.copy()
    df["sentiment"] = df["ds"].map(sentiment_series).fillna(sentiment_series.mean())
    return df


def build_sentiment_series_from_csv(csv_path, predict_fn: Callable[[str], dict]) -> pd.Series:
    """
    Lee un CSV de reseñas (columnas: date, review_text), corre predict_fn
    (src.sentiment_analyzer.predict.predict) sobre cada texto y agrupa por
    fecha, usando scores["positive"] como score numérico 0-1.
    """
    reviews_df = pd.read_csv(csv_path, parse_dates=["date"])

    scores = []
    for _, row in reviews_df.iterrows():
        result = predict_fn(row["review_text"])
        scores.append(result["scores"]["positive"])
    reviews_df["sentiment_score"] = scores

    daily_series = reviews_df.groupby("date")["sentiment_score"].mean()
    return daily_series
