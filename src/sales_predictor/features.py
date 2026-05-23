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
    df = raw_df[raw_df["Store"] == store_id][["Date", "Sales"]].copy()
    df = df.rename(columns={"Date": "ds", "Sales": "y"})
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def add_sentiment_regressor(df: pd.DataFrame, sentiment_series: pd.Series) -> pd.DataFrame:
    """
    Une el sentimiento promedio diario al DataFrame de ventas.
    sentiment_series: índice=fecha, valor=score promedio (−1 a 1 o 0 a 1).
    """
    df = df.copy()
    df["sentiment"] = df["ds"].map(sentiment_series).fillna(sentiment_series.mean())
    return df
