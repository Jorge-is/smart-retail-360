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


# Categorías conocidas del dataset Rossmann (store.csv)
_STORE_TYPE_CATEGORIES = ["a", "b", "c", "d"]
_ASSORTMENT_CATEGORIES = ["a", "b", "c"]

STORE_META_COLS = (
    ["CompetitionDistance", "Promo2"]
    + [f"StoreType_{c}" for c in _STORE_TYPE_CATEGORIES]
    + [f"Assortment_{c}" for c in _ASSORTMENT_CATEGORIES]
)


def add_store_features(df: pd.DataFrame, store_csv_path) -> pd.DataFrame:
    store_meta = pd.read_csv(store_csv_path)
    df = df.merge(store_meta, left_on="store_id", right_on="Store", how="left")

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].max() * 2
    )
    df["Promo2"] = df["Promo2"].fillna(0).astype(int)

    df = pd.get_dummies(
        df,
        columns=["StoreType", "Assortment"],
        prefix=["StoreType", "Assortment"],
        dtype=int,
    )
    # Asegura que las columnas existan aunque alguna categoría no haya
    # aparecido en este split particular del dataset.
    for col in STORE_META_COLS:
        if col not in df.columns:
            df[col] = 0

    return df.drop(columns=["Store"])


def add_sentiment_regressor(df: pd.DataFrame, sentiment_series: pd.Series) -> pd.DataFrame:
    """
    Une el sentimiento promedio diario al DataFrame de ventas.
    sentiment_series: índice=fecha, valor=score promedio (0 a 1).
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

_POSITIVE_REVIEW_TEMPLATES = [
    "Excelente atención, encontré todo lo que buscaba muy rápido.",
    "Muy buena experiencia de compra, el local estaba impecable.",
    "El personal fue súper amable y me ayudó a elegir bien.",
    "Precios justos y buena variedad de productos, volveré seguro.",
    "Todo perfecto, la tienda tenía justo lo que necesitaba.",
    "Compré varias cosas y quedé muy conforme con la calidad.",
    "Una experiencia de compra muy agradable, recomendable.",
    "El local estaba ordenado y la atención fue rápida y amable.",
    "Encontré ofertas muy buenas, salí contento de la tienda.",
    "Ambiente agradable y buen servicio, sin dudas vuelvo.",
]

_NEUTRAL_REVIEW_TEMPLATES = [
    "La compra estuvo bien, nada fuera de lo común.",
    "Todo normal, encontré lo que buscaba sin mayores problemas.",
    "Atención correcta, aunque nada memorable.",
    "El local estaba como siempre, sin grandes cambios.",
    "Compra estándar, cumplió con lo esperado.",
]

_NEGATIVE_REVIEW_TEMPLATES = [
    "Mucha fila para pagar, tardé bastante en salir de la tienda.",
    "No encontré varios productos que buscaba en las góndolas.",
    "El local estaba algo desordenado, faltaba stock de varias cosas.",
    "La atención fue lenta, esperé demasiado para que me ayudaran.",
    "Precios más altos que en otras tiendas de la zona.",
    "Poca variedad de productos disponibles ese día.",
    "El local estaba muy lleno y la experiencia fue incómoda.",
]

def generate_synthetic_reviews(
    reference_sales: pd.DataFrame,
    noise: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    
    import random

    rng = random.Random(seed)

    df = reference_sales.copy().sort_values("ds").reset_index(drop=True)
    df["baseline"] = df["y"].rolling(window=7, min_periods=1, center=True).mean()
    df["above_trend"] = df["y"] > df["baseline"]

    rows = []
    for _, row in df.iterrows():
        above = row["above_trend"]
        if rng.random() < noise:
            above = rng.random() < 0.5  # ruido: se ignora la tendencia real

        if above:
            text = rng.choice(_POSITIVE_REVIEW_TEMPLATES)
        else:
            text = rng.choice(_NEUTRAL_REVIEW_TEMPLATES + _NEGATIVE_REVIEW_TEMPLATES)

        n_reviews = rng.choice([1, 1, 2])  # mayoría 1 reseña/día, a veces 2
        for _ in range(n_reviews):
            rows.append({"date": row["ds"].strftime("%Y-%m-%d"), "review_text": text})

    return pd.DataFrame(rows)
