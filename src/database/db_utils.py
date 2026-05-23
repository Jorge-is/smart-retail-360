import sqlite3
from pathlib import Path
from src.utils.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    schema_path = Path(__file__).parent / "schema.sql"
    with get_connection() as conn:
        conn.executescript(schema_path.read_text())


def save_image_prediction(filename: str, top_prediction: str, confidence: float, raw_result: str) -> int:
    sql = """
        INSERT INTO image_predictions (filename, top_prediction, confidence, raw_result)
        VALUES (?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (filename, top_prediction, confidence, raw_result))
        return cursor.lastrowid


def save_sentiment_prediction(text_snippet: str, sentiment: str, confidence: float, raw_result: str) -> int:
    sql = """
        INSERT INTO sentiment_predictions (text_snippet, sentiment, confidence, raw_result)
        VALUES (?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (text_snippet[:200], sentiment, confidence, raw_result))
        return cursor.lastrowid


def save_sales_forecast(store_id: int, horizon_days: int, sentiment_score: float, raw_result: str) -> int:
    sql = """
        INSERT INTO sales_forecasts (store_id, horizon_days, sentiment_score, raw_result)
        VALUES (?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (store_id, horizon_days, sentiment_score, raw_result))
        return cursor.lastrowid
