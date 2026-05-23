-- SmartRetail 360 — Schema SQLite

CREATE TABLE IF NOT EXISTS image_predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    filename      TEXT NOT NULL,
    top_prediction TEXT NOT NULL,
    confidence    REAL NOT NULL,
    raw_result    TEXT   -- JSON completo del predict()
);

CREATE TABLE IF NOT EXISTS sentiment_predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    text_snippet  TEXT NOT NULL,
    sentiment     TEXT NOT NULL CHECK(sentiment IN ('positive', 'neutral', 'negative')),
    confidence    REAL NOT NULL,
    raw_result    TEXT
);

CREATE TABLE IF NOT EXISTS sales_forecasts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    store_id      INTEGER NOT NULL,
    horizon_days  INTEGER NOT NULL,
    sentiment_score REAL,
    raw_result    TEXT
);
