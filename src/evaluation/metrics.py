"""Funciones de cálculo de métricas reutilizables entre módulos."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """
    Calcula métricas estándar de clasificación.
    Usado por los módulos 1 (image_classifier) y 2 (sentiment_analyzer).

    Returns:
        {
            "accuracy": float,
            "precision_macro": float,
            "recall_macro": float,
            "f1_macro": float,
            "report": str  # classification_report completo
        }
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "report": report,
    }


def compute_regression_metrics(y_true, y_pred) -> dict:
    """
    Calcula métricas estándar de regresión, para series de tiempo continuas.

    Returns:
        {"mae": float, "rmse": float, "mape": float}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # np.maximum(y_true, 1) evita división por cero en días con 0 ventas
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)

    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}
