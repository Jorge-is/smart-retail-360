"""Funciones de cálculo de métricas reutilizables entre módulos."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """
    Calcula métricas estándar de clasificación.

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
