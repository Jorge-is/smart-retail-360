"""Análisis de viabilidad del modelo basado en umbrales definidos."""

VIABILITY_THRESHOLDS = {
    "image_classifier": {
        "efficientnet": {"accuracy_min": 0.85, "f1_macro_min": 0.80},
        "mobilenetv2": {"accuracy_min": 0.78, "f1_macro_min": 0.73},
    },
    "sentiment_analyzer": {
        "beto": {"accuracy_min": 0.80, "f1_macro_min": 0.80},
        "random_forest": {"accuracy_min": 0.75, "f1_macro_min": 0.70},
    },
    "sales_predictor": {
        "prophet": {"mape_max": 15.0, "mape_warn": 25.0},
        "xgboost": {"mape_max": 15.0, "mape_warn": 25.0},
    },
}


def assess_viability(metrics: dict, module: str, model_name: str | None = None) -> dict:
    """
    Evalúa si el modelo supera los umbrales de viabilidad definidos.
    Válido para módulos de CLASIFICACIÓN (image_classifier, sentiment_analyzer).
    Para sales_predictor usar assess_regression_viability() en su lugar.

    Args:
        metrics: Resultado de compute_metrics() — debe incluir "accuracy" y "f1_macro".
        module: "image_classifier" | "sentiment_analyzer"
        model_name: Nombre del modelo dentro del módulo (ej. "efficientnet", "beto").
                    Si es None, usa el primer umbral disponible del módulo.

    Returns:
        {
            "is_viable": bool,
            "reasons": list[str],
            "recommendation": str  # "Apto para producción" | "Requiere mejoras" | "No viable"
        }
    """
    module_thresholds = VIABILITY_THRESHOLDS.get(module)
    if module_thresholds is None:
        return {
            "is_viable": False,
            "reasons": [f"Módulo '{module}' no tiene umbrales definidos."],
            "recommendation": "No viable",
        }

    if model_name and model_name in module_thresholds:
        thresholds = module_thresholds[model_name]
    else:
        thresholds = next(iter(module_thresholds.values()))

    reasons = []
    passed = 0

    accuracy = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1_macro", 0.0)

    if accuracy >= thresholds["accuracy_min"]:
        passed += 1
    else:
        reasons.append(
            f"Accuracy {accuracy:.2%} < umbral {thresholds['accuracy_min']:.2%}"
        )

    if f1 >= thresholds["f1_macro_min"]:
        passed += 1
    else:
        reasons.append(
            f"F1-macro {f1:.2%} < umbral {thresholds['f1_macro_min']:.2%}"
        )

    is_viable = passed == 2
    if is_viable:
        recommendation = "Apto para producción"
    elif passed == 1:
        recommendation = "Requiere mejoras"
    else:
        recommendation = "No viable"

    return {
        "is_viable": is_viable,
        "reasons": reasons if reasons else ["Todos los umbrales superados."],
        "recommendation": recommendation,
    }


def assess_regression_viability(metrics: dict, model_name: str = "prophet") -> dict:
    """
    Evalúa viabilidad de un modelo de forecasting (módulo 3) según su MAPE.

    Escala usada (estándar en forecasting — a menor MAPE, mejor):
      MAPE < 15%          -> Apto para producción
      15% <= MAPE < 25%   -> Requiere mejoras
      MAPE >= 25%          -> No viable

    Args:
        metrics: Resultado de compute_regression_metrics() — debe incluir "mape".
        model_name: "prophet" | "xgboost"

    Returns:
        {"is_viable": bool, "reasons": list[str], "recommendation": str}
    """
    thresholds = VIABILITY_THRESHOLDS["sales_predictor"].get(
        model_name, {"mape_max": 15.0, "mape_warn": 25.0}
    )
    mape = metrics.get("mape", 100.0)

    if mape < thresholds["mape_max"]:
        is_viable = True
        recommendation = "Apto para producción"
        reasons = [f"MAPE {mape:.2f}% < {thresholds['mape_max']}%"]
    elif mape < thresholds["mape_warn"]:
        is_viable = False
        recommendation = "Requiere mejoras"
        reasons = [f"MAPE {mape:.2f}% entre {thresholds['mape_max']}% y {thresholds['mape_warn']}%"]
    else:
        is_viable = False
        recommendation = "No viable"
        reasons = [f"MAPE {mape:.2f}% >= {thresholds['mape_warn']}%"]

    return {"is_viable": is_viable, "reasons": reasons, "recommendation": recommendation}
