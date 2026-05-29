# API Contracts — SmartRetail 360

Cada módulo expone una función `predict()` con I/O estandarizada.
Esto permite que el dashboard y los tests sean independientes del modelo subyacente.

---

## Módulo 1 — Clasificador de imágenes

**Función:** `src.image_classifier.predict.predict(image, model_name)`

**Input:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `image` | `PIL.Image` | Imagen en cualquier modo (RGB, RGBA, L). Se convierte internamente. |
| `model_name` | `str` | `"efficientnet"` (default) \| `"mobilenetv2"` |

**Output:**
```json
{
  "model_used": "efficientnet",
  "top_prediction": "Apparel",
  "top_3": [
    {"class": "Apparel", "confidence": 0.87},
    {"class": "Footwear", "confidence": 0.08},
    {"class": "Accessories", "confidence": 0.04}
  ],
  "inference_time_ms": 23.4
}
```

**Invariantes:**
- `top_3` siempre tiene entre 1 y 3 elementos.
- `confidence` ∈ [0.0, 1.0].
- `top_prediction` == `top_3[0]["class"]`.
- `model_used` refleja el modelo efectivamente usado en la inferencia.

**Modelos disponibles:**

| `model_name` | Archivo esperado | Accuracy objetivo |
|---|---|---|
| `"efficientnet"` | `models/image_classifier/efficientnet_b0.keras` | ≥ 85% |
| `"mobilenetv2"` | `models/image_classifier/mobilenetv2.keras` | ≥ 78% |

---

## Módulo 2 — Análisis de sentimiento

**Función:** `src.sentiment_analyzer.predict.predict(text, model_name)`

**Input:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `str` | Reseña en texto libre. No hay límite hard, pero BETO trunca a 256 tokens. |
| `model_name` | `str` | `"beto"` (default) \| `"random_forest"` |

**Output:**
```json
{
  "model_used": "beto",
  "sentiment": "positive",
  "confidence": 0.91,
  "scores": {
    "positive": 0.91,
    "neutral": 0.06,
    "negative": 0.03
  }
}
```

**Invariantes:**
- `sentiment` ∈ `{"positive", "neutral", "negative"}`.
- `confidence` ∈ [0.0, 1.0].
- `scores["positive"] + scores["neutral"] + scores["negative"]` ≈ 1.0 (± 1e-3).
- `confidence` == `scores[sentiment]`.
- `model_used` refleja el modelo efectivamente usado.

**Modelos disponibles:**

| `model_name` | Archivo esperado | F1-macro objetivo |
|---|---|---|
| `"beto"` | `models/sentiment_analyzer/beto_finetuned/` | ≥ 0.80 |
| `"random_forest"` | `models/sentiment_analyzer/tfidf_baseline.pkl` | ≥ 0.70 |

**Función auxiliar:** `set_mode(mode: str)` — alias legacy para cambiar el modelo activo sin recargar.

---

## Módulo 3 — Predicción de ventas (opcional)

> **Estado:** placeholder. No se trabaja activamente en esta entrega.
> Si se retoma, ver `notebooks/04_sales_predictor_prophet.ipynb`.

**Función:** `src.sales_predictor.predict.predict(store_id, horizon_days, sentiment_score)`

**Input:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `store_id` | `int` | ID de la tienda (1–1115 en Rossmann). |
| `horizon_days` | `int` | Días a pronosticar: típicamente 7, 15 o 30. |
| `sentiment_score` | `float \| None` | Score de sentimiento (0.0–1.0). `None` si no se usa como regressor. |

**Output:**
```json
{
  "forecast": [
    {"date": "2024-03-01", "predicted_sales": 4823.0, "lower": 3900.0, "upper": 5740.0}
  ],
  "metrics": {"mae": 312.5, "rmse": 420.1, "mape": 8.7}
}
```

---

## Módulo de evaluación compartida

**Paquete:** `src.evaluation`

### `metrics.compute_metrics(y_true, y_pred, labels)`

```python
{
    "accuracy": float,
    "precision_macro": float,
    "recall_macro": float,
    "f1_macro": float,
    "report": str
}
```

### `confusion_matrix.plot_confusion_matrix(y_true, y_pred, labels, title, save_path)`

Devuelve un `matplotlib.Figure` con el heatmap de la matriz de confusión.

### `viability.assess_viability(metrics, module, model_name)`

```python
{
    "is_viable": bool,
    "reasons": list[str],
    "recommendation": "Apto para producción" | "Requiere mejoras" | "No viable"
}
```
