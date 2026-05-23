# API Contracts — SmartRetail 360

Cada módulo expone una función `predict()` con I/O estandarizada.
Esto permite que el dashboard y los tests sean independientes del modelo subyacente.

---

## Módulo 1 — Clasificador de imágenes

**Función:** `src.image_classifier.predict.predict(image)`

**Input:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `image` | `PIL.Image` | Imagen en cualquier modo (RGB, RGBA, L). Se convierte internamente. |

**Output:**
```json
{
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
- La suma de `confidence` de todos los elementos puede ser < 1.0 (se muestran los top-3 de softmax, no normalizados entre sí).
- `top_prediction` == `top_3[0]["class"]`.

---

## Módulo 2 — Análisis de sentimiento

**Función:** `src.sentiment_analyzer.predict.predict(text)`

**Input:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `text` | `str` | Reseña en texto libre. No hay límite hard, pero BETO trunca a 256 tokens. |

**Output:**
```json
{
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

**Función auxiliar:** `set_mode(mode: str)` — cambia entre `"beto"` y `"tfidf"` sin recargar ambos modelos.

---

## Módulo 3 — Predicción de ventas

**Función:** `src.sales_predictor.predict.predict(store_id, horizon_days, sentiment_score)`

**Input:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `store_id` | `int` | ID de la tienda (1–1115 en Rossmann). |
| `horizon_days` | `int` | Días a pronosticar: típicamente 7, 15 o 30. |
| `sentiment_score` | `float \| None` | Score de sentimiento (0.0–1.0). `None` si el modelo no tiene este regressor. |

**Output:**
```json
{
  "forecast": [
    {"date": "2024-03-01", "predicted_sales": 4823.0, "lower": 3900.0, "upper": 5740.0},
    {"date": "2024-03-02", "predicted_sales": 5100.0, "lower": 4200.0, "upper": 6000.0}
  ],
  "metrics": {"mae": 312.5, "rmse": 420.1, "mape": 8.7}
}
```

**Invariantes:**
- `len(forecast)` == `horizon_days`.
- `lower` ≤ `predicted_sales` ≤ `upper`.
- Todos los valores de ventas son ≥ 0.
- `metrics` puede tener valores `null` si no se calcularon (inferencia pura).

---

## Integración sentimiento → ventas

El pipeline integrado:

```
reseñas (CSV)
    → sentiment_analyzer.predict()  [por fila]
    → promedio de scores["positive"]
    → sales_predictor.predict(sentiment_score=avg)
```

El score de sentimiento se mapea al rango esperado por Prophet (0.0 = muy negativo, 1.0 = muy positivo) antes de pasarlo como regressor.
