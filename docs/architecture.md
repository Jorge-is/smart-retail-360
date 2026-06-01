# Arquitectura — SmartRetail 360

## Visión general

```
┌─────────────────────────────────────────────┐
│              app/ (Streamlit)                │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ Página 1 │ │ Página 2 │ │  Página 3   │  │
│  │ Imágenes │ │ Sentiment│ │   Ventas    │  │
│  └────┬─────┘ └────┬─────┘ └──────┬──────┘  │
└───────┼────────────┼──────────────┼──────────┘
        │            │              │
        ▼            ▼              ▼
┌──────────────────────────────────────────────┐
│                 src/ (Python)                │
│  image_classifier   sentiment_analyzer   sales_predictor
│     predict()           predict()          predict()   │
└──────────────────────────────────────────────┘
        │            │              │
        ▼            ▼              ▼
┌──────────────────────────────────────────────┐
│             models/ (archivos)               │
│  efficientnet_b0.keras  tfidf.pkl / beto/  prophet.joblib │
└──────────────────────────────────────────────┘
        │            │              │
        ▼            ▼              ▼
┌──────────────────────────────────────────────┐
│           data/SQLite (resultados)           │
│   image_predictions  sentiment_predictions  sales_forecasts │
└──────────────────────────────────────────────┘
```

## Principios de diseño

1. **API common**: cada módulo expone solo `predict()` al exterior. El dashboard no sabe nada de TensorFlow, scikit-learn ni Prophet.
2. **Lazy loading**: los modelos se cargan en memoria solo cuando se llama `predict()` por primera vez (singleton).
3. **Separación entrenamiento/inferencia**: los notebooks en Colab entrenan y guardan `.keras`/`.pkl`/`.joblib`. El dashboard solo hace inferencia.
4. **Fallback graceful**: si el modelo no está disponible, el dashboard muestra un mensaje claro en lugar de crashear.

## Flujo de datos

```
Google Colab (GPU)
    → entrenamiento con dataset
    → model.save() / joblib.dump
    → descarga manual → models/

models/ → src/*/predict.py → app/ → SQLite (resultados guardados)
```

## Módulos y responsabilidades

| Módulo | Responsabilidad |
|--------|----------------|
| `src/image_classifier/` | EfficientNet-B0: modelo, preprocess, train, predict |
| `src/sentiment_analyzer/` | BETO + TF-IDF: dos modelos, switching en runtime |
| `src/sales_predictor/` | Prophet: features temporales, predicción con regressor |
| `src/database/` | SQLite: guardar histórico de predicciones |
| `src/utils/` | Config global, logging |
| `app/` | Dashboard Streamlit: 4 páginas + componentes reutilizables |
