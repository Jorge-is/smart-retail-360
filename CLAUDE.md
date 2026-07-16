# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dashboard
streamlit run app/app.py

# Tests
pytest tests/
pytest tests/test_image_classifier.py          # módulo específico
pytest tests/test_sentiment_analyzer.py -k TestPreprocess  # clase específica

# Entrenamiento (diseñado para correr en Google Colab, no local)
python -m src.image_classifier.train
python -m src.sentiment_analyzer.train
python -c "from src.sales_predictor.train import train_all; train_all()"
```

## Architecture

Three active AI modules (M1, M2, M3), each exposing a single `predict()` function. The Streamlit dashboard calls only those functions — it has no direct knowledge of TensorFlow, scikit-learn, Prophet, or XGBoost.

M3 (sales_predictor) trains Prophet per-store across all 1115 Rossmann stores, plus one global XGBoost model over the same 1115 for comparison. `SALES_STORE_SUBSET` in `src/utils/config.py` only restricts which stores appear in the dashboard's selector, not what's trained — the rest of the catalog has a model but isn't exposed in the UI. Its sentiment regressor comes from synthetic reviews generated from the sales trend itself (`generate_synthetic_reviews()` in `features.py`) — a known methodological limitation, not real independent review data. See `docs/api_contracts.md` for details.

### The predict() contract

Every module exposes the same pattern in `src/<module>/predict.py`:

```python
# image_classifier   →  predict(image: PIL.Image, model_name: str = "efficientnet") -> dict
# sentiment_analyzer →  predict(text: str, model_name: str = "beto") -> dict
# sales_predictor    →  predict(store_id, horizon_days, sentiment_score, reference_date=None) -> dict
```

Full I/O spec is in `docs/api_contracts.md`. This is the integration boundary — don't break it.

### Model loading

All modules use a module-level singleton (`_model = None`). Models load on the first `predict()` call. If the serialized file doesn't exist, the function raises — the Streamlit pages catch this and show a "model not available" message instead of crashing.

### Module layout

```
src/<module>/
    predict.py      ← only file the dashboard imports
    train.py        ← training script (runs in Colab)
    preprocess.py   ← transforms / text cleaning
    eda.py          ← reusable EDA helpers (M1 and M2 only)
```

`image_classifier` has two model files: `model_efficientnet.py` (EfficientNet-B0, main model) and `model_mobilenetv2.py` (MobileNetV2, baseline). `train.py` accepts `--model efficientnet|mobilenetv2`.

`sentiment_analyzer` has two model backends: `baseline_model.py` (Random Forest + TF-IDF) and `beto_model.py` (BETO). Select at call time via `predict(text, model_name="beto"|"random_forest")`.

### Evaluation module

`src/evaluation/` is a shared package used by both M1 and M2:
- `metrics.py` — accuracy, precision, recall, F1
- `confusion_matrix.py` — seaborn heatmap generation
- `viability.py` — assesses whether a model meets production thresholds

### Streamlit app

`app/app.py` is the single entry point. Pages live in `app/views/` as modules with a `render()` function — `app.py` registers each `render` callable via `st.Page(...)` and routes with `st.navigation(...).run()`. Shared UI lives in `app/components/`.

Note: the directory is named `views/`, not `pages/` — Streamlit auto-generates a native multi-page nav menu for any `pages/` folder next to the entry script, which conflicted with the router when the app used a custom `streamlit-option-menu` sidebar (empty screens when clicking the phantom nav items). Kept as `views/` after migrating to `st.navigation`/`st.Page` to avoid re-introducing that risk.

M3 (sales predictor) page is fully wired to `src.sales_predictor.predict`, with a "Comparación de modelos" tab reading `models/sales_predictor/metrics.json`.

### Config

All paths and constants come from `src/utils/config.py`, which reads `.env` via `python-dotenv`. Never hardcode paths — use the constants from `config.py` (`DATA_DIR`, `MODELS_DIR`, `IMAGE_CLASSIFIER_MODEL_PATH`, etc.).

### Serialized models

| Module | File | Format |
|--------|------|--------|
| image_classifier | `models/image_classifier/efficientnet_b0.keras` | Keras `model.save` |
| image_classifier | `models/image_classifier/mobilenetv2.keras` | Keras `model.save` |
| sentiment_analyzer | `models/sentiment_analyzer/tfidf_baseline.pkl` | joblib Pipeline (RandomForest) |
| sentiment_analyzer | `models/sentiment_analyzer/beto_finetuned/` | HuggingFace `save_pretrained` |
| sales_predictor | `models/sales_predictor/prophet_models.joblib` | joblib dict `{store_id: Prophet}`, consolidado (1115 tiendas) |
| sales_predictor | `models/sales_predictor/xgboost_global.joblib` | joblib `LogTargetXGBRegressor`, un solo modelo global |

Models are in `.gitignore`. Train in Colab (notebooks in `notebooks/`) and copy the output files into `models/`.
