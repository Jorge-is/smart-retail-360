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
```

## Architecture

Two active AI modules (M1 + M2) plus one optional placeholder (M3), each exposing a single `predict()` function. The Streamlit dashboard calls only those functions — it has no direct knowledge of TensorFlow, scikit-learn, or Prophet.

### The predict() contract

Every module exposes the same pattern in `src/<module>/predict.py`:

```python
# image_classifier   →  predict(image: PIL.Image, model_name: str = "efficientnet") -> dict
# sentiment_analyzer →  predict(text: str, model_name: str = "beto") -> dict
# sales_predictor    →  predict(store_id, horizon_days, sentiment_score) -> dict  [optional/placeholder]
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

`app/app.py` is the single entry point. Pages live in `app/pages/` as modules with a `render()` function — the router in `app.py` calls `render()` after the `option_menu` selection. Shared UI lives in `app/components/`.

M3 (sales predictor) page shows a "Módulo en desarrollo" placeholder — do not wire it to real model calls.

### Config

All paths and constants come from `src/utils/config.py`, which reads `.env` via `python-dotenv`. Never hardcode paths — use the constants from `config.py` (`DATA_DIR`, `MODELS_DIR`, `IMAGE_CLASSIFIER_MODEL_PATH`, etc.).

### Serialized models

| Module | File | Format |
|--------|------|--------|
| image_classifier | `models/image_classifier/efficientnet_b0.keras` | Keras `model.save` |
| image_classifier | `models/image_classifier/mobilenetv2.keras` | Keras `model.save` |
| sentiment_analyzer | `models/sentiment_analyzer/tfidf_baseline.pkl` | joblib Pipeline (RandomForest) |
| sentiment_analyzer | `models/sentiment_analyzer/beto_finetuned/` | HuggingFace `save_pretrained` |
| sales_predictor | `models/sales_predictor/prophet_model.joblib` | joblib Prophet (optional) |

Models are in `.gitignore`. Train in Colab (notebooks in `notebooks/`) and copy the output files into `models/`.
