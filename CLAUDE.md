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
python -m src.sales_predictor.train
```

## Architecture

Three independent AI modules, each exposing a single `predict()` function. The Streamlit dashboard calls only those functions — it has no direct knowledge of PyTorch, Prophet, or scikit-learn.

### The predict() contract

Every module exposes the same pattern in `src/<module>/predict.py`:

```python
# image_classifier  →  predict(image: PIL.Image) -> dict
# sentiment_analyzer →  predict(text: str) -> dict
# sales_predictor    →  predict(store_id, horizon_days, sentiment_score) -> dict
```

Full I/O spec is in `docs/api_contracts.md`. This is the integration boundary — don't break it.

### Model loading

All three modules use a module-level singleton (`_model = None`). Models load on the first `predict()` call. If the serialized file doesn't exist, the function raises — the Streamlit pages catch this and show a "model not available" message instead of crashing.

### Module layout

```
src/<module>/
    predict.py      ← only file the dashboard imports
    model.py        ← architecture definition
    train.py        ← training script (runs in Colab)
    preprocess.py   ← transforms / text cleaning
```

`sentiment_analyzer` has two model backends (`baseline_model.py` = TF-IDF, `beto_model.py` = BETO). Switch at runtime with `set_mode("beto" | "tfidf")` before calling `predict()`.

### Streamlit app

`app/app.py` is the single entry point. Pages live in `app/pages/` as modules with a `render()` function — the router in `app.py` calls `render()` after the `option_menu` selection. Shared UI lives in `app/components/`.

### Sentiment → sales integration

The key differentiator: `sales_predictor.predict(sentiment_score=<float>)` accepts the average positive score from `sentiment_analyzer` as an exogenous regressor for Prophet. This is documented in `docs/api_contracts.md` under "Integración sentimiento → ventas".

### Config

All paths and constants come from `src/utils/config.py`, which reads `.env` via `python-dotenv`. Never hardcode paths — use the constants from `config.py` (`DATA_DIR`, `MODELS_DIR`, `IMAGE_CLASSIFIER_MODEL_PATH`, etc.).

### Serialized models

| Module | File | Format |
|--------|------|--------|
| image_classifier | `models/image_classifier/efficientnet_b0.pt` | `torch.save` state dict |
| sentiment_analyzer | `models/sentiment_analyzer/tfidf_baseline.pkl` | joblib Pipeline |
| sentiment_analyzer | `models/sentiment_analyzer/beto_finetuned/` | HuggingFace `save_pretrained` |
| sales_predictor | `models/sales_predictor/prophet_model.joblib` | joblib Prophet |

Models are in `.gitignore`. Train in Colab (notebooks in `notebooks/`) and copy the output files into `models/`.
