import joblib
from pathlib import Path
from prophet import Prophet

from src.utils.config import sales_prophet_path, SALES_PROPHET_MODELS_PATH
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_model(use_sentiment: bool = False) -> Prophet:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    if use_sentiment:
        model.add_regressor("sentiment")
    return model


def save_model(model: Prophet, store_id: int) -> None:
    path = sales_prophet_path(store_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Prophet (tienda %s) guardado en %s", store_id, path)


def load_model(store_id: int) -> Prophet:
    path = sales_prophet_path(store_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No hay modelo Prophet entrenado para store_id={store_id}. "
            f"Revisá SALES_STORE_SUBSET en config.py o corré train.py."
        )
    return joblib.load(path)


def consolidate_models(store_ids: list[int]) -> Path:
    models = {}
    missing = []
    for store_id in store_ids:
        path = sales_prophet_path(store_id)
        if path.exists():
            models[store_id] = joblib.load(path)
        else:
            missing.append(store_id)

    if missing:
        logger.warning(
            "%s tiendas sin modelo individual entrenado, se omiten de la consolidación: %s",
            len(missing), missing,
        )

    SALES_PROPHET_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, SALES_PROPHET_MODELS_PATH)
    logger.info("%s modelos Prophet consolidados en %s", len(models), SALES_PROPHET_MODELS_PATH)
    return SALES_PROPHET_MODELS_PATH


def load_all_models() -> dict:
    """Carga el archivo consolidado (todas las tiendas en un dict, un solo .joblib)."""
    if not SALES_PROPHET_MODELS_PATH.exists():
        raise FileNotFoundError(
            f"No existe {SALES_PROPHET_MODELS_PATH}. Correr consolidate_models() primero."
        )
    return joblib.load(SALES_PROPHET_MODELS_PATH)
