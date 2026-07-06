# SmartRetail 360

Plataforma de inteligencia artificial para e-commerce que integra clasificación de imágenes, análisis de sentimiento y predicción de demanda en un dashboard unificado construido con Streamlit.

## Módulos

| Módulo | Estado | Tecnología | Dataset |
|--------|--------|-----------|---------|
| Clasificación de productos por imagen | Activo | EfficientNet-B0 + MobileNetV2 (TensorFlow) | [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) (Kaggle) |
| Análisis de sentimiento de reseñas | Activo | BETO + Random Forest | [mteb/amazon_reviews_multi](https://huggingface.co/datasets/mteb/amazon_reviews_multi) (HF) |
| Predicción de demanda / ventas | En desarrollo | Prophet (Meta) | [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) (Kaggle) |

Cada módulo expone una única función `predict()` — el dashboard no conoce los detalles de TensorFlow, scikit-learn ni Prophet. Ver el contrato completo en [`docs/api_contracts.md`](docs/api_contracts.md) y la arquitectura en [`docs/architecture.md`](docs/architecture.md).

## Requisitos previos

- Python 3.10+
- Cuenta de [Kaggle](https://www.kaggle.com/) con API key (para descargar datasets)
- Los modelos entrenados (`.keras`, `.pkl`, `.joblib`) — se entrenan en Google Colab y no están incluidos en el repo

## Instalación

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd smart-retail-360

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales de Kaggle

# 5. Correr el dashboard
streamlit run app/app.py
```

## Uso

El entrenamiento de modelos está pensado para correr en Google Colab (GPU T4 gratuita), no localmente:

```bash
python -m src.image_classifier.train
python -m src.sentiment_analyzer.train
```

Los notebooks de EDA y entrenamiento viven en `notebooks/`. Una vez entrenado un modelo, descargá el archivo resultante (`.keras`, `.pkl` o carpeta de HuggingFace) y colocalo en `models/` respetando la estructura de la tabla de abajo.

### Tests

```bash
pytest tests/
pytest tests/test_image_classifier.py                       # módulo específico
pytest tests/test_sentiment_analyzer.py -k TestPreprocess    # clase específica
```

## Estructura del proyecto

```
smart-retail-360/
├── data/               # Datasets (en .gitignore si son grandes)
├── models/             # Modelos serializados (.keras, .pkl, .joblib)
├── notebooks/          # Notebooks de Colab para EDA y entrenamiento
├── src/                # Código fuente modular
│   ├── image_classifier/
│   ├── sentiment_analyzer/
│   ├── sales_predictor/    # placeholder, en desarrollo
│   ├── evaluation/         # Métricas y análisis de viabilidad compartidos
│   ├── database/
│   └── utils/
├── app/                # Dashboard Streamlit
│   ├── pages/
│   └── components/
├── tests/
└── docs/
```

### Modelos serializados

| Módulo | Archivo | Formato |
|--------|---------|---------|
| image_classifier | `models/image_classifier/efficientnet_b0.keras` | Keras `model.save` |
| image_classifier | `models/image_classifier/mobilenetv2.keras` | Keras `model.save` |
| sentiment_analyzer | `models/sentiment_analyzer/tfidf_baseline.pkl` | joblib Pipeline (Random Forest) |
| sentiment_analyzer | `models/sentiment_analyzer/beto_finetuned/` | HuggingFace `save_pretrained` |
| sales_predictor | `models/sales_predictor/prophet_model.joblib` | joblib Prophet (opcional) |

## Métricas objetivo

| Módulo | Modelo | Métrica | Objetivo |
|--------|--------|---------|---------|
| Clasificación de imágenes | MobileNetV2 | Accuracy test | ≥ 78% |
| Clasificación de imágenes | EfficientNet-B0 | Accuracy test | ≥ 85% |
| Sentimiento | Random Forest | F1 macro | ≥ 0.70 |
| Sentimiento | BETO | F1 macro | ≥ 0.80 |

Cada módulo entrega un análisis de viabilidad documentado usando `src/evaluation/viability.py`.

## Tecnologías

TensorFlow/Keras · PyTorch · HuggingFace Transformers · scikit-learn · Prophet · Streamlit · Polars · SQLite

## Equipo

Proyecto final del curso de Inteligencia Artificial, desarrollado por 6 integrantes. Roles, responsabilidades y organización interna en [TEAM.md](TEAM.md).
