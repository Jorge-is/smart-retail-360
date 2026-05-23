# SmartRetail 360

Plataforma de inteligencia artificial para e-commerce que integra tres capacidades de IA en un dashboard unificado:

| Módulo | Tecnología | Dataset |
|--------|-----------|---------|
| Clasificación de productos por imagen | EfficientNet-B0 (PyTorch) | Fashion Product Images (Kaggle) |
| Análisis de sentimiento de reseñas | BETO + TF-IDF baseline | Amazon Reviews Multilingual (HF) |
| Predicción de demanda / ventas | Prophet (Meta) | Rossmann Store Sales (Kaggle) |

**Elemento integrador:** el sentimiento promedio de las reseñas se usa como variable exógena en la predicción de ventas.

---

## Setup rápido

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd smart-retail-360

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales de Kaggle

# 5. Correr el dashboard
streamlit run app/app.py
```

---

## Estructura del proyecto

```
smartretail360/
├── data/               # Datasets (en .gitignore si son grandes)
├── models/             # Modelos serializados (.pt, .pkl, .joblib)
├── notebooks/          # Notebooks de Colab para entrenamiento
├── src/                # Código fuente modular
│   ├── image_classifier/
│   ├── sentiment_analyzer/
│   ├── sales_predictor/
│   ├── database/
│   └── utils/
├── app/                # Dashboard Streamlit
│   ├── pages/
│   └── components/
├── tests/
└── docs/
```

## Flujo de trabajo

1. **Entrenamiento** → Correr los notebooks en Google Colab (GPU T4 gratuita)
2. **Descargar modelos** → Guardar `.pt` / `.pkl` / `.joblib` en `models/`
3. **Dashboard** → `streamlit run app/app.py`

## Métricas objetivo

| Módulo | Métrica | Objetivo |
|--------|---------|---------|
| Clasificación imágenes | Accuracy test | ≥ 85% (5 clases) |
| Sentimiento BETO | F1 macro | ≥ 0.80 |
| Sentimiento TF-IDF | F1 macro | ≥ 0.70 |
| Predicción ventas | MAPE | ≤ 15% |

## Equipo

Proyecto Final — Curso de Inteligencia Artificial · 6 integrantes
