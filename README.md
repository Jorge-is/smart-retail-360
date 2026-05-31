# SmartRetail 360

Plataforma de inteligencia artificial para e-commerce que integra capacidades de IA en un dashboard unificado.

| Módulo | Estado | Tecnología | Dataset |
|--------|--------|-----------|---------|
| Clasificación de productos por imagen | Activo | EfficientNet-B0 + MobileNetV2 (TensorFlow) | Fashion Product Images (Kaggle) |
| Análisis de sentimiento de reseñas | Activo | BETO + Random Forest | Amazon Reviews Multilingual (HF) |
| Predicción de demanda / ventas | Opcional | Prophet (Meta) | Rossmann Store Sales (Kaggle) |

---

## Setup rápido

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

---

## Estructura del proyecto

```
smart-retail-360/
├── data/               # Datasets (en .gitignore si son grandes)
├── models/             # Modelos serializados (.keras, .pkl, .joblib)
├── notebooks/          # Notebooks de Colab para EDA y entrenamiento
│   ├── 00_eda_imagenes.ipynb
│   ├── 00_eda_sentimiento.ipynb
│   ├── 01_image_classifier_efficientnet.ipynb
│   ├── 01_image_classifier_mobilenetv2.ipynb
│   ├── 02_sentiment_baseline_randomforest.ipynb
│   ├── 03_sentiment_beto_finetuning.ipynb
│   └── 04_sales_predictor_prophet.ipynb  (opcional)
├── src/                # Código fuente modular
│   ├── image_classifier/
│   ├── sentiment_analyzer/
│   ├── sales_predictor/  (placeholder opcional)
│   ├── evaluation/       # Métricas y análisis de viabilidad compartidos
│   ├── database/
│   └── utils/
├── app/                # Dashboard Streamlit
│   ├── pages/
│   └── components/
├── tests/
└── docs/
```

## Flujo de trabajo

1. **EDA** → Correr `00_eda_*.ipynb` en Colab, documentar hallazgos
2. **Entrenamiento** → Correr notebooks de entrenamiento en Colab (GPU T4 gratuita)
3. **Descargar modelos** → Guardar `.keras` / `.pkl` en `models/`
4. **Dashboard** → `streamlit run app/app.py`

## Métricas objetivo

| Módulo | Modelo | Métrica | Objetivo |
|--------|--------|---------|---------|
| M1 — Clasificación imágenes | MobileNetV2 | Accuracy test | ≥ 78% |
| M1 — Clasificación imágenes | EfficientNet-B0 | Accuracy test | ≥ 85% |
| M2 — Sentimiento | Random Forest | F1 macro | ≥ 0.70 |
| M2 — Sentimiento | BETO | F1 macro | ≥ 0.80 |

Cada módulo entrega un **análisis de viabilidad** documentado usando `src/evaluation/viability.py`.

## Cronograma (7 semanas)

| Semana | Foco | Entregables |
|--------|------|-------------|
| 1 | Setup + EDA | Actualizar repo, ejecutar EDA, documentar hallazgos |
| 2 | Modelos clásicos | MobileNetV2 (M1) + Random Forest (M2) con métricas |
| 3 | Modelos avanzados | EfficientNet-B0 (M1) + BETO (M2) entrenados |
| 4 | Evaluación rigurosa | Matrices de confusión, comparativas, análisis de viabilidad |
| 5 | Dashboard integrado | Streamlit con EDA, M1, M2 y dashboard integrado |
| 6 | Pulido + despliegue | Streamlit Cloud, video de respaldo, documentación |
| 7 | Presentación | Ensayos, slides, demo final |

## Equipo

Proyecto Final — Curso de Inteligencia Artificial · 6 integrantes

Ver roles, responsabilidades y cronograma detallado en [TEAM.md](TEAM.md).
