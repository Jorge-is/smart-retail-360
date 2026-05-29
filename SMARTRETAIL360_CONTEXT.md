# SmartRetail 360 — Contexto del Proyecto

> Documento de contexto completo para planificación y construcción del skeleton del proyecto con Claude Code.

---

## 1. Resumen ejecutivo

**SmartRetail 360** es una plataforma de inteligencia artificial para e-commerce que ayuda a tiendas online a tomar tres decisiones críticas mediante un dashboard unificado:

1. **Organizar su catálogo** clasificando productos automáticamente por imagen.
2. **Entender a sus clientes** analizando el sentimiento de sus reseñas.
3. **Planear su inventario** prediciendo la demanda futura de ventas.

El proyecto es el **Proyecto Final del curso de Inteligencia Artificial**, desarrollado por un equipo universitario de **6 integrantes**, con dificultad **media-baja** y plazo estimado de **8 semanas**.

---

## 2. Contexto académico

| Aspecto | Detalle |
|---|---|
| Curso | Inteligencia Artificial (nivel universitario) |
| Tamaño del equipo | 6 integrantes |
| Duración | 8 semanas (part-time) |
| Nivel objetivo | Media-baja |
| Entregable | Demo funcional + presentación |
| Hardware | Google Colab (GPU T4 gratuita) para entrenamiento; CPU local para inferencia |

### Temas de IA combinados

El proyecto integra **3 temas** propuestos por el profesor:

- Tema 14: **Clasificación de productos en e-commerce** (visión por computadora)
- Tema 10: **Análisis de opiniones de clientes** (NLP)
- Tema 16: **Predicción de ventas o demanda** (series temporales)

### Narrativa diferenciadora

> "Somos consultores de una tienda online ficticia. Hoy mostraremos cómo nuestra plataforma ayuda a tomar 3 decisiones críticas: qué producto es, qué piensan los clientes, y cuánto se venderá."

El elemento que une los 3 módulos y eleva el proyecto: **usar el sentimiento promedio de las reseñas como variable exógena en la predicción de ventas**.

---

## 3. Stack tecnológico

### Lenguaje y entorno

- **Python 3.10+**
- **Google Colab** (GPU T4) para entrenamiento de modelos
- **Git + GitHub** para versionado
- **CPU local** para correr el dashboard

### Módulo 1 — Clasificación de imágenes de productos

| Componente | Tecnología |
|---|---|
| Framework | PyTorch + torchvision |
| Modelo | EfficientNet-B0 preentrenado en ImageNet (transfer learning) |
| Alternativa ligera | MobileNetV2 |
| Carga de imágenes | PIL / Pillow |
| Data augmentation | albumentations (opcional) |
| Métricas | scikit-learn (classification_report, confusion_matrix) |

**Flujo de entrenamiento:**
1. Cargar EfficientNet-B0 preentrenado.
2. Congelar capas convolucionales.
3. Reemplazar última capa por `nn.Linear` con N clases del dataset.
4. Entrenar solo la cabeza por 5–10 épocas (Adam, lr=1e-3).
5. Descongelar todo y fine-tuning con lr=1e-4 por 5–10 épocas más.
6. Guardar con `torch.save` y descargar al repo.

### Módulo 2 — Análisis de sentimiento

| Componente | Tecnología |
|---|---|
| Framework | Hugging Face Transformers + PyTorch |
| Modelo principal | BETO (`dccuchile/bert-base-spanish-wwm-uncased`) |
| Baseline simple | TF-IDF + Logistic Regression (scikit-learn) |
| Carga de dataset | Hugging Face `datasets` |
| Preprocesamiento | nltk o spaCy (opcional) |
| Visualización extra | wordcloud (nubes de palabras) |

**Estrategia recomendada:** Implementar AMBOS modelos. El baseline TF-IDF da algo funcional en semana 2; BETO en semana 4 muestra mejora. Esto genera narrativa de "comparamos enfoque clásico vs deep learning, mejora de +X%".

**Configuración BETO:** `AutoModelForSequenceClassification` con `num_labels=3` (positivo / neutro / negativo). Usar `Trainer` de Hugging Face.

### Módulo 3 — Predicción de ventas

| Componente | Tecnología |
|---|---|
| Modelo principal | Prophet (Meta) |
| Comparación opcional | XGBoost con features temporales |
| Manejo de series | Pandas |
| Visualización | Plotly + Matplotlib |
| Métricas | scikit-learn (MAE, RMSE, MAPE) |

**Idea integradora (clave del proyecto):** Pasar el sentimiento promedio del módulo 2 como `regressor` adicional en Prophet o como feature en XGBoost.

### Frontend / Dashboard

| Componente | Tecnología |
|---|---|
| Framework | **Streamlit** |
| Visualización | Plotly (interactivo), Matplotlib (estático) |
| Componentes extra | streamlit-option-menu (navegación lateral) |
| Despliegue opcional | Streamlit Community Cloud o Hugging Face Spaces |

### Almacenamiento y datos

| Componente | Tecnología |
|---|---|
| Base de datos | SQLite |
| Manejo de datos | Pandas + NumPy |
| Modelos serializados | pickle (sklearn) / torch.save (PyTorch) / joblib (Prophet) |

---

## 4. Datasets seleccionados

### Módulo 1 — Imágenes

**Fashion Product Images Dataset (Small)** — Kaggle

- URL: `kaggle.com/datasets/paramaggarwal/fashion-product-images-small`
- Tamaño: ~44,000 imágenes, ~280 MB
- Clases recomendadas: `masterCategory` (5 clases) para empezar; `subCategory` (~45 clases) si hay tiempo
- Imágenes con fondo blanco, ideal para transfer learning

### Módulo 2 — Reseñas en español

**Amazon Reviews Multilingual (config `es`)** — Hugging Face

- URL: `huggingface.co/datasets/amazon_reviews_multi`
- Tamaño: ~210k reseñas en español (train split)
- Mapeo de etiquetas:
  - Rating 1–2 → negativo
  - Rating 3 → neutro
  - Rating 4–5 → positivo
- Carga directa con `datasets.load_dataset("amazon_reviews_multi", "es")`

### Módulo 3 — Series temporales

**Rossmann Store Sales** — Kaggle

- URL: `kaggle.com/competitions/rossmann-store-sales/data`
- Tamaño: ~1M filas, 1,115 tiendas, 2.5 años de historia
- Features clave: fecha, ventas, clientes, promociones, feriados, tipo de tienda
- Suficiente historia para que Prophet capture estacionalidad anual

### Estrategia de integración entre datasets

Los tres datasets son independientes (no comparten productos/tiendas). Se usa la **Opción 1 — Demos independientes con narrativa unificada**:

- Cada módulo opera sobre su propio dataset.
- El dashboard presenta los 3 como capacidades de la plataforma.
- La integración sentimiento → ventas se hace con **datos simulados** (generar reseñas sintéticas asociadas a ventas) o con un mini-dataset complementario.

---

## 5. Arquitectura del proyecto

### Estructura de directorios propuesta

```
smartretail360/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/                          # Datasets (en .gitignore si son grandes)
│   ├── raw/                       # Datos originales sin procesar
│   ├── processed/                 # Datos limpios listos para entrenar
│   └── external/                  # Datos descargados de Kaggle/HF
│
├── models/                        # Modelos entrenados serializados
│   ├── image_classifier/
│   │   └── efficientnet_b0.pt
│   ├── sentiment_analyzer/
│   │   ├── tfidf_baseline.pkl
│   │   └── beto_finetuned/
│   └── sales_predictor/
│       └── prophet_model.joblib
│
├── notebooks/                     # Notebooks de Colab para entrenamiento
│   ├── 01_image_classifier_training.ipynb
│   ├── 02_sentiment_baseline_tfidf.ipynb
│   ├── 03_sentiment_beto_finetuning.ipynb
│   └── 04_sales_predictor_prophet.ipynb
│
├── src/                           # Código fuente modular
│   ├── __init__.py
│   │
│   ├── image_classifier/
│   │   ├── __init__.py
│   │   ├── model.py               # Definición de arquitectura
│   │   ├── train.py               # Script de entrenamiento
│   │   ├── predict.py             # Función predict() — API común
│   │   └── preprocess.py          # Transforms de imágenes
│   │
│   ├── sentiment_analyzer/
│   │   ├── __init__.py
│   │   ├── baseline_model.py      # TF-IDF + Logistic Regression
│   │   ├── beto_model.py          # Wrapper BETO
│   │   ├── train.py
│   │   ├── predict.py             # Función predict() — API común
│   │   └── preprocess.py          # Limpieza de texto, tokenización
│   │
│   ├── sales_predictor/
│   │   ├── __init__.py
│   │   ├── prophet_model.py
│   │   ├── xgboost_model.py       # Opcional
│   │   ├── train.py
│   │   ├── predict.py             # Función predict() — API común
│   │   └── features.py            # Feature engineering temporal
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql             # Definición de tablas SQLite
│   │   └── db_utils.py            # Conexión y queries
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Configuración global
│       └── logging_config.py
│
├── app/                           # Dashboard Streamlit
│   ├── app.py                     # Entry point principal
│   ├── pages/
│   │   ├── 1_clasificador_productos.py
│   │   ├── 2_analisis_sentimiento.py
│   │   ├── 3_prediccion_ventas.py
│   │   └── 4_dashboard_integrado.py
│   ├── components/
│   │   ├── sidebar.py
│   │   ├── metrics_card.py
│   │   └── charts.py
│   └── assets/
│       └── logo.png
│
├── tests/                         # Tests unitarios básicos
│   ├── test_image_classifier.py
│   ├── test_sentiment_analyzer.py
│   └── test_sales_predictor.py
│
└── docs/
    ├── architecture.md
    ├── api_contracts.md           # API común entre módulos
    └── presentation_outline.md
```

### API común entre módulos

Cada módulo expone una función `predict()` con I/O estandarizada para facilitar la integración:

```python
# src/image_classifier/predict.py
def predict(image: PIL.Image) -> dict:
    """
    Returns:
        {
            "top_prediction": str,
            "top_3": [{"class": str, "confidence": float}, ...],
            "inference_time_ms": float
        }
    """

# src/sentiment_analyzer/predict.py
def predict(text: str) -> dict:
    """
    Returns:
        {
            "sentiment": "positive" | "neutral" | "negative",
            "confidence": float,
            "scores": {"positive": float, "neutral": float, "negative": float}
        }
    """

# src/sales_predictor/predict.py
def predict(store_id: int, horizon_days: int, sentiment_score: float = None) -> dict:
    """
    Returns:
        {
            "forecast": [{"date": str, "predicted_sales": float, "lower": float, "upper": float}, ...],
            "metrics": {"mae": float, "rmse": float, "mape": float}
        }
    """
```

---

## 6. Funcionalidades del dashboard

### Página 1 — Clasificador de productos
- Drag-and-drop de imagen única o lote (CSV con paths)
- Top-3 predicciones con barras de confianza
- Métricas del modelo (accuracy, F1) en sidebar

### Página 2 — Análisis de sentimiento
- Input de texto libre para análisis individual
- Carga masiva de CSV de reseñas
- Visualizaciones:
  - Distribución de sentimientos (pie chart)
  - Nube de palabras positivas vs negativas
  - Reseñas más positivas/negativas destacadas

### Página 3 — Predicción de ventas
- Selector de tienda y horizonte (7/15/30 días)
- Gráfico interactivo con histórico + predicción + intervalo de confianza
- Tabla de top productos con tendencia alza/baja
- Slider opcional: "¿qué pasa si el sentimiento promedio sube/baja X%?"

### Página 4 — Dashboard integrado
- Vista ejecutiva con KPIs de los 3 módulos
- Storytelling: "Subiste estas N imágenes, las clasificamos en X categorías, las reseñas son Y% positivas, y proyectamos Z ventas para los próximos 30 días"

---

## 7. Distribución del trabajo (6 personas)

| Rol | Personas | Responsabilidades |
|---|---|---|
| **Módulo 1 — Imágenes** | 2 | Entrenamiento EfficientNet, preprocessing, API predict, integración a Streamlit |
| **Módulo 2 — Sentimiento** | 2 | Baseline TF-IDF, fine-tuning BETO, preprocessing, API predict, integración |
| **Módulo 3 — Ventas** | 1 | Prophet model, feature engineering temporal, API predict |
| **Integración + Dashboard** | 1 | Streamlit app, conexión entre módulos, SQLite, despliegue, demo |

**Regla:** Cada persona es dueña de su área, pero TODOS revisan PRs de TODOS via GitHub.

---

## 8. Cronograma (8 semanas)

| Semana | Objetivo | Entregables |
|---|---|---|
| 1 | Setup | Repo Git, estructura de carpetas, datasets descargados, EDA básico, API contracts definidos |
| 2 | Baselines | Modelo simple funcionando en cada módulo (incluso si la precisión es baja) |
| 3 | Modelos finales | EfficientNet entrenado, BETO fine-tuned, Prophet ajustado |
| 4 | Evaluación | Métricas finales, comparaciones (TF-IDF vs BETO, Prophet vs XGBoost) |
| 5 | Integración inicial | Skeleton de Streamlit con las 3 páginas funcionando aisladas |
| 6 | Dashboard integrado | Página 4 unificada, integración sentimiento → ventas |
| 7 | Pulido | Refinamiento visual, despliegue online, video de respaldo, testing |
| 8 | Presentación | Ensayos de demo, documentación, slides finales |

---

## 9. Configuración de Google Colab

```python
# Celda inicial de cada notebook
from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers datasets prophet albumentations
```

**Tips:**
- Activar GPU: Runtime → Change runtime type → T4 GPU
- Guardar datasets en Google Drive (Colab borra archivos al cerrar sesión)
- Usar checkpoints (`torch.save`) cada época para entrenamientos largos
- Sesiones gratis: ~12h máximo, desconexión por inactividad ~90 min
- Si limita: Colab Pro ($10/mes) dividido entre 6 = $1.67 por persona

---

## 10. Riesgos identificados y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Módulos desarrollados aislados sin integración | Definir API común en semana 1; revisiones cruzadas semanales |
| Datasets en español ruidosos | Preprocesamiento sólido; empezar con dataset limpio |
| Poca data histórica para Prophet | Usar Rossmann (2.5 años, suficiente estacionalidad) |
| "Se ve como 3 proyectos pegados" | Invertir tiempo en página 4 integrada; sentimiento como regressor de ventas |
| Demo en vivo falla | Grabar video de respaldo en semana 7 |
| Colab desconecta entrenamiento largo | Checkpoints cada época; modelos guardados en Drive |

---

## 11. Métricas objetivo (transparencia académica)

| Módulo | Métrica | Objetivo razonable |
|---|---|---|
| Clasificación imágenes | Accuracy en test | ≥ 85% (5 clases), ≥ 70% (45 clases) |
| Sentimiento BETO | F1 macro | ≥ 0.80 |
| Sentimiento TF-IDF | F1 macro | ≥ 0.70 (baseline) |
| Predicción ventas | MAPE | ≤ 15% |

**Importante:** Mostrar métricas honestas en la presentación. Los jurados valoran rigor sobre números inflados.

---

## 12. Stack consolidado — requirements.txt referencial

```txt
# Core
python>=3.10
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3

# Deep Learning
torch>=2.0
torchvision>=0.15
transformers>=4.35
datasets>=2.14

# Computer Vision
Pillow>=10.0
albumentations>=1.3

# NLP
nltk>=3.8
wordcloud>=1.9

# Time Series
prophet>=1.1
xgboost>=2.0

# Dashboard
streamlit>=1.28
plotly>=5.17
matplotlib>=3.7
streamlit-option-menu>=0.3

# Storage
# sqlite3 es built-in en Python
joblib>=1.3

# Utils
python-dotenv>=1.0
tqdm>=4.66
```

---

## 13. Comandos clave para Claude Code

Al iniciar la sesión con Claude Code, las siguientes tareas son prioritarias:

1. **Crear estructura de carpetas** según sección 5.
2. **Inicializar `requirements.txt`** con el contenido de sección 12.
3. **Crear `README.md`** con resumen del proyecto y pasos de setup.
4. **Crear `.gitignore`** apropiado para Python + datos grandes + modelos.
5. **Crear archivos `__init__.py`** en cada paquete de `src/`.
6. **Generar stubs de las funciones `predict()`** en cada módulo (sección 5).
7. **Crear skeleton de `app/app.py`** y las 4 páginas de Streamlit con placeholders.
8. **Crear `docs/api_contracts.md`** documentando la API común.
9. **Crear notebooks vacíos** en `notebooks/` con celdas iniciales (mount Drive, pip install).
10. **Crear `schema.sql`** con tablas básicas para guardar resultados de predicciones.

---

## 14. Próximos pasos sugeridos tras el skeleton

1. Cada pareja descarga su dataset y hace EDA en su notebook.
2. Implementar baseline simple (semana 2).
3. Iterar hacia modelo final (semanas 3–4).
4. Integrar todo en Streamlit (semanas 5–6).
5. Pulir y presentar (semanas 7–8).

---

**Fin del documento de contexto.**
