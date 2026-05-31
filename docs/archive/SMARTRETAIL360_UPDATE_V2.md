# SmartRetail 360 — Actualización del Plan (v2)

> **Documento de actualización** del proyecto SmartRetail 360 tras feedback del profesor en la primera exposición.
> Este documento complementa al original `SMARTRETAIL360_CONTEXT.md` (no lo reemplaza).
> Claude Code debe aplicar estos cambios sobre el skeleton ya construido.

---

## 1. Resumen de la actualización

Tras la primera exposición, el profesor entregó feedback que obliga a ajustar el alcance, el stack y los modelos. Los cambios principales son:

1. **Reducción de alcance:** el proyecto se enfoca en **Módulo 1 (Imágenes) y Módulo 2 (Sentimiento)** para la entrega final. El Módulo 3 (Ventas) **NO se elimina del skeleton**, queda como módulo opcional/futuro si sobra tiempo.
2. **Polars reemplaza a Pandas** en todo el proyecto.
3. **Random Forest reemplaza a Logistic Regression** como modelo clásico de sentimiento.
4. **MobileNetV2 se suma a EfficientNet-B0** como segundo modelo de clasificación de imágenes para comparativa.
5. **EDA explícito** se incorpora al flujo de cada módulo.
6. **Matriz de confusión y análisis de viabilidad** se incorporan como salidas estándar de cada módulo.
7. **Cronograma reducido a 7 semanas** (queda una semana menos del plan original).

---

## 2. Contexto del cambio

| Aspecto | Plan original (v1) | Plan actualizado (v2) |
|---|---|---|
| Módulos activos | 3 (Imágenes, Sentimiento, Ventas) | **2 activos + 1 opcional** |
| Duración | 8 semanas | **7 semanas** |
| Manejo de datos | Pandas | **Polars** |
| ML clásico (M2) | Logistic Regression | **Random Forest** |
| Modelos M1 | EfficientNet-B0 | **MobileNetV2 + EfficientNet-B0** |
| EDA | Implícito | **Explícito, notebook dedicado por módulo** |
| Matriz de confusión | Mencionada | **Obligatoria con análisis de viabilidad** |
| Estado del Módulo 3 | Activo | **Opcional / placeholder** |

---

## 3. Cambios al stack tecnológico

### 3.1 Reemplazos directos

| Componente | Antes | Ahora | Notas |
|---|---|---|---|
| Manejo de datos | `pandas` | `polars` | 5-10× más rápido, mejor manejo de memoria, sintaxis moderna |
| ML clásico (sentimiento) | `LogisticRegression` (sklearn) | `RandomForestClassifier` (sklearn) | Más potente, captura relaciones no lineales |

### 3.2 Adiciones

| Componente | Tecnología | Función |
|---|---|---|
| Segundo modelo CNN | MobileNetV2 (Keras Applications) | Modelo de comparación frente a EfficientNet-B0 |
| EDA estructurado | matplotlib + seaborn (o Plotly) | Análisis exploratorio en notebook dedicado |
| Análisis de viabilidad | scikit-learn + matriz de confusión + classification_report | Evaluación crítica de cada modelo |

### 3.3 `requirements.txt` actualizado

```txt
# Core
python>=3.10
numpy>=1.24
polars>=0.20          # NUEVO: reemplaza a pandas como motor principal
pandas>=2.0           # Mantener (algunas librerías como Prophet aún lo requieren)
scikit-learn>=1.3

# Deep Learning (TensorFlow, no PyTorch)
tensorflow>=2.15
transformers>=4.35
datasets>=2.14

# Computer Vision
Pillow>=10.0
albumentations>=1.3

# NLP
nltk>=3.8
wordcloud>=1.9

# Time Series (Modulo 3 opcional, NO eliminar)
prophet>=1.1
xgboost>=2.0

# Dashboard
streamlit>=1.28
plotly>=5.17
matplotlib>=3.7
seaborn>=0.13         # NUEVO: para EDA
streamlit-option-menu>=0.3

# Storage
joblib>=1.3

# Utils
python-dotenv>=1.0
tqdm>=4.66
```

---

## 4. Cambios en la estructura del proyecto

### 4.1 Mantener la estructura existente

**NO eliminar** ninguna carpeta del skeleton. Específicamente:
- `src/sales_predictor/` se mantiene **intacto** como módulo placeholder/opcional
- `notebooks/04_sales_predictor_prophet.ipynb` se mantiene como notebook opcional
- En el dashboard, la página de ventas queda como "Próximamente" o se oculta del menú principal

### 4.2 Nuevos archivos/notebooks a agregar

Agregar dentro de `notebooks/`:

```
notebooks/
├── 00_eda_imagenes.ipynb              # NUEVO: EDA dataset Fashion Products
├── 00_eda_sentimiento.ipynb           # NUEVO: EDA dataset Amazon Reviews
├── 01_image_classifier_mobilenetv2.ipynb  # NUEVO: entrenamiento MobileNetV2
├── 01_image_classifier_efficientnet.ipynb # RENOMBRADO (antes era el único)
├── 02_sentiment_baseline_randomforest.ipynb  # RENOMBRADO (antes era logistic)
├── 03_sentiment_beto_finetuning.ipynb
└── 04_sales_predictor_prophet.ipynb   # OPCIONAL: dejar como está
```

### 4.3 Cambios en `src/image_classifier/`

Agregar archivo para el segundo modelo:

```
src/image_classifier/
├── __init__.py
├── model_efficientnet.py     # RENOMBRADO de model.py
├── model_mobilenetv2.py      # NUEVO
├── train.py                  # ACTUALIZAR: parametrizar para entrenar cualquiera de los dos
├── predict.py                # ACTUALIZAR: soportar selección de modelo
├── preprocess.py             # Sin cambios mayores
└── eda.py                    # NUEVO: funciones reutilizables para EDA
```

### 4.4 Cambios en `src/sentiment_analyzer/`

```
src/sentiment_analyzer/
├── __init__.py
├── baseline_model.py         # ACTUALIZAR: cambiar LogisticRegression por RandomForest
├── beto_model.py             # Sin cambios
├── train.py
├── predict.py
├── preprocess.py
└── eda.py                    # NUEVO: funciones reutilizables para EDA
```

### 4.5 Nuevo módulo de evaluación compartida

Crear nuevo paquete para métricas y análisis de viabilidad reutilizables entre módulos:

```
src/evaluation/
├── __init__.py
├── metrics.py                # Cálculo de accuracy, precision, recall, F1
├── confusion_matrix.py       # Generación + visualización de matriz de confusión
└── viability.py              # Función que evalúa "¿el modelo es viable?" según umbrales
```

### 4.6 Cambios en `app/`

```
app/
├── app.py
├── pages/
│   ├── 1_clasificador_productos.py     # ACTUALIZAR: permitir elegir entre 2 modelos
│   ├── 2_analisis_sentimiento.py       # ACTUALIZAR: permitir elegir entre 2-3 modelos
│   ├── 3_prediccion_ventas.py          # MANTENER pero marcar como "Próximamente"
│   ├── 4_dashboard_integrado.py        # ACTUALIZAR: integrar solo M1+M2
│   └── 0_eda.py                        # NUEVO: página dedicada a mostrar EDA
└── ...
```

---

## 5. Cambios módulo por módulo

### 5.1 Módulo 1 — Clasificación de productos por imagen

**Cambios principales:**
- Implementar **dos modelos**: MobileNetV2 (modelo de comparación) y EfficientNet-B0 (modelo final).
- Ambos usan **transfer learning** desde pesos preentrenados en ImageNet.
- Framework: **TensorFlow / Keras** (no PyTorch).
- Carga del dataset y exploración con **Polars**.

**EDA esperado (notebook `00_eda_imagenes.ipynb`):**
- Distribución de clases (¿está balanceado?)
- Conteo de imágenes por `masterCategory` y `subCategory`
- Visualización de muestras por categoría (4-5 imágenes por clase)
- Estadísticas de dimensiones de imagen (verificar si todas son 60×80 o varían)
- Detección de metadata inconsistente o imágenes faltantes
- Decisión documentada: usar `masterCategory` (5 clases) o `subCategory` (~45 clases)

**Pipeline de entrenamiento:**
```
1. Cargar dataset con Polars
2. Train/Val/Test split estratificado (70/15/15)
3. Data augmentation con tf.keras.layers.Random* (RandomFlip, RandomRotation, RandomBrightness)
4. Cargar modelo preentrenado (MobileNetV2 o EfficientNetB0) con include_top=False
5. Congelar base, agregar GlobalAveragePooling2D + Dropout + Dense final
6. Compilar con Adam(lr=1e-3), categorical_crossentropy
7. Entrenar 5-10 épocas (solo la cabeza)
8. Descongelar, fine-tuning con Adam(lr=1e-4) por 5-10 épocas más
9. Guardar modelo entrenado (.keras o .h5)
```

**Métricas obligatorias:**
- Accuracy en test set
- Precision, Recall, F1 por clase
- Matriz de confusión (visualizada con seaborn)
- Tiempo de inferencia promedio por imagen

**Análisis de viabilidad:**
- Comparar accuracy de ambos modelos
- Identificar clases con mayor confusión
- Conclusión explícita: "¿es viable para producción?" con justificación basada en métricas

---

### 5.2 Módulo 2 — Análisis de sentimiento

**Cambios principales:**
- Modelo clásico: **Random Forest** (no Logistic Regression).
- Mantener **BETO** como modelo final con fine-tuning.
- Opcional: agregar Logistic Regression como tercer modelo si sobra tiempo.
- Carga y manejo de reseñas con **Polars**.

**EDA esperado (notebook `00_eda_sentimiento.ipynb`):**
- Distribución de ratings (1-5) y de sentimientos (positivo/neutro/negativo tras mapeo)
- Longitud de reseñas (caracteres y palabras): histograma
- Palabras más frecuentes por categoría de sentimiento
- Nubes de palabras (positivas vs negativas)
- Detección de reseñas vacías, duplicadas o con caracteres anómalos
- Análisis de balance de clases y decisión sobre técnica de balanceo (class_weight, oversampling, etc.)

**Pipeline Random Forest:**
```
1. Cargar dataset con Polars, mapear ratings → 3 clases
2. Train/Val/Test split estratificado
3. Preprocesamiento: lowercase, normalización Unicode, tokenización, stopwords
4. Vectorización TF-IDF (max_features=10000, ngram_range=(1,2))
5. RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced", n_jobs=-1)
6. Entrenar
7. Guardar modelo con joblib
```

**Pipeline BETO:**
```
1. Tokenización con AutoTokenizer (dccuchile/bert-base-spanish-wwm-uncased)
2. Cargar modelo TFAutoModelForSequenceClassification con num_labels=3
3. Fine-tuning con Adam(lr=2e-5), 2-3 épocas, batch_size=16-32
4. Guardar modelo con save_pretrained
```

**Métricas obligatorias:**
- Accuracy global
- Precision, Recall, F1 macro y por clase
- Matriz de confusión (visualizada)
- Comparación lado a lado de Random Forest vs BETO

**Análisis de viabilidad:**
- Tabla comparativa de modelos con todas las métricas
- Análisis de errores: ejemplos de reseñas mal clasificadas por cada modelo
- Conclusión: "¿qué modelo recomendamos para producción y por qué?"

---

### 5.3 Módulo 3 — Predicción de ventas (OPCIONAL)

**Estado:** placeholder, no se trabaja activamente.

**Acciones:**
- **NO eliminar** `src/sales_predictor/` ni el notebook correspondiente.
- En el dashboard, la página de ventas debe mostrar un mensaje:
  > "🚧 Módulo en desarrollo — Disponible en versión futura del proyecto"
- Si al llegar a la semana 5 o 6 el equipo va bien con M1 y M2, **considerar retomarlo** con alcance reducido (solo Prophet sin XGBoost, sin integración con sentimiento).
- Mantener `requirements.txt` con `prophet` y `xgboost` para no romper el skeleton.

---

## 6. Nuevo paquete `src/evaluation/`

Crear módulo de evaluación compartida que cualquier módulo pueda importar.

### 6.1 `src/evaluation/metrics.py`

```python
"""
Funciones de cálculo de métricas reutilizables.
"""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def compute_metrics(y_true, y_pred, labels=None):
    """
    Returns:
        {
            "accuracy": float,
            "precision_macro": float,
            "recall_macro": float,
            "f1_macro": float,
            "report": str  # classification_report como string
        }
    """
    # TODO: implementar
    pass
```

### 6.2 `src/evaluation/confusion_matrix.py`

```python
"""
Generación y visualización de matrices de confusión.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels, title="Matriz de Confusión", save_path=None):
    """
    Genera y opcionalmente guarda una matriz de confusión visualizada.
    """
    # TODO: implementar con seaborn heatmap
    pass
```

### 6.3 `src/evaluation/viability.py`

```python
"""
Análisis de viabilidad del modelo basado en umbrales definidos.
"""

VIABILITY_THRESHOLDS = {
    "image_classifier": {"accuracy_min": 0.85, "f1_macro_min": 0.80},
    "sentiment_analyzer": {"accuracy_min": 0.80, "f1_macro_min": 0.75},
}

def assess_viability(metrics: dict, module: str) -> dict:
    """
    Returns:
        {
            "is_viable": bool,
            "reasons": list[str],
            "recommendation": str  # "Apto para producción" | "Requiere mejoras" | ...
        }
    """
    # TODO: implementar
    pass
```

---

## 7. Cambios en los contratos de API entre módulos

Actualizar la firma de `predict()` para incluir el modelo seleccionado en M1 y M2:

```python
# src/image_classifier/predict.py
def predict(image, model_name: str = "efficientnet") -> dict:
    """
    Args:
        image: PIL.Image
        model_name: "efficientnet" | "mobilenetv2"
    Returns:
        {
            "model_used": str,
            "top_prediction": str,
            "top_3": [{"class": str, "confidence": float}, ...],
            "inference_time_ms": float
        }
    """

# src/sentiment_analyzer/predict.py
def predict(text: str, model_name: str = "beto") -> dict:
    """
    Args:
        text: str
        model_name: "beto" | "random_forest" | "logistic_regression" (opcional)
    Returns:
        {
            "model_used": str,
            "sentiment": "positive" | "neutral" | "negative",
            "confidence": float,
            "scores": {"positive": float, "neutral": float, "negative": float}
        }
    """
```

---

## 8. Cronograma actualizado (7 semanas)

| Semana | Foco | Entregables |
|---|---|---|
| 1 | Setup + EDA | Actualizar repo con cambios, ejecutar EDA de ambos datasets, documentar hallazgos |
| 2 | Modelos clásicos / baselines | MobileNetV2 (M1) + Random Forest (M2) funcionando con métricas |
| 3 | Modelos avanzados | EfficientNet-B0 (M1) + BETO (M2) entrenados |
| 4 | Evaluación rigurosa | Matrices de confusión, comparativas, análisis de viabilidad documentado |
| 5 | Dashboard integrado | Streamlit con páginas de EDA, M1, M2 y dashboard integrado funcionando |
| 6 | Pulido + despliegue | Streamlit Cloud, video de respaldo, ajustes visuales, documentación final |
| 7 | Presentación | Ensayos, slides actualizados, demo final |

**Buffer:** si al final de semana 5 todo está bien, considerar retomar M3 con alcance reducido para semana 6.

---

## 9. Distribución del equipo actualizada (6 personas)

| Rol | Personas | Responsabilidades |
|---|---|---|
| Módulo 1 — Imágenes | 2 | EDA, MobileNetV2, EfficientNet-B0, métricas, análisis viabilidad |
| Módulo 2 — Sentimiento | 2 | EDA, Random Forest, BETO, métricas, análisis viabilidad |
| Dashboard + integración | 1 | Streamlit con M1+M2, página EDA, despliegue |
| Documentación + presentación | 1 | Informe, slides, video demo, coordinación de entregables |

---

## 10. Métricas objetivo actualizadas

| Módulo | Modelo | Métrica | Objetivo |
|---|---|---|---|
| M1 — Imágenes | MobileNetV2 | Accuracy en test | ≥ 78% |
| M1 — Imágenes | EfficientNet-B0 | Accuracy en test | ≥ 85% |
| M2 — Sentimiento | Random Forest | F1-macro | ≥ 0.70 |
| M2 — Sentimiento | BETO | F1-macro | ≥ 0.80 |

**Análisis de viabilidad obligatorio:** cada módulo debe entregar una conclusión documentada de si su modelo final es viable o no, basada en las métricas observadas y la matriz de confusión.

---

## 11. Lista priorizada de tareas para Claude Code

Al recibir este documento, Claude Code debe actualizar el skeleton existente con las siguientes acciones, **en este orden**:

### Prioridad ALTA

1. **Actualizar `requirements.txt`** según sección 3.3 (agregar polars, seaborn; mantener pandas para Prophet).
2. **Actualizar `src/sentiment_analyzer/baseline_model.py`**: reemplazar `LogisticRegression` por `RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)`.
3. **Crear `src/image_classifier/model_efficientnet.py`** (renombrar el actual `model.py` si existe).
4. **Crear `src/image_classifier/model_mobilenetv2.py`** con la misma estructura.
5. **Actualizar `src/image_classifier/train.py`** para parametrizar qué modelo entrenar.
6. **Actualizar contratos de `predict()`** según sección 7 (agregar parámetro `model_name`).
7. **Crear paquete `src/evaluation/`** con los 3 archivos descritos en sección 6.

### Prioridad MEDIA

8. **Crear notebooks de EDA**: `notebooks/00_eda_imagenes.ipynb` y `notebooks/00_eda_sentimiento.ipynb` con estructura inicial (celdas con TODOs y secciones definidas).
9. **Renombrar notebooks existentes** según sección 4.2.
10. **Crear nuevo notebook** `notebooks/01_image_classifier_mobilenetv2.ipynb`.
11. **Actualizar `app/pages/3_prediccion_ventas.py`**: mostrar mensaje "Módulo en desarrollo".
12. **Crear `app/pages/0_eda.py`**: página de Streamlit que muestre los hallazgos del EDA.

### Prioridad BAJA

13. **Reemplazar imports de `pandas` por `polars`** donde sea directo (mantener pandas donde Prophet lo requiera).
14. **Actualizar `app/app.py`** para reflejar que M3 está deshabilitado en el menú principal.
15. **Actualizar `README.md`** del repo con el nuevo cronograma y stack.
16. **Actualizar `docs/api_contracts.md`** con las nuevas firmas de `predict()`.

### NO HACER

- ❌ NO eliminar `src/sales_predictor/`
- ❌ NO eliminar `notebooks/04_sales_predictor_prophet.ipynb`
- ❌ NO eliminar `prophet` ni `xgboost` del `requirements.txt`
- ❌ NO eliminar imports relacionados a M3

---

## 12. Notas finales

- El proyecto mantiene su filosofía: **stack 100% open source, datasets públicos, reproducibilidad total**.
- Los modelos siguen entrenándose en **Google Colab (GPU T4 gratuita)**.
- El despliegue del dashboard sigue siendo en **Streamlit Community Cloud** (gratuito).
- **TensorFlow** se mantiene como framework de deep learning (no PyTorch).
- Si surge alguna ambigüedad durante la actualización, Claude Code debe **preferir mantener lo existente sobre eliminar**, especialmente en cosas relacionadas a M3.

---

**Fin del documento de actualización v2.**
