# SmartRetail 360 — Equipo de Trabajo

## Integrantes

| Nombre | Rol principal | Módulo |
|--------|---------------|--------|
| Jorge | Módulo 2 — lead (EDA + Random Forest + métricas + contratos API) | M2 — Sentimiento |
| Ghinno | Módulo 2 (BETO fine-tuning) | M2 — Sentimiento |
| Anthony | Módulo 1 — lead (EDA + EfficientNet-B0 + métricas + viabilidad) | M1 — Imágenes |
| César | Módulo 1 (MobileNetV2) | M1 — Imágenes |
| Víctor | Dashboard (páginas M1, M2, EDA, integrado) | Dashboard |
| Jeremy | Dashboard — lead (routing, componentes, deploy, UX) | Dashboard |

> Documentación, slides e informe son responsabilidad compartida de todo el equipo.

---

## Equipos de trabajo

### Equipo M1 — Clasificación de productos por imagen

**Integrantes:** Anthony (lead), César

**Responsabilidades:**

*Anthony — lead + modelo principal:*
- EDA del dataset Fashion Products (distribución de clases, muestras, dimensiones de imagen)
- Entrenamiento de EfficientNet-B0 con transfer learning (TensorFlow/Keras)
- Data augmentation (RandomFlip, RandomRotation, RandomBrightness)
- Generación de métricas: accuracy, precision, recall, F1 por clase
- Análisis de viabilidad comparativo entre ambos modelos
- Guardar modelos entrenados en `models/image_classifier/` (formato `.keras`)

*César — modelo baseline:*
- Entrenamiento de MobileNetV2 con transfer learning (TensorFlow/Keras)
- Matriz de confusión visualizada con seaborn
- Integrar ambos modelos en `src/image_classifier/predict.py` (selector `model_name`)

**Herramientas:** Google Colab (GPU T4), TensorFlow 2.15+, Keras Applications, Polars, scikit-learn, Matplotlib, Seaborn, Jupyter

**Archivos clave:**
- `notebooks/00_eda_imagenes.ipynb`
- `notebooks/01_image_classifier_mobilenetv2.ipynb`
- `notebooks/01_image_classifier_efficientnet.ipynb`
- `src/image_classifier/model_mobilenetv2.py`
- `src/image_classifier/model_efficientnet.py`
- `src/image_classifier/train.py`
- `src/image_classifier/predict.py`
- `src/evaluation/`

---

### Equipo M2 — Análisis de sentimiento

**Integrantes:** Jorge (líder del proyecto + lead M2), Ghinno

**Responsabilidades:**

*Jorge — lead + modelo baseline:*
- EDA del dataset Amazon Reviews (distribución de ratings, longitud de reseñas, nubes de palabras)
- Análisis de balance de clases y decisión sobre técnica de balanceo
- Entrenamiento de Random Forest (TF-IDF, n_estimators=200, class_weight="balanced")
- Generación de métricas: accuracy global, F1 macro y por clase, matriz de confusión
- Análisis de viabilidad y recomendación de modelo para producción
- Mantener el contrato de API `predict(text, model_name)` actualizado
- Guardar modelos en `models/sentiment_analyzer/`

*Ghinno — modelo principal:*
- Fine-tuning de BETO (dccuchile/bert-base-spanish-wwm-uncased, 2-3 épocas)
- Tabla comparativa Random Forest vs BETO con análisis de errores
- Guardar modelo en `models/sentiment_analyzer/beto_finetuned/`

**Herramientas:** Google Colab (GPU T4), scikit-learn, HuggingFace Transformers, TensorFlow, Polars, NLTK, WordCloud, Matplotlib, Seaborn, Jupyter, joblib

**Archivos clave:**
- `notebooks/00_eda_sentimiento.ipynb`
- `notebooks/02_sentiment_baseline_randomforest.ipynb`
- `notebooks/03_sentiment_beto_finetuning.ipynb`
- `src/sentiment_analyzer/baseline_model.py`
- `src/sentiment_analyzer/beto_model.py`
- `src/sentiment_analyzer/predict.py`
- `src/evaluation/`

---

### Equipo Dashboard + Integración

**Integrantes:** Jeremy (lead), Víctor

**Responsabilidades:**

*Víctor — páginas y lógica de negocio:*
- Página M1: selector de modelo, carga de imagen, visualización de resultados
- Página M2: selector de modelo, input de texto, visualización de scores
- Página EDA: visualizar hallazgos de ambos módulos
- Página Dashboard Integrado: flujo M1 → M2 con métricas combinadas
- Página M3: mantener placeholder "Módulo en desarrollo"
- Integrar `src/evaluation/` para mostrar matrices de confusión en la UI

*Jeremy — routing, componentes y deploy:*
- `app/app.py`: routing, menú principal, configuración de páginas
- `app/components/`: sidebar, charts, metrics_card (componentes reutilizables)
- Deploy en Streamlit Community Cloud
- Pruebas de la app end-to-end (asegurar que levanta sin modelos entrenados)
- UX y polish visual previo a la exposición

**Herramientas:** Streamlit, Plotly, Polars, PIL, Python, Git, Streamlit Community Cloud

**Archivos clave:**
- `app/app.py`
- `app/pages/clasificador_productos.py`
- `app/pages/analisis_sentimiento.py`
- `app/pages/eda.py`
- `app/pages/dashboard_integrado.py`
- `app/pages/prediccion_ventas.py`
- `app/components/`

---

## Responsabilidades transversales de Jorge (líder)

- Seguimiento semanal del avance de todos los equipos
- Revisión y aprobación de Pull Requests antes de merge a `main`
- Resolución de conflictos de integración entre módulos
- Verificar que el contrato `predict()` de cada módulo se respeta
- Toma de decisiones técnicas ante bloqueos (datasets, métricas, modelos)
- Coordinación de documentación, slides e informe con todo el equipo
- Coordinación final para la demo del 14/07

---

## Cronograma — 7 semanas

> **Fechas clave:**
> - Segunda presentación y exposición: **07/07/2026**
> - Entrega final con demo: **14/07/2026**

---

### Semana 1 — Setup + EDA
**Período:** 30 May – 05 Jun 2026

| Tarea | Responsable | Entregable |
|-------|-------------|------------|
| Revisar skeleton v2, clonar repo, instalar dependencias | Todos | Entorno local funcionando |
| EDA dataset Fashion Products (distribución clases, muestras, dimensiones) | Anthony | `notebooks/00_eda_imagenes.ipynb` completo con hallazgos |
| EDA dataset Amazon Reviews (ratings, longitud, balance de clases) | Jorge | `notebooks/00_eda_sentimiento.ipynb` completo con hallazgos |
| Estructura inicial de páginas Streamlit y routing | Jeremy | `app/app.py` y estructura de `app/pages/` funcionando |
| Página EDA básica (secciones definidas, placeholder de gráficos) | Víctor | `app/pages/eda.py` con estructura lista |
| Revisión de EDA + decisiones documentadas (clases, balanceo) | Jorge | Comentarios en PRs + decisiones registradas |

---

### Semana 2 — Modelos baseline / clásicos
**Período:** 06 Jun – 12 Jun 2026

| Tarea | Responsable | Entregable |
|-------|-------------|------------|
| Entrenamiento MobileNetV2 (transfer learning, head + fine-tuning) | César | `notebooks/01_image_classifier_mobilenetv2.ipynb` + `mobilenetv2.keras` |
| Entrenamiento Random Forest (TF-IDF 10k features, n=200, class_weight balanced) | Jorge | `notebooks/02_sentiment_baseline_randomforest.ipynb` + `tfidf_baseline.pkl` |
| Preparar notebook EfficientNet-B0 + revisar avance de César | Anthony | `notebooks/01_image_classifier_efficientnet.ipynb` con estructura lista |
| Integrar `src/evaluation/` en los notebooks (metrics + confusion matrix) | César + Jorge | Métricas y matrices generadas en ambos notebooks |
| Página M1 en Streamlit: carga de imagen, inferencia MobileNetV2 | Víctor | `clasificador_productos.py` funcional con MobileNetV2 |
| Componentes reutilizables: `metrics_card`, `charts` | Jeremy | `app/components/` listos para usar en páginas |

---

### Semana 3 — Modelos avanzados
**Período:** 13 Jun – 19 Jun 2026

| Tarea | Responsable | Entregable |
|-------|-------------|------------|
| Entrenamiento EfficientNet-B0 (fine-tuning completo) | Anthony | `notebooks/01_image_classifier_efficientnet.ipynb` + `efficientnet_b0.keras` |
| Fine-tuning BETO (2-3 épocas, batch_size=16-32, lr=2e-5) | Ghinno | `notebooks/03_sentiment_beto_finetuning.ipynb` + `beto_finetuned/` |
| Integrar ambos modelos de imagen en `predict.py` (selector `model_name`) | César | `src/image_classifier/predict.py` soportando ambos modelos |
| Integrar BETO en `predict.py` de sentimiento | Jorge | `src/sentiment_analyzer/predict.py` con BETO migrado a TF |
| Página M2 en Streamlit: input de texto, inferencia, visualización de scores | Víctor | `analisis_sentimiento.py` funcional |
| Primer borrador de slides para la exposición | Todos | Borrador con secciones repartidas por equipo |

---

### Semana 4 — Evaluación rigurosa
**Período:** 20 Jun – 26 Jun 2026

| Tarea | Responsable | Entregable |
|-------|-------------|------------|
| Comparativa MobileNetV2 vs EfficientNet-B0: métricas, matrices, tiempo de inferencia | Anthony + César | Sección "Análisis de viabilidad" en ambos notebooks |
| Comparativa Random Forest vs BETO: métricas, análisis de errores, ejemplos mal clasificados | Ghinno + Jorge | Sección "Análisis de viabilidad" en ambos notebooks |
| `src/evaluation/viability.py` funcionando con umbrales reales observados | Jorge | `viability.py` implementado y testeado |
| Página Dashboard Integrado: mostrar métricas de M1 y M2 juntas | Víctor | `dashboard_integrado.py` funcional |
| Integrar `src/evaluation/` en la UI (matrices de confusión visibles) | Jeremy | Matrices renderizadas en páginas M1 y M2 |

---

### Semana 5 — Dashboard completo + integración final
**Período:** 27 Jun – 03 Jul 2026

| Tarea | Responsable | Entregable |
|-------|-------------|------------|
| Integrar hallazgos de EDA en la página EDA de Streamlit (gráficos reales) | Víctor | `eda.py` con visualizaciones de ambos módulos |
| Actualizar páginas M1 y M2 para soportar selector de modelo completo | Víctor | Páginas con todos los modelos disponibles |
| Pruebas de integración end-to-end: upload imagen → predicción → sentimiento | Jeremy + Jorge | App corriendo sin errores con ambos modelos |
| Deploy inicial en Streamlit Community Cloud | Jeremy | URL pública de prueba funcionando |
| Slides finales listos para ensayo | Todos | Presentación completa, ensayo coordinado por Jorge |
| Ensayo general de la presentación | Todos | Repartición de secciones, tiempo ajustado |

---

### Semana 6 — Segunda presentación + pulido
**Período:** 04 Jul – 10 Jul 2026

| Hito | Fecha | Responsable |
|------|-------|-------------|
| **SEGUNDA PRESENTACION Y EXPOSICION** | **07/07/2026** | Todos |

| Tarea | Responsable | Entregable |
|-------|-------------|------------|
| Presentación y demo en vivo (07/07) | Todos | Exposición completada |
| Aplicar feedback recibido en la exposición | Jorge (coordina) | Issues abiertos + ajustes priorizados |
| Ajustes visuales y UX del dashboard post-exposición | Víctor + Jeremy | App pulida con feedback aplicado |
| Deploy final en Streamlit Community Cloud | Jeremy | URL pública definitiva |
| Empezar grabación del video demo | Todos | Borrador del video |
| Evaluación de M3 opcional: ¿hay tiempo y capacidad? | Jorge | Decisión documentada (retomar Prophet o no) |

---

### Semana 7 — Cierre + entrega final
**Período:** 11 Jul – 14 Jul 2026

| Hito | Fecha | Responsable |
|------|-------|-------------|
| **ENTREGA FINAL CON DEMO** | **14/07/2026** | Todos |

| Tarea | Responsable | Entregable |
|-------|-------------|------------|
| Video demo final grabado y editado | Todos | Video demo subido |
| README final con instrucciones completas | Todos | `README.md` definitivo |
| Informe final (objetivo, datasets, modelos, métricas, conclusiones) | Todos | Informe entregado |
| Verificación final de la app en Streamlit Cloud | Jeremy | App accesible públicamente |
| Tag de release `v1.0` en GitHub | Jorge | Release publicado en el repo |
| **Demo final en vivo** | **Todos** | Entrega 14/07/2026 |

---

## Objetivos de métricas por módulo

| Módulo | Modelo | Métrica | Objetivo |
|--------|--------|---------|----------|
| M1 | MobileNetV2 | Accuracy en test | ≥ 78% |
| M1 | EfficientNet-B0 | Accuracy en test | ≥ 85% |
| M2 | Random Forest | F1-macro | ≥ 0.70 |
| M2 | BETO | F1-macro | ≥ 0.80 |

Cada módulo entrega un análisis de viabilidad documentado usando `src/evaluation/viability.py`.

---

## Flujo de trabajo con Git

- Rama principal: `main` (protegida, solo Jorge puede mergear)
- Ramas de trabajo: `feature/<nombre>` o `module/<m1|m2|dashboard>`
- PRs obligatorios con revisión de Jorge antes de merge
- Commits en formato [Conventional Commits](https://www.conventionalcommits.org/)

---

## Nota sobre Módulo 3

`src/sales_predictor/` y `notebooks/04_sales_predictor_prophet.ipynb` se mantienen en el repo como placeholder.
Si al finalizar la semana 5 M1 y M2 están completos, el equipo evalúa retomar M3 con alcance reducido (Prophet únicamente, sin integración de sentimiento).
La decisión la toma Jorge en la semana 6.
