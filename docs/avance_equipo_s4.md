# Informe de Avance — Equipo completo

**Fecha:** 24/06/2026 — Semana 4 (20–26 Jun)  
**Evaluador:** Jorge Flores (líder del proyecto)  
**Próximas fechas clave:** Segunda exposición 07/07/2026 · Entrega final 14/07/2026

---

## Resumen ejecutivo

| Integrante | Rol | Cumplimiento S1–S4 | Estado |
|------------|-----|--------------------|--------|
| Anthony | M1 lead — EfficientNet-B0 | ~80% | ⚠️ S4 bloqueada por César |
| Jorge | M2 lead — RF + contratos + arquitectura | ~75% | ⚠️ Notebooks ejecutados — comparativa pendiente de Ghinno |
| César | M1 — MobileNetV2 | ~10% | ❌ Sin commits propios, modelo sin entrenar |
| Ghinno | M2 — BETO fine-tuning | ~15% | ❌ Notebook sin ejecutar, modelo ausente |
| Jeremy | Dashboard lead — routing + components | ~60% | ⚠️ Deploy pendiente, sin pruebas e2e |
| Víctor | Dashboard — páginas y lógica | ~40% | ⚠️ EDA y dashboard integrado con datos falsos |

---

## Anthony — Módulo 1: EfficientNet-B0

### Completado ✅

| Tarea | Detalle |
|-------|---------|
| EDA Fashion Products | 44.446 filas, 3 clases filtradas, hallazgos correctos |
| Entrenamiento EfficientNet-B0 | Fase 1 (freeze) + Fase 2 (fine-tuning), 20 épocas totales |
| Data augmentation | `RandomFlip`, `RandomRotation`, `RandomBrightness` en `train.py` |
| Modelo `efficientnet_b0.keras` | Commiteado en `main` (49 MB) |
| Evaluación con `src/evaluation/` | `compute_metrics` + `plot_confusion_matrix` + `assess_viability` |
| Accuracy en test | **99.6%** — supera ampliamente la meta de ≥85% |
| Coordinación `IMAGE_CLASSES` | Reducción a 3 clases mergeada a `main`, César re-entrenará sobre esta base |

### Pendiente ❌

| Tarea | Semana | Bloqueo |
|-------|--------|---------|
| Comparativa MobileNetV2 vs EfficientNet (métricas, matrices, tiempo de inferencia) | S4 — vence 26/06 | Bloqueado hasta que César entregue `mobilenetv2.keras` |
| Limpiar 7 TODOs residuales del notebook EDA | Antes del 07/07 | No bloqueante |
| Reformatear conclusiones del EDA (sacar del bloque de código Markdown) | Antes del 07/07 | No bloqueante |

---

## Jorge — Módulo 2: Random Forest + arquitectura compartida

### Completado ✅

| Tarea | Detalle |
|-------|---------|
| `src/evaluation/` completo | `metrics.py`, `confusion_matrix.py`, `viability.py` implementados y testeados |
| `src/image_classifier/predict.py` | Selector `model_name` soportando `efficientnet` y `mobilenetv2` |
| `src/sentiment_analyzer/predict.py` | Selector `model_name` soportando `beto` y `random_forest` |
| Contratos API | `docs/api_contracts.md` actualizado y alineado |
| `IMAGE_CLASSES` coordinado | Reducción a 3 clases mergeada a `main` |
| Migración PyTorch → TensorFlow | Toda la base de código migrada |
| EDA Amazon Reviews | `notebooks/00_eda_sentimiento.ipynb` ejecutado en Colab — 11/11 celdas con output, commiteado |
| Notebook Random Forest | `notebooks/02_sentiment_baseline_randomforest.ipynb` ejecutado en Colab — 9/10 celdas con output, commiteado |
| `tfidf_baseline.pkl` | Generado y disponible localmente (1.9 GB — no se versiona por tamaño, excluido en `.gitignore`) |

### Pendiente ❌

| Tarea | Semana | Detalle |
|-------|--------|---------|
| Comparativa RF vs BETO (sección en notebook) | S4 — vence 26/06 | Bloqueado hasta que Ghinno entregue `beto_finetuned/` y su notebook con outputs |

---

## César — Módulo 1: MobileNetV2

### Completado ✅

Ningún entregable propio commiteado. `predict.py` ya soporta MobileNetV2 (implementado por Jorge).

### Pendiente ❌

| Tarea | Semana | Detalle |
|-------|--------|---------|
| Entrenar MobileNetV2 en Colab y commitear `mobilenetv2.keras` | S2 (vencida) | `notebooks/01_image_classifier_mobilenetv2.ipynb` tiene 17 celdas estructuradas pero ninguna con output |
| Integrar `src/evaluation/` en notebook (cell-14 tiene `# TODO:`) | S2 (vencida) | Una línea de cambio — reemplazar sklearn directo por `compute_metrics` y `plot_confusion_matrix` |
| Comparativa MobileNetV2 vs EfficientNet | S4 — vence 26/06 | Imposible sin `mobilenetv2.keras` |

> **Situación crítica:** César es el bloqueante principal del equipo. Sin `mobilenetv2.keras` no hay comparativa de S4 para Anthony ni para él mismo, y la página M1 del dashboard solo puede usar EfficientNet.  
> El notebook ya tiene la estructura completa — solo hace falta ejecutarlo en Colab con GPU T4.

---

## Ghinno — Módulo 2: BETO fine-tuning

### Completado ✅

| Tarea | Detalle |
|-------|---------|
| Análisis comparativo RF vs BETO documentado | `docs/m2_comparativa_beto_vs_rf.md` — métricas por clase, análisis de errores, recomendación de producción |

### Pendiente ❌

| Tarea | Semana | Detalle |
|-------|--------|---------|
| Ejecutar fine-tuning BETO en Colab y commitear `beto_finetuned/` | S3 (vencida) | `notebooks/03_sentiment_beto_finetuning.ipynb` — 6 celdas sin output; modelo ausente |
| Alcanzar meta F1-macro ≥ 0.80 | S3 | F1-macro reportado: **0.7455** — no supera la meta. La clase Neutro arrastra el promedio (F1=0.51). Evaluar si más épocas o ajuste de hiperparámetros mejoran el resultado |
| Comparativa RF vs BETO en el notebook (sección, no solo `.md`) | S4 — vence 26/06 | El `.md` existe pero el notebook no tiene esta sección integrada |

> **Nota sobre métricas:** según `docs/m2_comparativa_beto_vs_rf.md`, ningún modelo alcanza su meta (RF: 0.6488 vs ≥0.70; BETO: 0.7455 vs ≥0.80). Ambos tienen dificultad estructural con la clase Neutro. Considerar documentar el análisis de por qué la meta no se alcanza y qué se intentó — es un entregable válido para la exposición.

---

## Jeremy — Dashboard: routing, componentes y deploy

### Completado ✅

| Tarea | Detalle |
|-------|---------|
| `app/app.py` | Routing completo, menú principal, configuración de páginas (124 líneas, sin TODOs) |
| `app/components/charts.py` | `sentiment_pie`, `class_distribution_chart` y otros — 109 líneas |
| `app/components/metrics_card.py` | `render_kpi_row` — 43 líneas |
| `app/components/sidebar.py` | Sidebar reutilizable — 8 líneas |
| Integración matrices de confusión | Páginas M1 y M2 muestran `plot_confusion_matrix` desde `src/evaluation/` |
| Tab "Evaluación del modelo" | KPIs y matriz de confusión disponibles en páginas M1 y M2 |

### Pendiente ❌

| Tarea | Semana | Detalle |
|-------|--------|---------|
| Deploy en Streamlit Community Cloud | S5 (27 Jun – 03 Jul) | No hay `.streamlit/config.toml`; URL pública ausente |
| Pruebas end-to-end documentadas | S5 | App levanta sin modelos entrenados (FileNotFoundError manejado), pero sin prueba formal |

---

## Víctor — Dashboard: páginas y lógica de negocio

### Completado ✅

| Página | Detalle |
|--------|---------|
| `clasificador_productos.py` | Funcional — carga de imagen, inferencia vía `predict()`, tab de evaluación con `src/evaluation/` (123 líneas) |
| `analisis_sentimiento.py` | Funcional — input de texto, inferencia vía `predict()`, visualización de scores (114 líneas) |

### Pendiente ❌

| Tarea | Semana | Detalle |
|-------|--------|---------|
| `eda.py` con datos reales | S5 | Datos completamente hardcodeados (`"Shirts": 1200`, `"Tshirts": 950`, etc.) — sin conexión a los hallazgos del EDA real |
| `dashboard_integrado.py` con métricas reales | S4 / S5 | KPIs vienen de `session_state` pero el gráfico donut usa valores hardcodeados (`values=[4118, 739, 423]`); logs de sistema son estáticos |
| Página M1 con selector MobileNetV2 operativo | Después de César | El selector existe pero MobileNetV2 lanza `FileNotFoundError` hasta que César entregue el modelo |

---

## Semáforo general — Semana 4

```
Anthony  ████████░░  80%   ✅ Núcleo entregado — pendiente comparativa S4
Jorge    ███████░░░  75%   ⚠️ Notebooks ejecutados — comparativa S4 bloqueada por Ghinno
César    █░░░░░░░░░  10%   ❌ CRÍTICO — modelo MobileNetV2 sin entrenar
Ghinno   ██░░░░░░░░  15%   ❌ CRÍTICO — modelo BETO sin commitear, meta no alcanzada
Jeremy   ██████░░░░  60%   ⚠️ Deploy y pruebas e2e pendientes (S5)
Víctor   ████░░░░░░  40%   ⚠️ EDA y dashboard integrado con datos placeholder
```

## Acciones críticas antes del 26/06 (2 días)

1. **César** → Ejecutar `01_image_classifier_mobilenetv2.ipynb` en Colab y commitear `mobilenetv2.keras`
2. **Ghinno** → Ejecutar `03_sentiment_beto_finetuning.ipynb` en Colab y commitear `beto_finetuned/`

Sin los entregables de César y Ghinno la Semana 5 (integración y dashboard completo) no puede arrancar.
