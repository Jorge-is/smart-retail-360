# Informe de Avance — Equipo completo

**Fecha:** 24/06/2026 — Semana 4 (20–26 Jun) · **Actualización Víctor:** 28/06/2026 — inicio Semana 5 · **Actualización César/M1:** 02/07/2026  
**Evaluador:** Jorge Flores (líder del proyecto)  
**Próximas fechas clave:** Segunda exposición 07/07/2026 · Entrega final 14/07/2026

---

## Resumen ejecutivo

| Integrante | Rol | Cumplimiento S1–S4 | Estado |
|------------|-----|--------------------|--------|
| Anthony | M1 lead — EfficientNet-B0 | ~80% | ⚠️ S4 bloqueada por César |
| Jorge | M2 lead — RF + contratos + arquitectura | ~75% | ⚠️ Notebooks ejecutados — comparativa pendiente de Ghinno |
| César | M1 — MobileNetV2 | ~50% 🆙 | ⚠️ Modelo entrenado y commiteado (94% Fase 1), bug de clases corregido, comparativa escrita — fine-tuning y evaluación con `src/evaluation/` siguen pendientes; ver nota de autoría |
| Ghinno | M2 — BETO fine-tuning | ~15% | ❌ Notebook sin ejecutar, modelo ausente |
| Jeremy | Dashboard lead — routing + components | ~60% | ⚠️ Deploy pendiente, sin pruebas e2e |
| Víctor | Dashboard — páginas y lógica | ~55% 🆙 | ⚠️ Session state real implementado — KPIs y matrices con datos incorrectos/falsos |

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
| Comparativa MobileNetV2 vs EfficientNet (métricas, matrices, tiempo de inferencia) | S4 — vencida 26/06 | Ya no bloqueada — `mobilenetv2.keras` disponible desde el 01/07 — pero pendiente de escribir |
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

> **Actualizado al 02/07/2026** — rama `module/m1-cesar`, commit `63d5c38`.

### Completado ✅

| Tarea | Detalle |
|-------|---------|
| Entrenamiento MobileNetV2 (Fase 1 — cabeza, backbone congelado) | `notebooks/01_image_classifier_mobilenetv2.ipynb` ejecutado en Colab, 23 celdas con output |
| Modelo `mobilenetv2.keras` | Commiteado en la rama (9.66 MB) — desbloquea a Anthony y a Víctor |
| Accuracy en test | **93.83%** — supera ampliamente la meta de ≥78% (TEAM.md) |
| Data augmentation | `RandomFlip`, `RandomRotation`, `RandomBrightness` (igual que EfficientNet) |
| Integración en `predict.py` | Ya soportado (`_MODEL_PATHS["mobilenetv2"]`, implementado por Jorge) — ahora con archivo real detrás, deja de lanzar `FileNotFoundError` |
| Bug de mapeo de clases (`IMAGE_CLASSES`) | **Corregido 02/07/2026** — ver nota abajo |
| Comparativa MobileNetV2 vs EfficientNet | **Escrita 02/07/2026** — sección 8 agregada al notebook, con métricas reales de ambos modelos y análisis de viabilidad |

### Pendiente ❌ / Desviaciones ⚠️

| Tarea | Semana | Detalle |
|-------|--------|---------|
| Fine-tuning completo (Fase 2) | S2 (vencida) | Celdas 14–15 del notebook están comentadas (`# for layer in model.layers: layer.trainable = True ...`) — nunca se ejecutaron. El resultado de 93.83% es **solo Fase 1** (cabeza), no el entrenamiento "head + fine-tuning" que pide TEAM.md. Sigue pendiente de correr en Colab |
| Integrar `src/evaluation/` en el notebook | S2 (vencida) | El TODO original desapareció, pero **no fue reemplazado** — la evaluación (celda 18) usa `sklearn.metrics.classification_report`/`confusion_matrix` directo, no `compute_metrics`/`plot_confusion_matrix` compartidos. Además queda una celda 19 duplicada y sin ejecutar (código muerto). No se tocó esta sesión — requiere ejecución en Colab para validar que el output no cambie |
| Medir tiempo de inferencia | S4 | Ni el notebook de MobileNetV2 ni el de EfficientNet miden latencia — dato pedido por TEAM.md para la comparativa de S4, señalado en la nueva sección 8 pero sin resolver |

> **Nota sobre autoría:** el trabajo de entrenamiento (Fase 1) es de César; el commit `63d5c38` fue ejecutado/subido por Anthony (firmado con su identity de Git `XCypherXx <anthonycondori25@gmail.com>`, la misma que usa en sus commits de EfficientNet `137a2b5`) — probablemente porque corrieron el Colab desde la cuenta de Anthony. El avance se acredita a César; queda como nota de proceso para que a futuro cada quien commitee desde su propia identidad de Git.
>
> **Bug crítico corregido (config.py):** `src/utils/config.py` tenía `IMAGE_CLASSES = ["Apparel", "Footwear", "Accessories"]`, pero `image_dataset_from_directory` (usado en `train.py` y en ambos notebooks) asigna índices por orden **alfabético de carpeta** (`Accessories=0, Apparel=1, Footwear=2`, confirmado en la celda 7 del notebook: `Clases: train_ds.class_names`). Como `predict.py` mapea `IMAGE_CLASSES[i]` al índice `i` de salida del modelo, las etiquetas devueltas estaban **cruzadas**. **Corregido** el 02/07/2026 reordenando `IMAGE_CLASSES = ["Accessories", "Apparel", "Footwear"]` — no requiere reentrenar ningún modelo, ambos ya usaban ese orden internamente.
>
> **Segundo hallazgo del mismo bug (notebook EfficientNet):** al armar la comparativa se encontró que `notebooks/01_image_classifier_efficientnet.ipynb` (de Anthony, en `main`) tenía el mismo error pero hardcodeado localmente (`VALID_CLASSES = ["Apparel", "Footwear", "Accessories"]` en la celda de evaluación), independiente de `config.py` — el fix de arriba no lo cubría. El accuracy global (99.6%) no cambia porque es agregado, pero el desglose por clase estaba cruzado: los soportes (1694/3209/1383) coincidían exactamente entre el reporte de MobileNetV2 (bien etiquetado) y el de EfficientNet (mal etiquetado), lo que confirmó el mismo bug. **Corregido** el 02/07/2026: la celda ahora deriva `VALID_CLASSES` de `test_ds.class_names` (igual que hace este notebook) en vez de hardcodearlo, y se recalcularon a mano las cifras por clase de la celda de resultado. La celda de evaluación con `src/evaluation/` nunca se había ejecutado (sin output), así que no hay salida vieja que corregir ahí — falta que Anthony la corra en Colab para tener el reporte y la matriz de confusión reales.
>
> **Situación:** el bloqueo original (modelo sin entrenar) está resuelto, el bug de clases está corregido en ambos módulos y la comparativa ya está escrita con datos reales. Sigue pendiente: Fase 2 de fine-tuning de MobileNetV2, integración de `src/evaluation/` en su notebook, y medir tiempo de inferencia en ambos modelos.

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

> **Actualizado al 28/06/2026** — rama `cambios-victor`, commit `d219ab4`.

### Completado ✅

| Página / Feature | Detalle |
|-----------------|---------|
| `clasificador_productos.py` | Funcional — selector de modelo, carga de imagen, inferencia, feedback con `model_label`/`model_desc`, error handling diferenciado |
| `analisis_sentimiento.py` | Funcional — selector BETO / RF, input de texto, scores, feedback visual, error handling diferenciado |
| Session state cross-page | `last_image_prediction`, `last_sentiment_prediction`, `sentiment_counts`, `avg_sentiment` actualizados en tiempo real al predecir |
| Dashboard donut conectado | Gráfico de sentimiento lee de `session_state["sentiment_counts"]` — ya no usa `[4118, 739, 423]` estático |
| `eda.py` refactorizado | Estructura con secciones M1 y M2, placeholders explícitos vs datos falsos silenciosos (anterior) |

### Pendiente ❌ / Bugs activos ⚠️

| Tarea | Semana | Detalle |
|-------|--------|---------|
| `eda.py` con datos reales | S5 — vence 03/07 | 4 secciones siguen siendo `st.info("🔄 Placeholder...")` — hallazgos de Anthony y Jorge disponibles en notebooks |
| KPIs del dashboard con valores **incorrectos** | CRÍTICO — antes del deploy | EfficientNet: `"87.5%"` (real: **99.6%**); RF F1: `"0.7124"` (real: **0.6488**); RF Acc: `"72.8%"` (real: **73.84%**); BETO Acc: `"75.8%"` (real: **79%**) |
| Bug `col2` duplicado en `dashboard_integrado.py` | CRÍTICO — layout roto | `col2` redefinido dentro del `try` de matrices — "Últimas Acciones" renderiza en columna incorrecta dependiendo del entorno |
| Matrices de confusión con 6 clases y lógica incorrecta | Antes del deploy | Labels usan 6 clases ficticias (proyecto tiene 3); generación de `y_true`/`y_pred` produce siempre diagonal perfecta |
| Página M1 con selector MobileNetV2 operativo | Después de César | Desbloqueado desde el 01/07 (`mobilenetv2.keras` ya existe) — falta que Víctor lo pruebe en la página; ojo con el posible bug de `IMAGE_CLASSES` (ver sección de César) |

---

## Semáforo general — Semana 4

```
Anthony  ████████░░  80%   ✅ Núcleo entregado — falta correr evaluación con src/evaluation/ y medir inferencia
Jorge    ███████░░░  75%   ⚠️ Notebooks ejecutados — comparativa S4 bloqueada por Ghinno
César    █████░░░░░  50%   🆙 Modelo entrenado (94% Fase 1), bug de IMAGE_CLASSES corregido, comparativa escrita — falta Fase 2 y evaluación compartida
Ghinno   █████░░░░░  50%   ⚠️ Notebook con outputs entregado tardío — análisis errores y viabilidad pendientes
Jeremy   ██████░░░░  60%   ⚠️ Deploy y pruebas e2e pendientes (S5)
Víctor   █████░░░░░  55%   ⚠️ Session state real — KPIs incorrectos y matrices con datos falsos (bugs críticos)
```

> Semáforo actualizado al 02/07/2026. César actualizado con el commit `63d5c38` de la rama `module/m1-cesar` + correcciones de esta sesión.

## Acciones críticas — actualizadas al 02/07/2026

1. **César** → ejecutar Fase 2 (fine-tuning completo, celdas 14–15 comentadas) e integrar `src/evaluation/` en el notebook en lugar de sklearn directo
2. **Anthony** → correr en Colab la celda de evaluación con `src/evaluation/` (nunca se ejecutó) para tener el reporte y la matriz de confusión reales, ahora que `VALID_CLASSES` está corregido
3. **Anthony + César** → medir tiempo de inferencia de ambos modelos, dato que la comparativa dejó marcado como faltante
4. **Ghinno** → ejecutar `03_sentiment_beto_finetuning.ipynb` en Colab y commitear `beto_finetuned/`
5. **Víctor** → probar el selector MobileNetV2 en la página M1 ahora que el modelo existe y el bug de `IMAGE_CLASSES` está corregido

Sin los entregables de Ghinno, la Semana 5 (integración y dashboard completo) no puede cerrarse con confianza.
