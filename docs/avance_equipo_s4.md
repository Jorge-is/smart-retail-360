# Informe de Avance — Equipo completo

**Fecha:** 24/06/2026 — Semana 4 (20–26 Jun) · **Actualización Víctor:** 28/06/2026 — inicio Semana 5 · **Actualización César/M1:** 02/07/2026 · **Actualización Ghinno/M2:** 04/07/2026  
**Evaluador:** Jorge Flores (líder del proyecto)  
**Próximas fechas clave:** Segunda exposición 07/07/2026 · Entrega final 14/07/2026

---

## Resumen ejecutivo

| Integrante | Rol | Cumplimiento S1–S4 | Estado |
|------------|-----|--------------------|--------|
| Anthony | M1 lead — EfficientNet-B0 | ~90% 🆙 | ⚠️ Ya no bloqueado — código de evaluación y tiempo de inferencia listo, falta ejecutarlo en Colab |
| Jorge | M2 lead — RF + contratos + arquitectura | ~85% 🆙 | ✅ Comparativa RF vs BETO agregada en ambos notebooks (04/07) — sin pendientes propios |
| César | M1 — MobileNetV2 | ~70% 🆙 | ⚠️ Modelo Fase 1 entrenado y commiteado (94%), bug de clases corregido, comparativa escrita, código de Fase 2 + `src/evaluation/` + tiempo de inferencia listo — falta ejecutarlo en Colab; ver nota de autoría |
| Ghinno | M2 — BETO fine-tuning | ~70% 🆙 | ⚠️ Modelo entrenado, copiado e integrado (predict() operativo) — integración la hizo Jorge, no Ghinno; análisis de errores aún sin re-ejecutar |
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
| Comparativa MobileNetV2 vs EfficientNet (métricas, matrices, tiempo de inferencia) | S4 — vencida 26/06 | **Escrita 02/07/2026** en ambos notebooks — pendiente de refrescar con números reales una vez se ejecute Fase 2 (César) y la celda de evaluación (Anthony) |
| Ejecutar la celda de evaluación con `src/evaluation/` en Colab | S3 (vencida) | Nunca se ejecutó — código ya tiene medición de tiempo de inferencia agregada, solo falta correrlo |
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
| Comparativa RF vs BETO (sección en notebook) | **Agregada 04/07/2026** — sección "Comparativa Random Forest vs BETO" en ambos notebooks (`02_sentiment_baseline_randomforest.ipynb` sección 10, `03_sentiment_beto_finetuning.ipynb`), con tabla de métricas real (RF 0.6488 vs BETO 0.7475 F1-macro) y celda ejecutada con output real |

### Pendiente ❌

Sin pendientes propios — el último ítem (comparativa en notebook) se destrababa con la entrega de Ghinno y quedó cerrado el mismo día que se integró el modelo BETO.

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
| Comparativa MobileNetV2 vs EfficientNet | **Escrita 02/07/2026** — sección 8 en el notebook de César + sección espejo en el de Anthony (TEAM.md pide la sección en ambos), con métricas reales |
| Fine-tuning completo (Fase 2) — **código** | **Preparado 02/07/2026** — celdas 14–15 descomentadas y corregidas (antes eran texto muerto) |
| Integrar `src/evaluation/` en el notebook — **código** | **Migrado 02/07/2026** — la celda de evaluación ahora usa `compute_metrics`/`plot_confusion_matrix`/`assess_viability` en vez de sklearn directo; se eliminó la celda 19 duplicada y el hack de `shutil.copy` que saltaba la Fase 2 |
| Medición de tiempo de inferencia — **código** | **Agregado 02/07/2026** — en ambos notebooks (MobileNetV2 y EfficientNet), con `time.perf_counter()` alrededor de `model.predict()` |

### Pendiente ❌ / Desviaciones ⚠️

| Tarea | Semana | Detalle |
|-------|--------|---------|
| Ejecutar Fase 2 + evaluación en Colab (MobileNetV2) | S2/S4 (vencidas) | El código ya está listo y corregido (ver arriba), pero **nadie lo corrió todavía** — no hay TensorFlow ni el dataset disponibles fuera de Colab para ejecutarlo. Los números de accuracy/F1 por clase que circulan (93.83%, solo Fase 1) van a cambiar una vez se corra Fase 2 |
| Ejecutar evaluación en Colab (EfficientNet) | S3 (vencida) | La celda de `src/evaluation/` nunca se había ejecutado; ahora tiene también medición de tiempo de inferencia, pero sigue sin correrse — falta que Anthony la ejecute para tener el reporte y la matriz de confusión reales |
| Confirmar tiempo de inferencia real | S4 | El código ya mide latencia en ambos notebooks, pero el dato en sí no existe hasta que se ejecuten |

> **Nota sobre autoría:** el trabajo de entrenamiento (Fase 1) es de César; el commit `63d5c38` fue ejecutado/subido por Anthony (firmado con su identity de Git `XCypherXx <anthonycondori25@gmail.com>`, la misma que usa en sus commits de EfficientNet `137a2b5`) — probablemente porque corrieron el Colab desde la cuenta de Anthony. El avance se acredita a César; queda como nota de proceso para que a futuro cada quien commitee desde su propia identidad de Git.
>
> **Bug crítico corregido (config.py):** `src/utils/config.py` tenía `IMAGE_CLASSES = ["Apparel", "Footwear", "Accessories"]`, pero `image_dataset_from_directory` (usado en `train.py` y en ambos notebooks) asigna índices por orden **alfabético de carpeta** (`Accessories=0, Apparel=1, Footwear=2`, confirmado en la celda 7 del notebook: `Clases: train_ds.class_names`). Como `predict.py` mapea `IMAGE_CLASSES[i]` al índice `i` de salida del modelo, las etiquetas devueltas estaban **cruzadas**. **Corregido** el 02/07/2026 reordenando `IMAGE_CLASSES = ["Accessories", "Apparel", "Footwear"]` — no requiere reentrenar ningún modelo, ambos ya usaban ese orden internamente. Verificado corriendo `tests/test_image_classifier.py` (16/16 tests OK) contra los modelos `.keras` reales.
>
> **Segundo hallazgo del mismo bug (notebook EfficientNet):** al armar la comparativa se encontró que `notebooks/01_image_classifier_efficientnet.ipynb` (de Anthony) tenía el mismo error pero hardcodeado localmente (`VALID_CLASSES = ["Apparel", "Footwear", "Accessories"]` en la celda de evaluación), independiente de `config.py`. El accuracy global (99.6%) no cambia porque es agregado, pero el desglose por clase estaba cruzado: los soportes (1694/3209/1383) coincidían exactamente entre el reporte de MobileNetV2 (bien etiquetado) y el de EfficientNet (mal etiquetado). **Corregido**: la celda ahora deriva `VALID_CLASSES` de `test_ds.class_names`.
>
> **Restructuración del notebook de MobileNetV2 (02/07/2026):** al descomentar la Fase 2 se detectó un problema de orden — la evaluación (sección 6) leía el modelo desde `mobilenetv2.keras`, pero el guardado "oficial" solo pasaba en la sección 7, *después* de la evaluación. El notebook original resolvía esto con un `shutil.copy` que copiaba el checkpoint de Fase 1 al path final antes de evaluar — es decir, **siempre evaluaba Fase 1, nunca Fase 2**, aunque la Fase 2 se llegara a ejecutar. Se corrigió moviendo el guardado real al final de la Fase 2 (antes de evaluar) y la sección 7 ahora solo confirma que el archivo existe, en vez de re-guardarlo.
>
> **Código listo, ejecución pendiente:** todo el código de esta sesión (Fase 2, migración a `src/evaluation/`, medición de tiempo de inferencia, comparativa en ambos notebooks) pasó chequeo de sintaxis y los 16 tests de la suite siguen en verde, pero **nada de esto se ejecutó contra el dataset real** — no hay TensorFlow ni `data/processed/fashion_products/` disponibles fuera de Colab. A César y Anthony solo les queda correr sus respectivos notebooks de punta a punta en Colab; no deberían necesitar escribir código nuevo.

---

## Ghinno — Módulo 2: BETO fine-tuning

> **Actualizado al 04/07/2026** — modelo `beto_finetuned/` recibido e integrado. Ver detalle completo en `docs/avance_ghinno_m2_beto.md`.

### Completado ✅

| Tarea | Detalle |
|-------|---------|
| Análisis comparativo RF vs BETO documentado | `docs/m2_comparativa_beto_vs_rf.md` — métricas por clase, análisis de errores, recomendación de producción |
| Fine-tuning ejecutado en Colab y modelo copiado a `models/sentiment_analyzer/beto_finetuned/` | Reentrenado con class weights + 5 épocas, F1-macro **0.7475** (mejor checkpoint, época 3, gracias a `load_best_model_at_end=True`) |
| `predict(model_name="beto")` operativo end-to-end | Verificado localmente con 3 textos (positivo/negativo/neutro) — confianza >0.83 en los tres casos |

### Pendiente ❌

| Tarea | Semana | Detalle |
|-------|--------|---------|
| Alcanzar meta F1-macro ≥ 0.80 | S3 | F1-macro reportado: **0.7475** — no supera la meta. La clase Neutro sigue arrastrando el promedio. Diagnóstico de "límite estructural ~0.747" respaldado por tabla de métricas por época real |
| ~~Comparativa RF vs BETO en el notebook~~ | ~~S4~~ | **Resuelto 04/07/2026 — por Jorge, no por Ghinno.** Sección agregada en ambos notebooks con la tabla real |
| ~~Re-ejecutar celda de análisis de errores, filtrada por clase Neutro~~ | ~~Antes del 07/07~~ | **Resuelto** — Ghinno la re-ejecutó en Colab, ya filtrada por `real == 'neutro'` (382 errores encontrados) |
| ~~Commit del modelo integrado y de los fixes en `src/sentiment_analyzer/`~~ | ~~Antes del 07/07~~ | **Resuelto** — `beto_model.py`, `predict.py`, `requirements.txt` y el notebook ya están commiteados y mergeados a `main` |

> **Dos bugs de integración encontrados y corregidos por Jorge (no por Ghinno) al copiar el modelo:** (1) `requirements.txt` pedía `transformers>=4.35`, que hoy resuelve a la v5 — versión que **eliminó el soporte de TensorFlow** por completo, rompiendo el import de `TFAutoModelForSequenceClassification` en `beto_model.py`. (2) La notebook de Ghinno entrena con `AutoModelForSequenceClassification` + `Trainer` de **PyTorch**, no TensorFlow, así que aunque el import funcionara, los pesos no coincidían con el framework esperado. Se migró `beto_model.py` y `predict.py` a PyTorch y se agregó `torch>=2.2` a `requirements.txt`. Esto no es contribución de Ghinno — sigue en cero su commit history sobre `src/sentiment_analyzer/`.
>
> **Nota sobre métricas:** según `docs/m2_comparativa_beto_vs_rf.md`, ningún modelo alcanza su meta (RF: 0.6488 vs ≥0.70; BETO: 0.7475 vs ≥0.80). Ambos tienen dificultad estructural con la clase Neutro. Considerar documentar el análisis de por qué la meta no se alcanza y qué se intentó — es un entregable válido para la exposición.

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
Anthony  █████████░  90%   🆙 Código listo (evaluación + tiempo de inferencia) — falta correrlo en Colab
Jorge    ████████░░  85%   🆙 Comparativa RF vs BETO en ambos notebooks — sin pendientes propios
César    ███████░░░  70%   🆙 Código completo (Fase 2, src/evaluation/, comparativa, tiempo de inferencia) — falta correrlo en Colab
Ghinno   ████████░░  80%   🆙 Modelo integrado y análisis de errores re-ejecutado — cero contribución propia a src/ sigue abierto
Jeremy   ██████░░░░  60%   ⚠️ Deploy y pruebas e2e pendientes (S5)
Víctor   █████░░░░░  55%   ⚠️ Session state real — KPIs incorrectos y matrices con datos falsos (bugs críticos)
```

> Semáforo actualizado al 04/07/2026. César y Anthony actualizados con las correcciones y el código preparado el 02/07 — ninguno de los dos requiere escribir código nuevo, solo ejecutar en Colab. Ghinno actualizado al 04/07 con el modelo BETO ya integrado.

## Acciones críticas — actualizadas al 04/07/2026

1. **César** → correr `01_image_classifier_mobilenetv2.ipynb` completo en Colab (Fase 1 ya la corrió; falta Fase 2 + evaluación con `src/evaluation/` + tiempo de inferencia, todo con código ya listo) y commitear el `mobilenetv2.keras` final
2. **Anthony** → correr la celda de evaluación de `01_image_classifier_efficientnet.ipynb` en Colab (nunca se ejecutó) para tener el reporte real, la matriz de confusión y el tiempo de inferencia
3. **Anthony + César** → una vez ejecutados ambos notebooks, actualizar la tabla de la comparativa (sección 8 en el notebook de César, sección espejo en el de Anthony) con los números reales — hoy tiene los últimos valores conocidos, marcados como pendientes de refresco
4. ~~**Ghinno** → re-ejecutar la celda de análisis de errores filtrada por clase Neutro~~ → **Resuelto 04/07** (382 errores de clase Neutro encontrados)
5. **Víctor** → probar el selector MobileNetV2 en la página M1 ahora que el modelo existe y el bug de `IMAGE_CLASSES` está corregido

El bloqueo de integración de Ghinno quedó resuelto — el modelo BETO ya corre en `predict()`, y la comparativa RF vs BETO ya está en ambos notebooks (`02` y `03`). M2 está funcionalmente cerrado salvo la meta de F1-macro ≥ 0.80 (no alcanzada, documentada como límite estructural) y la falta de contribución de Ghinno al código fuente.
