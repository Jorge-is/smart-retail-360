# Informe de Avance — Equipo completo

**Fecha:** 24/06/2026 — Semana 4 (20–26 Jun) · **Actualización Víctor:** 28/06/2026 — inicio Semana 5 · **Actualización César/M1:** 05/07/2026 · **Actualización Ghinno/M2:** 04/07/2026  
**Evaluador:** Jorge Flores (líder del proyecto)  
**Próximas fechas clave:** Segunda exposición 07/07/2026 · Entrega final 14/07/2026

---

## Resumen ejecutivo

| Integrante | Rol | Cumplimiento S1–S4 | Estado |
|------------|-----|--------------------|--------|
| Anthony | M1 lead — EfficientNet-B0 | ~90% 🆙 | ⚠️ Ya no bloqueado — código de evaluación y tiempo de inferencia listo, falta ejecutarlo en Colab |
| Jorge | M2 lead — RF + contratos + arquitectura | ~85% 🆙 | ✅ Comparativa RF vs BETO agregada en ambos notebooks (04/07) — sin pendientes propios |
| César | M1 — MobileNetV2 | ~65% ⬇️ | 🔴 **Fase 2 NO se ejecutó de verdad** — el notebook y el commit dicen "evaluación completa Fase 1+2" pero las celdas de entrenamiento nunca corrieron (`execution_count: None`); se evaluó el mismo modelo de Fase 1 de siempre. Único dato nuevo real: tiempo de inferencia 4.17 ms/imagen |
| Ghinno | M2 — BETO fine-tuning | ~80% 🆙 | ⚠️ Modelo integrado y análisis de errores re-ejecutado — cero contribución propia a src/ sigue abierto |
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

> **Actualizado al 05/07/2026** — rama `module/m1-cesar`, commit `22f3fec` + `cf4192c`.

### Completado ✅

| Tarea | Detalle |
|-------|---------|
| Entrenamiento MobileNetV2 (Fase 1 — cabeza, backbone congelado) | `notebooks/01_image_classifier_mobilenetv2.ipynb` ejecutado en Colab, 23 celdas con output |
| Modelo `mobilenetv2.keras` | Commiteado en la rama (9.66 MB) — desbloquea a Anthony y a Víctor |
| Accuracy en test | **93.83%** — supera ampliamente la meta de ≥78% (TEAM.md). Sigue siendo el resultado de Fase 1 (ver hallazgo abajo) |
| Data augmentation | `RandomFlip`, `RandomRotation`, `RandomBrightness` (igual que EfficientNet) |
| Integración en `predict.py` | Ya soportado (`_MODEL_PATHS["mobilenetv2"]`, implementado por Jorge) — ahora con archivo real detrás, deja de lanzar `FileNotFoundError` |
| Bug de mapeo de clases (`IMAGE_CLASSES`) | **Corregido 02/07/2026** — ver nota abajo |
| Comparativa MobileNetV2 vs EfficientNet | **Escrita 02/07/2026** — sección 8 en el notebook de César + sección espejo en el de Anthony (TEAM.md pide la sección en ambos), con métricas reales. **Sigue sin refrescar** con el dato de tiempo de inferencia que ya existe (ver hallazgo abajo) |
| Integrar `src/evaluation/` en el notebook — **código** | **Migrado y ejecutado 05/07/2026** — la celda de evaluación (sección 6) corrió con `compute_metrics`/`plot_confusion_matrix`/`assess_viability`, con matriz de confusión guardada como PNG |
| Medición de tiempo de inferencia | **Dato real obtenido 05/07/2026** — **4.17 ms/imagen** (26.19s / 6286 imágenes), medido con `time.perf_counter()` sobre el modelo cargado en la evaluación |

### Pendiente ❌ / Desviaciones ⚠️

| Tarea | Semana | Detalle |
|-------|--------|---------|
| 🔴 **Ejecutar Fase 2 de verdad (fine-tuning completo)** | S2/S4 (vencidas) | **Hallazgo crítico 05/07/2026**: el notebook y el commit `22f3fec` ("evaluación completa MobileNetV2 con métricas reales") afirman haber evaluado "Fase 1 + Fase 2", pero **las celdas de entrenamiento nunca se ejecutaron** — la celda de `model.fit()` de Fase 1, la de Fase 2 (descongelar + `model.fit()`) y la de guardado final tienen `execution_count: None` y 0 outputs. La celda de evaluación cargó el `mobilenetv2.keras` que ya existía en Drive de la sesión anterior (solo Fase 1). Prueba: el `classification_report` es **idéntico byte a byte** al de la Fase 1 original (Accessories 0.96/0.82/0.88, Apparel 0.93/0.98/0.95, Footwear 0.95/0.98/0.96, accuracy 93.83%) — si Fase 2 hubiera corrido, esos números tendrían que cambiar |
| Actualizar sección 8 (comparativa) con el tiempo de inferencia real | Antes del 07/07 | La tabla sigue diciendo "No medido" pese a que la celda de evaluación ya lo mide (4.17 ms/imagen) |
| Ejecutar evaluación en Colab (EfficientNet) | S3 (vencida) | **Resuelto por Anthony el 04/07** — ver su sección |

> **Nota sobre autoría:** el trabajo de entrenamiento (Fase 1) es de César; los commits `63d5c38` y `22f3fec` fueron ejecutados/subidos por Anthony (firmados con su identity de Git `XCypherXx <anthonycondori25@gmail.com>`, la misma que usa en sus commits de EfficientNet) — probablemente porque corrieron el Colab desde la cuenta de Anthony. El avance se acredita a César; queda como nota de proceso para que a futuro cada quien commitee desde su propia identidad de Git.
>
> **Bug crítico corregido (config.py):** `src/utils/config.py` tenía `IMAGE_CLASSES = ["Apparel", "Footwear", "Accessories"]`, pero `image_dataset_from_directory` (usado en `train.py` y en ambos notebooks) asigna índices por orden **alfabético de carpeta** (`Accessories=0, Apparel=1, Footwear=2`, confirmado en la celda 7 del notebook: `Clases: train_ds.class_names`). Como `predict.py` mapea `IMAGE_CLASSES[i]` al índice `i` de salida del modelo, las etiquetas devueltas estaban **cruzadas**. **Corregido** el 02/07/2026 reordenando `IMAGE_CLASSES = ["Accessories", "Apparel", "Footwear"]` — no requiere reentrenar ningún modelo, ambos ya usaban ese orden internamente. Verificado corriendo `tests/test_image_classifier.py` (16/16 tests OK) contra los modelos `.keras` reales.
>
> **Segundo hallazgo del mismo bug (notebook EfficientNet):** al armar la comparativa se encontró que `notebooks/01_image_classifier_efficientnet.ipynb` (de Anthony) tenía el mismo error pero hardcodeado localmente (`VALID_CLASSES = ["Apparel", "Footwear", "Accessories"]` en la celda de evaluación), independiente de `config.py`. El accuracy global (99.6%) no cambia porque es agregado, pero el desglose por clase estaba cruzado: los soportes (1694/3209/1383) coincidían exactamente entre el reporte de MobileNetV2 (bien etiquetado) y el de EfficientNet (mal etiquetado). **Corregido**: la celda ahora deriva `VALID_CLASSES` de `test_ds.class_names`.
>
> **Restructuración del notebook de MobileNetV2 (02/07/2026):** al descomentar la Fase 2 se detectó un problema de orden — la evaluación (sección 6) leía el modelo desde `mobilenetv2.keras`, pero el guardado "oficial" solo pasaba en la sección 7, *después* de la evaluación. El notebook original resolvía esto con un `shutil.copy` que copiaba el checkpoint de Fase 1 al path final antes de evaluar — es decir, **siempre evaluaba Fase 1, nunca Fase 2**, aunque la Fase 2 se llegara a ejecutar. Se corrigió moviendo el guardado real al final de la Fase 2 (antes de evaluar) y la sección 7 ahora solo confirma que el archivo existe, en vez de re-guardarlo.
>
> **Hallazgo del 05/07/2026 — Fase 2 sigue sin ejecutarse pese a la corrección de orden:** la restructuración del 02/07 arregló el bug de que "siempre se evaluaba Fase 1", pero en esta corrida (`22f3fec`) el problema fue distinto: las celdas de `model.fit()` de Fase 1 y Fase 2 directamente **no se corrieron** (no es un bug de código, es que no se ejecutaron esas celdas en Colab antes de guardar el notebook). Por eso la evaluación terminó leyendo el modelo viejo que ya estaba en Drive. **Acción pendiente:** César tiene que volver a Colab y ejecutar en orden — Fase 1 (`model.fit`), Fase 2 (unfreeze + `model.fit`), guardado final — sin saltarse ninguna celda, antes de volver a correr la evaluación.
>
> **Nota:** un hallazgo equivalente (números de un modelo evaluado con GPU vs. CPU sin declararlo) aparece también en `notebooks/05_viability_comparison.ipynb` de Anthony — ver esa sección para el detalle. Ambos casos son del mismo tipo de riesgo: dar por buena una corrida sin verificar qué celdas realmente ejecutaron.

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
Anthony  █████████░  90%   🆙 Evaluación real ejecutada — falta refrescar tiempo de inferencia de EfficientNet en 05_viability_comparison (corrió sin GPU)
Jorge    ████████░░  85%   ✅ Comparativa RF vs BETO en ambos notebooks — sin pendientes propios
César    ██████░░░░  65%   🔴 Fase 2 NO se ejecutó pese a lo que dice el commit — se re-evaluó el modelo viejo de Fase 1. Falta volver a Colab
Ghinno   ████████░░  80%   🆙 Modelo integrado y análisis de errores re-ejecutado — cero contribución propia a src/ sigue abierto
Jeremy   ██████░░░░  60%   ⚠️ Deploy y pruebas e2e pendientes (S5)
Víctor   █████░░░░░  55%   ⚠️ Session state real — KPIs incorrectos y matrices con datos falsos (bugs críticos)
```

> Semáforo actualizado al 05/07/2026. César baja de 70% a 65% por el hallazgo de Fase 2 no ejecutada — es una regresión de confiabilidad, no de código (el código está bien, faltó correrlo). Jorge sube a 85% con la comparativa RF/BETO cerrada. Ghinno sin cambios desde 04/07.

## Acciones críticas — actualizadas al 05/07/2026

1. 🔴 **César** → volver a Colab y ejecutar **en orden** las celdas de Fase 1 (`model.fit`), Fase 2 (unfreeze + `model.fit`) y guardado final del notebook `01_image_classifier_mobilenetv2.ipynb` — ninguna de las tres corrió en la última subida, por eso la evaluación siguió dando el resultado de Fase 1 (93.83%). Después, volver a correr la evaluación (sección 6) para obtener el reporte real de Fase 1+2
2. **Anthony** → re-correr `05_viability_comparison.ipynb` con GPU activada (Runtime → Change runtime type → T4 GPU) — la corrida actual midió 530 ms/imagen para EfficientNet-B0 en CPU, que no es comparable con los 7.65 ms/imagen medidos con GPU en `01_image_classifier_efficientnet.ipynb`
3. **Anthony + César** → una vez que Fase 2 de MobileNetV2 corra de verdad, actualizar la tabla de comparativa (sección 8 en el notebook de César, sección espejo en el de Anthony) con los números finales, incluyendo tiempo de inferencia en ambos (Anthony ya tiene 7.65 ms/img, César ya tiene 4.17 ms/img de Fase 1 — falta el de Fase 2 real)
4. ~~**Ghinno** → re-ejecutar la celda de análisis de errores filtrada por clase Neutro~~ → **Resuelto 04/07** (382 errores de clase Neutro encontrados)
5. **Víctor** → probar el selector MobileNetV2 en la página M1 ahora que el modelo existe y el bug de `IMAGE_CLASSES` está corregido — ojo que el modelo detrás del selector sigue siendo el de Fase 1 hasta que César resuelva el punto 1

El bloqueo de integración de Ghinno quedó resuelto — el modelo BETO ya corre en `predict()`, y la comparativa RF vs BETO ya está en ambos notebooks (`02` y `03`). M2 está funcionalmente cerrado salvo la meta de F1-macro ≥ 0.80 (no alcanzada, documentada como límite estructural) y la falta de contribución de Ghinno al código fuente. El riesgo más urgente ahora es el de César: `module/m1-cesar` se mergeó a `main` con la Fase 2 todavía sin ejecutar de verdad — queda documentado como deuda abierta, no bloqueante para el resto del equipo pero sí para dar por cerrado M1.
