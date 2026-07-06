# Informe de Avance — Anthony (Módulo 1: EfficientNet-B0)

**Rama:** `module/m1-anthony`  
**Fecha de evaluación inicial:** 22/06/2026 — inicio Semana 4  
**Última actualización:** 24/06/2026  
**Evaluador:** Jorge Flores (líder del proyecto)

---

## 1. Historial de commits de Anthony

| Hash | Fecha | Mensaje |
|------|-------|---------|
| `137a2b5` | 23 Jun 2026 | `feat(m1): entrenar EfficientNet-B0, accuracy 99.6% en test` |
| `a3a673c` | 22 Jun 2026 | `feat(m1): completar EDA y estructura base del notebook EfficientNet` |
| `900b22e` | 22 Jun 2026 | `fix(config): actualizar IMAGE_CLASSES a 3 clases segun api_contracts.md` |

Archivos tocados en total:

- `notebooks/00_eda_imagenes.ipynb`
- `notebooks/01_image_classifier_efficientnet.ipynb`
- `src/utils/config.py` (1 línea cambiada)
- `models/image_classifier/efficientnet_b0.keras` (49 MB — modelo entrenado)

---

## 2. Análisis técnico por archivo

### `notebooks/00_eda_imagenes.ipynb` — EDA Fashion Products

| Sección | Estado | Detalle |
|---------|--------|---------|
| Carga de dataset (styles.csv) | ✅ Ejecutado | 44.446 filas, advertencia de CSV malformado manejada correctamente |
| Distribución de masterCategory | ✅ Ejecutado | 7 clases detectadas, barplot generado |
| Visualización de muestras por categoría | ✅ Ejecutado | Grid de 35 imágenes generado |
| Estadísticas de dimensiones | ✅ Ejecutado | Todas 60×80px, histograma generado |
| Detección de inconsistencias | ✅ Ejecutado | 5 IDs sin imagen, 0 nulos, 0 duplicados identificados |
| Decisión de categoría | ⚠️ Parcial | El código corre y muestra la decisión pero el comentario dice "ajustar según resultados" — no es conclusivo |
| Sección de conclusiones | ⚠️ Mal formateado | El contenido está dentro de un bloque de código Markdown (` ``` `) en lugar de prosa — parece copy-paste de draft |
| Cell de imports (cell-2) | ❌ Sin output | Las importaciones nunca se ejecutaron en Colab |
| TODOs residuales | ⚠️ 7 TODOs | Comentarios `# TODO:` en celdas que sí tienen código ejecutado debajo — inconsistente |

**Veredicto EDA:** el trabajo analítico está hecho y los hallazgos son correctos. La presentación tiene problemas de prolijidad pendientes.

---

### `notebooks/01_image_classifier_efficientnet.ipynb` — Entrenamiento EfficientNet-B0

> **Actualizado al commit `137a2b5` (23/06/2026).** El entrenamiento completo fue ejecutado y los outputs están persistidos en el notebook.

| Sección | Estado | Detalle |
|---------|--------|---------|
| Setup de Drive y rutas | ✅ Ejecutado | Drive montado, directorios creados |
| pip install | ⚠️ Sin output | Celda sin ejecutar registrada |
| Descarga del dataset | ⚠️ Comentado | Apropiado (requiere kaggle.json), pero no hay output que confirme que el dataset existe |
| EDA rápido (filtro a 3 clases) | ✅ Ejecutado | 41.911 filas tras filtrar Apparel, Footwear, Accessories |
| Preparación de splits train/val/test | ✅ Implementado / ⚠️ Sin output | `train_test_split` + copia a directorios por split implementado en cell-6, pero sin output — los splits ya existían en Drive (cell-5 confirma directorios) |
| Instanciar modelo vía `src.image_classifier.train` | ✅ Ejecutado | Notebook llama a `train()` de `train.py`; integra correctamente con el módulo de source |
| **Fase 1 — entrenar cabeza (freeze_backbone=True, 10 épocas)** | ✅ Ejecutado | Logs de Keras visibles; `val_accuracy` alcanza ~0.996 en Época 10 |
| **Fase 2 — fine-tuning completo (lr=1e-4, 10 épocas)** | ✅ Ejecutado | `val_accuracy` ≈ 0.997 al final; entrenamiento completo |
| **Data augmentation** | ✅ Implementado | Definido en `train.py` → `_build_augmentation()`: `RandomFlip`, `RandomRotation`, `RandomBrightness`. Aplicado al `train_ds` antes de las dos fases |
| **`model.save()` → `efficientnet_b0.keras`** | ✅ Ejecutado | Guardado en `/content/smart-retail-360/models/image_classifier/efficientnet_b0.keras`; archivo commiteado (49 MB) |
| **Evaluación: `classification_report`** | ✅ Ejecutado | Output completo: precision/recall/f1 por clase + accuracy global = 1.00 sobre test set |
| **Matriz de confusión con seaborn** | ✅ Ejecutado / ⚠️ Sin imagen embebida | `sns.heatmap` ejecutado, output muestra `<Figure size 640x480 with 2 Axes>` pero la imagen no está guardada como base64 en el notebook — no visible offline |
| **Uso de `src/evaluation/`** | ❌ Ausente | Evaluación usa sklearn directamente; el módulo compartido `src/evaluation/` no está integrado |
| TODOs residuales en cell-7 | ⚠️ Cosmético | Los 5 comentarios `# TODO:` de la versión anterior quedaron como cabecera de la celda aunque el código debajo ejecutó correctamente — deuda de limpieza |

**Veredicto entrenamiento:** el modelo existe, fue entrenado con la pipeline correcta (transfer learning en 2 fases + data augmentation) y supera ampliamente la meta de accuracy. Queda deuda de presentación y de integración con `src/evaluation/`.

---

### `src/utils/config.py`

```diff
- IMAGE_CLASSES = ["Apparel", "Accessories", "Footwear", "Personal Care", "Sporting Goods"]
+ IMAGE_CLASSES = ["Apparel", "Accessories", "Footwear"]
```

La decisión de reducir a 3 clases es correcta y respaldada por el EDA (las otras clases tienen 1–105 muestras, inviables para entrenar). Sin embargo, este cambio tiene un **impacto lateral**:

- `src/image_classifier/model_efficientnet.py` usa `len(IMAGE_CLASSES)` como default para `num_classes`. Si César entrenó MobileNetV2 con las 5 clases previas, habrá un desajuste de arquitectura.
- El cambio afecta un config compartido — debe coordinarse con César antes del merge.

---

## 3. Estado por semana del cronograma

| Semana | Tarea de Anthony | Estado | Observación |
|--------|-----------------|--------|-------------|
| **S1** (30 May – 05 Jun) | EDA Fashion Products | ✅ Completado | Committed el 22/06 pero el trabajo analítico está hecho y es correcto |
| **S2** (06 – 12 Jun) | Preparar notebook EfficientNet + revisar a César | ✅ Completado | Estructura y pipeline de entrenamiento implementadas |
| **S3** (13 – 19 Jun) | **Entrenamiento EfficientNet-B0 completo** | ✅ Completado con retraso | Entregado el 23/06 (4 días tarde). Accuracy 99.6% supera la meta. Modelo commiteado. |
| **S4** (20 – 26 Jun) | **Comparativa MobileNetV2 vs EfficientNet-B0** | ❌ No iniciado | En curso (vence 26/06). Sin celda de comparativa en el notebook. Bloqueado hasta que César entregue MobileNetV2. |

---

## 4. Objetivos de métricas

| Modelo | Meta | Resultado | Estado |
|--------|------|-----------|--------|
| EfficientNet-B0 | Accuracy ≥ 85% en test | **99.6%** | ✅ Superado ampliamente |

Métricas por clase (test set, 6.286 imágenes):

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Apparel | 0.99 | 0.99 | 0.99 |
| Footwear | 1.00 | 1.00 | 1.00 |
| Accessories | 1.00 | 1.00 | 1.00 |
| **weighted avg** | **1.00** | **1.00** | **1.00** |

---

## 5. Porcentaje de cumplimiento

| Área | 22/06 | 24/06 | Observación |
|------|-------|-------|-------------|
| EDA completo con hallazgos | 80% | 80% | Deuda de prolijidad sin resolver |
| Notebook EfficientNet con entrenamiento | 5% | 90% | Entrenamiento completo, TODOs residuales y cell-6 sin output |
| Data augmentation | 0% | 100% | Implementado en `train.py` con 3 capas de augmentation |
| Métricas y matriz de confusión | 0% | 80% | classification_report ✅; figura de heatmap no embebida en notebook |
| Modelo `efficientnet_b0.keras` generado | 0% | 100% | Commiteado (49 MB) |
| Uso de `src/evaluation/` | 0% | 0% | No integrado; sklearn directo en su lugar |
| Análisis de viabilidad vs MobileNetV2 | 0% | 0% | Pendiente — S4 vence 26/06 |

**Cumplimiento global estimado: ~65–70%** _(subió desde ~25–30%)_

---

## 6. Riesgos identificados

1. ~~**La Semana 3 (entrenamiento) venció sin entregable.**~~ **RESUELTO** — entrenamiento completo commiteado el 23/06 con accuracy 99.6%.
2. **La Semana 4 (comparativa) vence el 26/06 (2 días).** Sin celda de comparativa en el notebook y bloqueado hasta que César tenga su modelo MobileNetV2 evaluado.
3. **Cambio en `IMAGE_CLASSES` puede romper integración con César.** Si César entrenó MobileNetV2 con 5 clases, la capa densa final tiene dimensión incorrecta. Requiere coordinación explícita antes del merge.
4. **`src/evaluation/` no está integrado.** El módulo de métricas compartido existe pero Anthony usa sklearn directamente — la presentación del 07/07 requiere matrices visibles en el dashboard usando la infraestructura común.
5. **Matriz de confusión no está embebida como imagen.** El heatmap se generó pero no se guardó en el notebook como figura base64 — no visible sin re-ejecutar.
6. **EDA notebook tiene deuda de prolijidad.** TODOs residuales y conclusiones en bloque de código no son presentables para entrega académica.

---

## 7. Recomendación sobre el merge

**La rama NO está lista para merge completo.**

El modelo entrenado y el EDA pueden mergearse con correcciones menores. El notebook de entrenamiento necesita limpieza antes de incorporarse a `main`.

### Acciones bloqueantes (no merge sin esto)

1. **Coordinar con César el cambio de `IMAGE_CLASSES` (3 vs 5)** antes de que llegue a `main` — riesgo de desajuste de arquitectura.
2. **Integrar `src/evaluation/`** para la generación de métricas y matriz de confusión — requerido para el dashboard del 07/07.

### Acciones urgentes (vencen 26/06)

3. **Celda de comparativa MobileNetV2 vs EfficientNet-B0** — es el entregable de S4. Requiere coordinar con César para tener las métricas de MobileNetV2 disponibles.

### Acciones de prolijidad (necesarias antes del 07/07)

4. Limpiar los 5 comentarios `# TODO:` residuales de cell-7 (el código ejecutó, los TODOs quedaron como basura).
5. Re-ejecutar cell-6 (preparación de splits) para que tenga output y el notebook sea reproducible end-to-end.
6. Guardar la figura de la matriz de confusión como imagen en el notebook (o exportar a `docs/`).
7. Limpiar TODOs residuales del EDA y reformatear las conclusiones (sacar del bloque de código Markdown).
8. Confirmar que `efficientnet_b0.keras` está disponible en Drive para el equipo de dashboard.
