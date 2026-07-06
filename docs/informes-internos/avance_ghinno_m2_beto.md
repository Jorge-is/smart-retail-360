# Informe de Avance — Ghinno (Módulo 2: BETO Fine-Tuning)

**Última actualización:** 04/07/2026 — modelo `beto_finetuned/` recibido e integrado
**Evaluación anterior:** 03/07/2026 — revisión completa de la rama `module/m2-ghinno`
**Evaluador:** Jorge Flores (líder del proyecto)

---

## 1. Historial de commits de Ghinno

Ghinno tiene **cuatro commits propios** en la rama `module/m2-ghinno` (dos nuevos desde la última revisión):

| Hash | Fecha | Archivos modificados | Descripción |
|------|-------|----------------------|-------------|
| `174f510` | 16 Jun 2026 | `docs/m2_comparativa_beto_vs_rf.md` (39 líneas) | Doc comparativo RF vs BETO |
| `c82a574` | 27 Jun 2026 | `notebooks/03_sentiment_beto_finetuning.ipynb` (+6222 líneas) | Notebook con outputs y métricas completas |
| `f45b94f` | 28 Jun 2026 | `notebooks/03_sentiment_beto_finetuning.ipynb` (+2352/-553) | Agrega análisis de errores y sección de viabilidad |
| `038af9e` | 29 Jun 2026 | `notebooks/03_sentiment_beto_finetuning.ipynb` (+2376/-1175) | Reentrena con class weights + 5 épocas |

> El merge `9c98c75` (main → `module/m2-ghinno`, 03/07) trae los cambios de M1 de Jorge/César/Anthony — no es contribución de Ghinno.

El código fuente de M2 (`beto_model.py`, `predict.py`, `train.py`) **sigue sin commits de Ghinno** — fue creado y mantenido por **Jorge Flores**. Sin cambios desde la última revisión.

---

## 2. Semana a semana

### Semana 3 (13–19 Jun) — "Fine-tuning de BETO"

| Entregable planificado | Estado | Evidencia |
|---|---|---|
| `notebooks/03_sentiment_beto_finetuning.ipynb` completo con outputs | ❌ No entregado en S3 | El commit de Ghinno en S3 (`174f510`, 16 Jun) fue solo el doc comparativo. El notebook llegó recién en S5. |

### Semana 4 (20–26 Jun) — "Comparativa RF vs BETO"

| Entregable planificado | Estado | Evidencia |
|---|---|---|
| Todo lo planificado para S4 | ❌ Nada llegó en esta ventana | El primer commit del notebook (`c82a574`) es del 27/06 — ya en S5. Cero actividad de Ghinno entre el 16/06 y el 27/06. |

### Semana 5 (27 Jun – 03 Jul) — entrega concentrada, fuera de cronograma

Los tres commits de notebook (27, 28 y 29 de junio) concentran todo el trabajo real de S3+S4 en los primeros tres días de S5:

| Entregable planificado (S3/S4) | Estado | Evidencia |
|---|---|---|
| Notebook con outputs ejecutados | ✅ Entregado | `c82a574` (27/06) — primera versión completa, 8 celdas. **Ojo:** esta versión no reentrena, solo carga un modelo ya fine-tuneado desde Drive (`AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)`) y lo evalúa — el proceso de entrenamiento que produjo ese modelo (F1=0.7455) no queda registrado en ningún notebook commiteado. |
| Reentrenamiento reproducible desde el notebook | ✅ Entregado (recién en `038af9e`) | A diferencia de `c82a574`, esta versión sí entrena desde el checkpoint base (`AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)`) — es la primera vez que el pipeline completo es reproducible de punta a punta desde el notebook. |
| Análisis de errores + ejemplos mal clasificados | ⚠️ **Regresión — se hizo y se perdió** | En `f45b94f` (28/06) la celda corrió con éxito y mostró 10 ejemplos reales mal clasificados (ver sección 4). En `038af9e` (29/06, estado actual del branch) **la misma celda quedó sin ejecutar** (`outputs: []`, `execution_count: None`) — probablemente porque se reentrenó el modelo y no se volvió a correr esa celda antes de comitear. Hoy, en el HEAD de la rama, no hay evidencia de análisis de errores. |
| Sección "Análisis de viabilidad" en el notebook | ✅ Entregado y verificado | Celda con matriz de confusión, gap de métricas, causa raíz (clase Neutro) y estrategias de cierre — con output real, consistente con los números de entrenamiento. |

---

## 3. Objetivo de métricas

| Modelo | Meta F1-macro | Obtenido | Resultado |
|--------|---------------|----------|-----------|
| BETO (versión original, `c82a574`) | ≥ 0.80 | 0.7455 | ❌ No alcanzado (–0.0545) |
| BETO (reentrenado con class weights + 5 épocas, `038af9e`) | ≥ 0.80 | **0.7475** | ❌ No alcanzado (–0.0525) |

**La estrategia de cierre de gap se implementó, pero el resultado es marginal:** +0.002 de F1-macro (0.7455 → 0.7475). Se verificó con la tabla de métricas por época (real, generada por el `Trainer` de HuggingFace) que el modelo **sobreajusta a partir de la época 4**: la pérdida de validación sube de 0.542 (época 3) a 0.658 (época 4) y 0.792 (época 5), mientras el F1-macro cae de 0.7475 a 0.7312. Gracias a `load_best_model_at_end=True`, el checkpoint guardado es el de la época 3 (el mejor), no el último — esto es correcto y está bien configurado.

| Época | Training Loss | Validation Loss | F1-Macro |
|-------|---------------|------------------|----------|
| 1 | 0.590 | 0.537 | 0.7473 |
| 2 | 0.560 | 0.541 | 0.7429 |
| 3 | 0.454 | 0.542 | **0.7475** (mejor, guardado) |
| 4 | 0.364 | 0.658 | 0.7359 |
| 5 | 0.260 | 0.792 | 0.7312 |

El diagnóstico de "límite estructural ~0.747" que Ghinno documenta en la celda de viabilidad está **respaldado por esta tabla real**, no es una afirmación sin evidencia.

**`docs/m2_comparativa_beto_vs_rf.md` quedó desactualizado:** sigue mostrando el resultado viejo (F1=0.7455, sin mencionar el experimento de class weights ni el 0.7475). No se tocó desde el commit del 16/06.

---

## 4. Análisis de errores — detalle de la regresión

En `f45b94f` (28/06) la celda de análisis de errores corrió y produjo 10 ejemplos reales. Un patrón notable: **los 10 ejemplos mostrados son casos donde la clase real es "negativo"** y el modelo predijo "neutro" o "positivo" — no hay ningún ejemplo de "neutro" mal clasificado, pese a que la clase Neutro es la que arrastra el F1-macro según el propio análisis de viabilidad. Ejemplos como:

- *"Fue un regalo para mi marido y a la semana tuvimos que ir a por otra ya que le daba muchos tirones..."* → Real: negativo, Predicho: neutro
- *"Después de unos meses de uso, encantada con el cubo de basura, se me acaba de romper el cierre..."* → Real: negativo, Predicho: positivo

Esto sugiere que el bucle solo recorre los primeros N ejemplos del set de validación sin filtrar por clase, en vez de buscar específicamente errores de la clase Neutro (que era el pedido original: "especialmente de clase Neutro"). El análisis existe, pero no está enfocado en el caso que más importa.

**Estado actual:** esta celda no está ejecutada en el HEAD de la rama (`038af9e`). Falta volver a correrla — el código en sí no tiene errores de sintaxis y ya demostró funcionar, así que es un re-run rápido en Colab, no un rediseño.

---

## 5. Porcentaje de cumplimiento

| Área | 28/06 | 03/07 | Actual (04/07) | Δ |
|------|-------|-------|-----------------|---|
| Fine-tuning ejecutado y reproducible desde el notebook | 90% | 95% | 95% | — |
| Notebook con código documentado y outputs | 70% | 80% | 80% | — |
| Comparativa RF vs BETO | 70% | 70% | 70% | — (doc sigue desactualizado) |
| Objetivo de métrica F1-macro ≥ 0.80 | 0% | 0% | 0% | — (0.7475 sigue sin alcanzar 0.80) |
| Análisis de errores en el notebook | 0% | 40% 🔁 | 40% | — (sigue pendiente de re-ejecutar en el HEAD) |
| Sección de viabilidad en el notebook | 0% | 90% | 90% | — |
| Integración en `src/` (`beto_model.py`, `predict.py`) | 0% | 0% | 100% ⚠️ | +100% — pero hecha por **Jorge**, no por Ghinno (ver 6.1) |

**Cumplimiento global estimado: ~65–70% → ~70–75%**

> El modelo ya corre end-to-end en el dashboard. El salto de esta actualización es funcional (el módulo pasó de "sin modelo integrado" a "predict() operativo"), pero no le suma cumplimiento a Ghinno — la integración y los dos bugs de compatibilidad (transformers v5 sin TF, mismatch PyTorch/TF) los resolvió Jorge, no el commit history de Ghinno.

> El salto principal viene de la sección de viabilidad (completa y verificada) y del reentrenamiento reproducible. El análisis de errores es un caso particular: el trabajo se hizo (hay evidencia en el historial de git) pero no sobrevivió al último commit — es la pieza más rápida de recuperar antes de la exposición.

---

## 6. Riesgos identificados

1. ~~**Notebook no versionado con outputs.**~~ → RESUELTO.
2. ~~**Bug en cell-5 (`trainer` indefinido).**~~ → RESUELTO.
3. ~~**Sección de viabilidad ausente.**~~ → RESUELTO (`f45b94f`/`038af9e`): matriz de confusión, gap de métricas y estrategia de cierre, con datos reales.
4. **Cero contribución al código fuente.** Sigue abierto — ningún commit de Ghinno sobre `src/sentiment_analyzer/`. La integración que finalmente funcionó (04/07) la hizo Jorge, no Ghinno (ver 6.1).
5. **Meta de F1-macro no alcanzada.** Gap se redujo levemente (–0.0545 → –0.0525) pero sigue sin alcanzarse. La estrategia de mitigación en producción (umbral de confianza para clase neutra) es razonable y está documentada.
6. **Análisis de errores no ejecutado en el HEAD actual — nuevo riesgo (regresión).** El trabajo existe en el historial (`f45b94f`) pero no en el estado final de la rama. Si se mergea tal cual, la evidencia para la exposición del 07/07 no está disponible sin antes correr esa celda en Colab.
7. **Análisis de errores no enfocado en la clase Neutro.** Cuando se re-ejecute, conviene filtrar los ejemplos por `real == 'neutro'` en vez de tomar los primeros N errores generales — el pedido original era específicamente sobre la clase problemática.
8. **`docs/m2_comparativa_beto_vs_rf.md` desactualizado.** No refleja el experimento de class weights ni el resultado de 0.7475.

---

## 6.1 Integración del modelo en `src/` (04/07/2026)

Ghinno re-ejecutó la notebook en Colab y generó `beto_finetuned/` en Drive (5 carpetas `checkpoint-*` de los checkpoints por época + los archivos del modelo final en la raíz, producto de `trainer.save_model(MODEL_DIR)` en la última celda). Jorge copió solo los archivos de la raíz — no los `checkpoint-*`, que son intermedios de entrenamiento — a `models/sentiment_analyzer/beto_finetuned/` y probó `predict()`.

**Dos bugs de integración encontrados y corregidos por Jorge (no por Ghinno):**

1. **`transformers` v5 eliminó el soporte de TensorFlow.** `requirements.txt` pedía `transformers>=4.35`, que hoy resuelve a la 5.9.0 — sin `TFAutoModelForSequenceClassification`. Esto rompía el import de `beto_model.py` antes de siquiera tocar los pesos.
2. **Mismatch de framework.** La notebook de Ghinno entrena con `AutoModelForSequenceClassification` + `Trainer` de **PyTorch** (celda 5: `import torch`, `WeightedTrainer(Trainer)`), no TensorFlow. `beto_model.py` esperaba pesos TF.

**Fix aplicado:** `beto_model.py` y `predict.py` migrados a PyTorch (`AutoModelForSequenceClassification`, tokenización `return_tensors="pt"`, softmax de `torch`); se agregó `torch>=2.2` a `requirements.txt`. Verificado localmente con `predict()` sobre 3 textos (positivo, negativo, neutro) — las tres clases predicen correctamente con confianza >0.83 en todos los casos.

**Esto no cambia el punto 4 de riesgos** ("cero contribución de Ghinno al código fuente") — la integración en `src/` la hizo Jorge, no Ghinno.

---

## 7. Recomendación

**El módulo dio un salto real en Semana 5**, concentrando en tres días (27–29 Jun) el trabajo de S3 y S4: notebook reproducible de punta a punta, reentrenamiento con una estrategia de mitigación concreta (class weights) respaldada por métricas de época verificables, y una sección de viabilidad completa con matriz de confusión y recomendación de producción. Esto es, en calidad, el mejor entregable de Ghinno hasta ahora.

El problema es puntual y fácil de resolver: **la celda de análisis de errores dejó de tener output en el último commit**, después de haber funcionado correctamente un día antes. Antes del 07/07 hace falta:

### Logros del período (27–29 Jun)

- ✅ Notebook reproducible de punta a punta (entrena desde el checkpoint base, ya no depende de un modelo pre-entrenado fuera del repo)
- ✅ Estrategia de cierre de gap implementada (class weights + más épocas + warmup + weight decay) y verificada con tabla de métricas por época real
- ✅ Sección de viabilidad completa: matriz de confusión, causa raíz del gap, umbral de confianza propuesto para producción
- ✅ Diagnóstico de overfitting a partir de época 4, respaldado por datos reales (no es una afirmación sin evidencia)

### Acciones urgentes (antes del 07/07)

1. **Re-ejecutar la celda de análisis de errores [crítico]:** el código ya funciona (se demostró en `f45b94f`), solo falta correrla contra el modelo final y comitear el output.
2. **Enfocar el análisis en la clase Neutro:** filtrar específicamente por `real == 'neutro'`, que es la clase que arrastra el F1-macro — los 10 ejemplos actuales son todos de la clase negativo.
3. **Actualizar `docs/m2_comparativa_beto_vs_rf.md`** con el resultado de 0.7475 y una nota sobre el experimento de class weights, para que el doc y el notebook no cuenten historias distintas.
4. **Opcional:** un commit mínimo en `src/sentiment_analyzer/beto_model.py` o `predict.py` para que Ghinno tenga trazabilidad en el código fuente del módulo — sigue siendo el único punto de M2 con cero contribución de Ghinno.
