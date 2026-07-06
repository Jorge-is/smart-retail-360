# Informe de Avance — Víctor (Dashboard: páginas y lógica de negocio)

**Fecha de evaluación:** 28/06/2026 — inicio Semana 5  
**Evaluador:** Jorge Flores (líder del proyecto)  
**Rama analizada:** `cambios-victor`

---

## 1. Historial de commits de Víctor

Víctor tiene **un commit propio** en la rama `cambios-victor`:

| Hash | Fecha | Archivos modificados | Descripción |
|------|-------|----------------------|-------------|
| `d219ab4` | 28 Jun 2026 | `app/app.py`, `app/pages/analisis_sentimiento.py`, `app/pages/clasificador_productos.py`, `app/pages/dashboard_integrado.py`, `app/pages/eda.py` | feat: agregar seguimiento de predicciones y métricas en el estado de la sesión |

**Alcance del commit:** 5 archivos, +211 líneas, −67 líneas.

---

## 2. Semana a semana

### Semanas 1–3 (30 May – 19 Jun) — Base de páginas

| Entregable planificado | Estado | Evidencia |
|---|---|---|
| `eda.py` con estructura lista (S1) | ✅ Entregado (previo a esta rama) | Página existe con tabs definidos |
| `clasificador_productos.py` funcional con MobileNetV2 (S2) | ⚠️ Parcial | Página funcional — MobileNetV2 bloqueado por César (sin modelo) |
| `analisis_sentimiento.py` funcional (S3) | ✅ Entregado | Input de texto, inferencia, scores, tabs operativos |

### Semana 4 (20–26 Jun) — "Evaluación rigurosa"

| Entregable planificado | Estado | Evidencia |
|---|---|---|
| `dashboard_integrado.py` funcional con métricas M1 y M2 | ⚠️ Parcial | Página existe pero KPIs venían de `session_state` con donut hardcodeado (`[4118, 739, 423]`) — no conectado a datos reales |

### Semana 5 (27 Jun – 03 Jul) — "Dashboard completo + integración" — EN CURSO

Commit del **28/06** corresponde al arranque de Semana 5. Avances:

| Entregable planificado (S5) | Estado | Evidencia |
|---|---|---|
| `eda.py` con hallazgos reales (gráficos de Anthony y Jorge) | ⚠️ Iniciado | Estructura con 4 secciones explícitas, pero todo son `st.info("🔄 Placeholder...")` — sin datos reales aún |
| Páginas M1 y M2 con selector de modelo completo | ✅ Avanzado | `model_label`, `model_desc` e `st.info()` de feedback al usuario añadidos; session_state completo |
| Session state cross-page (M1 → M2 → Dashboard) | ✅ Implementado | `last_image_prediction`, `last_sentiment_prediction`, `sentiment_counts`, `avg_sentiment` actualizados en tiempo real |

---

## 3. Análisis detallado del commit `d219ab4`

### `app/app.py` — Session state initializers

Tres nuevas claves inicializadas:
- `sentiment_counts` → `{"positive": 60, "neutral": 25, "negative": 15}` — valores iniciales aún hardcodeados pero ya reemplazados en tiempo real por las páginas M2
- `last_image_prediction` → `"Ninguna aún"` — tracking real
- `last_sentiment_prediction` → `"Ninguna aún"` — tracking real

### `app/pages/clasificador_productos.py` — Mejoras

| Cambio | Naturaleza |
|--------|------------|
| `"Imagenes"` → `"Imágenes"` (tilde) | Fix tipográfico |
| `"comparación"` → `"baseline"` | Alineación con terminología del proyecto |
| `model_label` + `model_desc` + `st.info()` | Feedback visual al usuario sobre el modelo activo |
| `session_state["last_image_prediction"]` actualizado en éxito real Y contingencia | Tracking real end-to-end |
| Separación `FileNotFoundError` vs `Exception` genérico | Mejor manejo de errores |
| `+= 1` → `.get("images_classified", 0) + 1` | Fix: evitaba `KeyError` si la clave no existía |

### `app/pages/analisis_sentimiento.py` — Mejoras

| Cambio | Naturaleza |
|--------|------------|
| Model info card actualizada a "BETO / Random Forest" | Refleja selector dual ya existente |
| `model_label` + `model_desc` + `st.info()` | Feedback visual al usuario |
| `sentiment_counts`, `avg_sentiment`, `last_sentiment_prediction` actualizados en tiempo real | Tracking real end-to-end |
| Separación `FileNotFoundError` vs `Exception` genérico | Mejor manejo de errores |
| `reviews_analyzed` con `.get()` | Fix: mismo patrón que M1 |

### `app/pages/dashboard_integrado.py` — Expansión significativa (+111 líneas netas)

| Sección añadida | Estado | Observación |
|----------------|--------|-------------|
| "Estado del flujo integrado" con `st.metric` para últimas predicciones | ✅ Real | Lee de `session_state` — se actualiza cuando el usuario predice |
| Donut de sentimiento conectado a `session_state["sentiment_counts"]` | ✅ Real | Ya no usa `[4118, 739, 423]` hardcodeado — responde a predicciones reales |
| KPIs de M1 y M2 ("Métricas de evaluación integradas") | ❌ Datos incorrectos | Ver sección de bugs |
| Matrices de confusión | ❌ Datos de ejemplo con errores | Ver sección de bugs |

### `app/pages/eda.py` — Refactorización estructural

| Antes | Después |
|-------|---------|
| Hardcodeaba datos silenciosamente (`"Shirts": 1200`) | Secciones M1 y M2 con `st.info("🔄 Placeholder...")` explícito |
| Importaba `sentiment_pie`, `class_distribution_chart` (datos falsos) | Sin imports innecesarios — estructura limpia |
| `st.header` + sin organización clara | `st.title` + secciones separadas por módulo con `st.divider()` |

**Avance real:** la estructura es más honesta (placeholder explícito vs datos falsos silenciosos), pero los gráficos reales siguen pendientes.

---

## 4. Bugs detectados

### Bug 1 — KPIs del dashboard con valores INCORRECTOS ❌

Los valores hardcodeados en `dashboard_integrado.py` no corresponden a las métricas reales del proyecto:

| KPI en dashboard | Valor en código | Valor real | Fuente |
|-----------------|-----------------|-----------|--------|
| `EfficientNet Acc.` | `"87.5%"` | **99.6%** | `avance_equipo_s4.md` / commit EfficientNet |
| `EfficientNet F1` | `"0.875"` | **~0.996** | Sin divergencia documentada con accuracy |
| `MobileNet Acc.` | `"80.7%"` | ❓ Sin entrenar | César no entregó el modelo |
| `RF F1` | `"0.7124"` | **0.6488** | `docs/m2_comparativa_beto_vs_rf.md` |
| `RF Acc.` | `"72.8%"` | **73.84%** | `docs/m2_comparativa_beto_vs_rf.md` |
| `BETO F1` | `"0.7455"` | **0.7455** | ✅ Correcto |
| `BETO Acc.` | `"75.8%"` | **79.0%** | `docs/m2_comparativa_beto_vs_rf.md` |

Estos KPIs se muestran sin disclaimers dentro del `col2` principal — un evaluador que vea la app creerá que EfficientNet tiene 87.5% cuando en realidad tiene 99.6%.

### Bug 2 — `col2` referenciado dos veces en el mismo scope ❌

```python
col1, col2 = st.columns([1, 1])   # outer cols — línea ~41

with col2:                         # primera entrada — Métricas de evaluación
    ...

# try block redefine col2:
col1, col2 = st.columns(2)        # inner cols para matrices — línea ~134
    with col1: ...
    with col2: ...

# al salir del try, col2 apunta a la inner col2 de las matrices
with col2:                         # ← bug: "Últimas Acciones del Sistema"
    ...                            #   va a parar al inner col2, no al outer
```

Si el `try` falla (sin `src/evaluation/`), `col2` vuelve al outer. Si tiene éxito, "Últimas Acciones del Sistema" renderiza dentro de la columna derecha de las matrices. El layout depende de si el modelo de evaluación está disponible.

### Bug 3 — Matrices de confusión con clases incorrectas y generación errónea ❌

```python
labels_m1 = ["Shirts", "Tshirts", "Outwear", "Jeans", "Tops", "Dresses"]  # ← 6 clases
```

El proyecto usa **3 clases** (Shirts, Tshirts, Outwear) desde la coordinación de `IMAGE_CLASSES`. La matriz 6×6 no corresponde al modelo real.

Además, la generación de `y_true` y `y_pred` para `plot_confusion_matrix` usa el mismo índice de columna para ambos:

```python
y_true = [i for row in m1_cm for i, count in enumerate(row) for _ in range(count)]
y_pred = [j for row in m1_cm for j, count in enumerate(row) for _ in range(count)]
# i y j son idénticos — ambos iteran el índice de columna, no el de fila
```

Resultado: `y_true == y_pred` → la matriz siempre aparece como diagonal perfecta (clasificador sin errores), independientemente de los valores en `m1_cm`.

---

## 5. Porcentaje de cumplimiento

| Área | Anterior (24/06) | Actual (28/06) | Δ |
|------|-----------------|----------------|---|
| Página M1 funcional (selector, inferencia, tabs) | 75% | 85% | +10% |
| Página M2 funcional (selector, inferencia, scores) | 80% | 90% | +10% |
| Session state cross-page (M1 → M2 → Dashboard) | 0% | 95% | +95% |
| `dashboard_integrado.py` con métricas reales | 10% | 30% | +20% |
| `eda.py` con hallazgos reales (gráficos de EDA) | 5% | 15% | +10% |
| Integración `src/evaluation/` en Dashboard | 0% | 20% | +20% |

**Cumplimiento global estimado: ~40%** → **~55%**

> El salto principal es el session state wiring — es trabajo real y bien ejecutado. El límite del avance es la cantidad de datos hardcodeados incorrectos que siguen presentes en el dashboard.

---

## 6. Riesgos identificados

1. **KPIs con valores incorrectos en producción [CRÍTICO].** Si la app se despliega con estos valores, la demo del 07/07 mostrará EfficientNet al 87.5% cuando realmente es 99.6%. Debe corregirse antes del deploy.
2. **Bug de `col2` duplicado.** El layout del dashboard cambia dependiendo de si `src/evaluation/` está disponible o no. Inconsistente y difícil de debuggear en demo en vivo.
3. **Matrices de confusión con 6 clases y lógica incorrecta.** Las matrices de ejemplo no representan el modelo real — si se muestran en la demo parecen un clasificador perfecto.
4. **`eda.py` sin datos reales.** La Semana 5 vence el 03/07 — quedan 5 días. Los gráficos de Anthony y Jorge están disponibles en los notebooks; falta conectarlos.
5. **MobileNetV2 bloqueado por César.** El selector existe pero sigue lanzando `FileNotFoundError` — sin cambio respecto a S4.

---

## 7. Recomendación

**El avance de esta semana es sólido en lo técnico (session state), pero la deuda de datos falsos/incorrectos es el riesgo más visible para la presentación del 07/07.**

### Logros del período (hasta 28/06)

- ✅ Session state cross-page completamente implementado — predicciones de M1 y M2 alimentan el Dashboard en tiempo real
- ✅ Error handling mejorado en M1 y M2 (separación `FileNotFoundError` / `Exception`)
- ✅ EDA refactorizado a estructura explícita con placeholders honestos
- ✅ Feedback visual al usuario sobre modelo activo en M1 y M2

### Acciones urgentes (antes del 07/07)

1. **Corregir KPIs del dashboard [CRÍTICO — antes del deploy]:** reemplazar los valores hardcodeados con las métricas reales. EfficientNet: 99.6% accuracy. BETO: F1=0.7455, Acc=79%. RF: F1=0.6488, Acc=73.84%. MobileNet: marcar como "pendiente" hasta que César entregue el modelo.
2. **Corregir bug de `col2` duplicado:** renombrar las columnas internas del bloque de matrices (`cm_col1, cm_col2`) para evitar la colisión de scope.
3. **Corregir matrices de confusión:** usar 3 clases reales (Shirts, Tshirts, Outwear). Corregir la generación de `y_true`/`y_pred` usando el índice de fila para `y_true` y el de columna para `y_pred`.
4. **Conectar EDA con datos reales [S5 — vence 03/07]:** los notebooks de Anthony y Jorge tienen los hallazgos; usar `class_distribution_chart` con los datos reales del EDA (44.446 muestras, distribución real de clases) y `sentiment_pie` con la distribución real de ratings.
5. **Opcional:** agregar `st.warning("Modelo no entrenado")` en la página M1 cuando MobileNetV2 no está disponible, en lugar de fallar silenciosamente en contingencia.
