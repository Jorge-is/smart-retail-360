# Outline de Presentación — SmartRetail 360

**Duración sugerida:** 15–20 minutos + preguntas

---

## 1. Introducción (2 min)

- Contexto: "Somos consultores de inteligencia artificial para una tienda online ficticia."
- Problema: toma de 3 decisiones críticas sin herramientas adecuadas.
- Solución: dashboard integrado con 3 módulos de IA.

## 2. Demo — Módulo 1: Clasificador de productos (4 min)

- Mostrar imagen → predicción en tiempo real.
- Explicar transfer learning: "Usamos EfficientNet-B0 ya entrenado en ImageNet y reentrenamos solo la cabeza."
- Mostrar métricas: accuracy por clase, confusion matrix.

## 3. Demo — Módulo 2: Análisis de sentimiento (4 min)

- Ingresar reseña → clasificación positivo/neutro/negativo.
- Comparar TF-IDF vs BETO: "El baseline clásico logra X%, BETO logra Y% — una mejora de Z%."
- Mostrar distribución de sentimientos en CSV de ejemplo.

## 4. Demo — Módulo 3: Predicción de ventas (4 min)

- Seleccionar tienda + horizonte → gráfico interactivo con intervalo de confianza.
- **El elemento diferenciador:** ajustar el slider de sentimiento y mostrar cómo cambia el pronóstico.
- Mostrar métricas: MAE, RMSE, MAPE.

## 5. Dashboard integrado (2 min)

- Narrativa unificada: "Clasificamos N imágenes, analizamos M reseñas (P% positivas), proyectamos las ventas para los próximos 30 días."
- KPIs de la sesión.

## 6. Arquitectura y decisiones técnicas (2 min)

- Diagrama de la arquitectura.
- Por qué EfficientNet, BETO, Prophet.
- Métricas honestas y lo que mejoraríamos con más tiempo.

## 7. Cierre y preguntas (2 min)

- Próximos pasos si fuera un producto real.
- Preguntas del jurado.

---

## Tips para la demo en vivo

- Tener video de respaldo grabado en semana 7.
- Cargar los modelos antes de empezar (lazy loading tarda la primera vez).
- Usar imágenes de prueba conocidas con buena predicción.
- No improvisar: tener reseñas de prueba escritas.
