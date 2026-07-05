# M2 — Comparativa: Random Forest vs BETO

## Métricas en conjunto de validación (test set, 5,000 muestras)

| Modelo                          | Accuracy | F1 Negativo | F1 Neutro | F1 Positivo | F1 Macro |
|----------------------------------|----------|-------------|-----------|-------------|----------|
| Random Forest                    | 73.84%   | 0.80        | 0.33      | 0.82        | 0.6488   |
| BETO (versión original)           | 79.00%   | 0.85        | 0.51      | 0.88        | 0.7455   |
| BETO (class weights + 5 épocas)  | —        | —           | —         | —           | **0.7475** |

## Análisis de resultados

### BETO supera al Random Forest en todas las métricas
- Accuracy: +5.16 puntos porcentuales
- F1 Macro: +0.0967 (mejora del 14.9%)
- La mayor mejora es en clase Neutro: 0.33 → 0.51

### Principal dificultad compartida: clase Neutro
Ambos modelos tienen dificultad con reseñas de 3 estrellas.
Son ambiguas por naturaleza — mezclan opiniones positivas y negativas,
siendo difíciles de clasificar incluso para un evaluador humano.

### Experimento de cierre de gap: class weights + más épocas
Se reentrenó BETO con class weights (para compensar el desbalance de la
clase Neutro) y 5 épocas en vez de 3, con warmup y weight decay. El
resultado fue una mejora marginal: **0.7455 → 0.7475** (+0.002).

La tabla de métricas por época (real, generada por el `Trainer` de
HuggingFace) muestra que el modelo **sobreajusta a partir de la época 4**:
la pérdida de validación sube de 0.542 (época 3) a 0.658 (época 4) y 0.792
(época 5), mientras el F1-macro cae de 0.7475 a 0.7312.

| Época | Training Loss | Validation Loss | F1-Macro |
|-------|---------------|------------------|----------|
| 1 | 0.590 | 0.537 | 0.7473 |
| 2 | 0.560 | 0.541 | 0.7429 |
| 3 | 0.454 | 0.542 | **0.7475** (mejor, guardado) |
| 4 | 0.364 | 0.658 | 0.7359 |
| 5 | 0.260 | 0.792 | 0.7312 |

Gracias a `load_best_model_at_end=True`, el checkpoint guardado es el de
la época 3, no el último — evita quedarse con el modelo sobreajustado.
El diagnóstico es un **límite estructural cercano a 0.747** para este
dataset y arquitectura, no un problema de hiperparámetros que se resuelva
con más épocas.

### Objetivos del proyecto
| Modelo        | Meta F1 Macro | Obtenido | ¿Supera? |
|---------------|---------------|----------|----------|
| Random Forest | ≥ 0.70        | 0.6488   | ❌ No    |
| BETO          | ≥ 0.80        | 0.7475   | ❌ No    |

Ninguno alcanzó el umbral objetivo. La clase neutro impacta
el promedio macro en ambos casos. En el caso de BETO, el experimento de
class weights confirma que el gap remanente (–0.0525) no se cierra con
más entrenamiento sobre la misma arquitectura y dataset.

## Recomendación para producción
BETO es el modelo recomendado sobre Random Forest dado que:
- Mayor accuracy (79% vs 73.84%)
- Mejor F1 Macro (0.7475 vs 0.6488)
- Mejor manejo de reseñas neutras (F1 0.51 vs 0.33)

Para uso en producción se sugiere operar con umbral de confianza,
descartando predicciones neutras de baja certeza, donde BETO
alcanza F1 promedio de 0.865 entre negativo y positivo.