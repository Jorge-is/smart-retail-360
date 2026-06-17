# M2 — Comparativa: Random Forest vs BETO

## Métricas en conjunto de validación (test set, 5,000 muestras)

| Modelo        | Accuracy | F1 Negativo | F1 Neutro | F1 Positivo | F1 Macro |
|---------------|----------|-------------|-----------|-------------|----------|
| Random Forest | 73.84%   | 0.80        | 0.33      | 0.82        | 0.6488   |
| BETO          | 79.00%   | 0.85        | 0.51      | 0.88        | 0.7455   |

## Análisis de resultados

### BETO supera al Random Forest en todas las métricas
- Accuracy: +5.16 puntos porcentuales
- F1 Macro: +0.0967 (mejora del 14.9%)
- La mayor mejora es en clase Neutro: 0.33 → 0.51

### Principal dificultad compartida: clase Neutro
Ambos modelos tienen dificultad con reseñas de 3 estrellas.
Son ambiguas por naturaleza — mezclan opiniones positivas y negativas,
siendo difíciles de clasificar incluso para un evaluador humano.

### Objetivos del proyecto
| Modelo        | Meta F1 Macro | Obtenido | ¿Supera? |
|---------------|---------------|----------|----------|
| Random Forest | ≥ 0.70        | 0.6488   | ❌ No    |
| BETO          | ≥ 0.80        | 0.7455   | ❌ No    |

Ninguno alcanzó el umbral objetivo. La clase neutro impacta
el promedio macro en ambos casos.

## Recomendación para producción
BETO es el modelo recomendado sobre Random Forest dado que:
- Mayor accuracy (79% vs 73.84%)
- Mejor F1 Macro (0.7455 vs 0.6488)
- Mejor manejo de reseñas neutras (F1 0.51 vs 0.33)

Para uso en producción se sugiere operar con umbral de confianza,
descartando predicciones neutras de baja certeza, donde BETO
alcanza F1 promedio de 0.865 entre negativo y positivo.