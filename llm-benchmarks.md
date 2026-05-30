# LLM Benchmarks Reference

Este documento sirve como referencia rápida para ver la evolución del modelo **CHFT (Complex Holographic Field Theory)** en comparación con el baseline del **LLM (Transformer)** de referencia entrenado sobre el dataset TinyStories.

## Tabla Comparativa de Benchmarks

| Versión del Modelo | Accuracy@1 | Perplexity (PPL) | Diversity Score | Dimensión ($D$) | Vocabulario (Tokens) | Historias Usadas | Tiempo de Entrenamiento | Estado / Descripción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LLM (Transformer) Target** | **35.00%** | **8.00** | **Alto** | - | - | - | Horas / Días | Meta de referencia a alcanzar/superar. |
| **CHFT v3 (Learned Attn + Multi-Hop)** | **32.70%** | **33.58** | **47.6%** | 8,192 | 7,559 | 3,000 | 142.3 min (Colab T4) | Versión actual optimizada con atención posicional y Hopfield Multi-Hop. |
| **CHFT v2 (Orthogonal + Decay)** | **24.62%** | **93.06** | **36.2%** | 4,096 | 5,424 | 1,000 | 245 segundos (Colab T4) | Con binding posicional aleatorio y decaimiento exponencial fijo. |
| **CHFT v2 (Scaled + Flat Pos)** | **8.29%** | **236.87** | **34.0%** | 4,096 | 5,424 | 1,000 | 250 segundos (Colab T4) | Con rotación posicional geométrica simple plana uniforme. |
| **Baseline (Frecuencias)** | **6.55%** | **7,559.00** | **0.0%** | - | 7,559 | 3,000 | - | Estadístico: siempre predice el token más común. |
| **Baseline (Saturado / Untrained)** | **0.61%** | **97.4M** | **7.0%** | 4,096 | 5,424 | 1,000 | - | Sin normalización $\sqrt{D}$ (softmax saturado, bucle de repeticiones). |

---

## Métricas Clave y Significado
1. **Accuracy@1:** Porcentaje de veces que el modelo predice exactamente el siguiente token correcto.
2. **Perplexity (PPL):** Medida de incertidumbre del modelo. A menor perplejidad, más seguro está el modelo de sus predicciones (un valor cercano a $1.0$ sería predicción perfecta; el azar es equivalente al tamaño del vocabulario).
3. **Diversity Score:** Porcentaje de tokens únicos en los textos generados. Evita bucles repetitivos infinitos (meta ideal > 40%).
