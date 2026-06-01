# LLM Benchmarks Reference

Este documento sirve como referencia rápida para ver la evolución del modelo **CHFT (Complex Holographic Field Theory)** en comparación con el baseline del **LLM (Transformer)** de referencia entrenado sobre el dataset TinyStories.

## Tabla Comparativa de Benchmarks

| Versión del Modelo | Accuracy@1 | Perplexity (PPL) | Diversity Score | Contexto (Tokens) | Parámetros | Historias | Estado / Descripción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Official TinyStories GPT-2 (124M)** | **68.0% - 73.5%** | **1.10 - 1.15** | **8% - 12% (TTR)** | 1024 | 124M | Completo | Gold Standard en escala grande de TinyStories. |
| **Official TinyStories 8-Layer** | **~80.0%** | **~2.40** | **Alto** | 512 | 28M | Completo | El "Sweet Spot" de los Transformers tradicionales. |
| **Official TinyStories 2-Layer** | **60.0% - 65.0%** | **3.00 - 4.50** | **Medio** | 512 | ~15M | Completo | Primer matchup directo. Los Transformers de 2-capas tienen problemas de coherencia. |
| **CHFT v2 (Original - Run de Época 10)** | **87.76%** | **1.87** | **72.8%** | 8 | 124M | 3,000 | Con fix de LayerNorm. *Nota: Accuracy inflada por data leakage de semilla.* |
| **CHFT v3 (Learned Attn + Multi-Hop)** | **32.70%** | **33.58** | **47.6%** | 8 | ~62M | 3,000 | Versión anterior optimizada con Hopfield Multi-Hop. |
| **CHFT v2 (Orthogonal + Decay)** | **24.62%** | **93.06** | **36.2%** | 8 | ~15M | 1,000 | Con binding posicional aleatorio y decaimiento exponencial fijo. |
| **Baseline (Frecuencias)** | **6.55%** | **7,559.00** | **0.0%** | - | - | 3,000 | Estadístico básico: siempre predice el token más frecuente. |

---

## Métricas Clave y Significado
1. **Accuracy@1:** Porcentaje de veces que el modelo predice exactamente el siguiente token correcto.
2. **Perplexity (PPL):** Medida de incertidumbre del modelo. A menor perplejidad, más seguro está el modelo de sus predicciones (un valor cercano a $1.0$ sería predicción perfecta; el azar es equivalente al tamaño del vocabulario).
3. **Diversity Score:** Porcentaje de tokens únicos en los textos generados. Evita bucles repetitivos infinitos (meta ideal > 40%).

---

## 📉 Plan de Reducción Progresiva (Paradigmas de Escala)

Para encontrar el límite matemático del paradigma CHFT en comparación con Transformers de atención tradicional (que fallan rápidamente al bajar de escala), se definen las siguientes metas de reducción de tamaño de parámetros:

1. **Midi Scale (~50M - 60M parámetros)**:
   * **Configuración sugerida**: Dimensión $D = 8,192$ (Vocabulario de 7.5k).
   * **Meta**: PPL < 4.5, Accuracy@1 > 75% - 80%.
   * **Propósito**: Validar si el modelo base de 124M estaba sobredimensionado.
2. **Tiny Scale (~10M - 15M parámetros)**:
   * **Configuración sugerida**: Dimensión $D = 1,024 - 2,048$.
   * **Meta**: PPL < 6.0 - 7.0 con alta coherencia de contexto.
   * **Propósito**: Demostrar que el paradigma CHFT comprime conocimiento de forma mucho más eficiente que un Transformer tradicional a baja escala.
3. **Nano Scale (~1M - 5M parámetros)**:
   * **Configuración sugerida**: Dimensión $D = 512 - 768$.
   * **Meta**: Identificar el punto de quiebre (donde PPL > 20 o se producen bucles infinitos).
   * **Propósito**: Encontrar el límite matemático de almacenamiento físico de la teoría de fases holográficas.
