# LLM Benchmarks Reference

Este documento sirve como referencia rápida para ver la evolución del modelo **CHFT (Complex Holographic Field Theory)** en comparación con el baseline del **LLM (Transformer)** de referencia entrenado sobre el dataset TinyStories.

## Tabla Comparativa de Benchmarks

| Versión del Modelo | Accuracy@1 | Perplexity (PPL) | Diversity Score | Contexto (Tokens) | Parámetros | Historias | Estado / Descripción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Official TinyStories 8-Layer** | **~80.0%** | **~2.40** | **Alto** | 512 | 28M | Completo | El "Sweet Spot" de los Transformers tradicionales en TinyStories. |
| **Official TinyStories 2-Layer** | **60.0% - 65.0%** | **3.00 - 4.50** | **Medio** | 512 | ~33M | Completo | Transformer de 2 capas. Tiene problemas notables de coherencia a largo plazo. |
| **Official TinyStories 1-Layer** | **18.0% - 25.0%** | **9.00 - 11.00** | **Bajo** | 512 | ~21M | Completo | Transformer de 1 capa. Genera gramática básica pero carece de estructura lógica. |
| **Local Transformer 2-Layer** | *TBD* | *TBD* | *TBD* | 64 | ~12.5M | 5,000 | Baseline local de 2 capas con dim=512. Por entrenar. |
| **Local Transformer 1-Layer** | **42.36%** | **15.35** | **33.1%** | 64 | 3.1M | 5,000 | Baseline local de 1 capa con dim=256. Bucles repetitivos. |
| **Local Transformer 1-L (Fast Run)** | **37.10%** | **23.27** | **28.9%** | 64 | 11.3M | 1,000 | Baseline de 1 capa rápido para iteración (dim=768). |
| **CHFT v2 (Fast Run)** | **28.49%** | **51.95** | **80.8%** | 64 | 4.2M | 1,000 | CHFT v2 restaurado para iteración rápida (dim=768). |
| **CHFT v2 (Nano - This Run)** | **33.16%** | **30.04** | **81.9%** | 64 | 6.8M | 5,000 | Run actual con contexto 64. Supera al baseline de frecuencias pero inferior en Acc a Transf 1L. |
| **CHFT v2 (Nano - Prev Run)** | **30.27%** | **37.95** | **77.2%** | 8 | 5.8M | 3,000 | Run anterior con contexto 8 y parada temprana en la época 8. |
| **CHFT v3 (Learned Attn + Multi-Hop)** | **32.70%** | **33.58** | **47.6%** | 8 | ~62M | 3,000 | Versión anterior optimizada con Hopfield Multi-Hop. |
| **CHFT v2 (Orthogonal + Decay)** | **24.62%** | **93.06** | **36.2%** | 8 | ~15M | 1,000 | Con binding posicional aleatorio y decaimiento exponencial fijo. |
| **Baseline (Frecuencias)** | **5.94%** | **8,875.00** | **0.0%** | - | - | 5,000 | Estadístico básico: siempre predice el token más frecuente. |
| **Future Reference (Wiki/Wikitext)**| *TBD* | *TBD* | *TBD* | 1024 | 124M | Wikipedia | GPT-2 (124M) reservado como benchmark para futuras fases con datasets genéricos. |

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
