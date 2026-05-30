# Propuesta CHFT v2 — Análisis Crítico y Hoja de Ruta para Prototipo Real

> **Autor:** Revisión crítica de la propuesta original CHFT  
> **Fecha:** 2026-05-29  
> **Objetivo:** Identificar los puntos débiles de la propuesta v1, reformular el paradigma con precisión técnica, y definir un camino de implementación concreto, ejecutable en una PC convencional.

---

## 1. Diagnóstico Crítico: Lo que la Propuesta v1 Promete vs. Lo que la Física y la Matemática Permiten

La propuesta v1 es intelectualmente ambiciosa y estéticamente coherente. Sin embargo, presenta varias afirmaciones que mezclan metáforas físicas con promesas funcionales sin justificación matemática suficiente. Este análisis es necesario para construir sobre ella algo que realmente funcione.

### 1.1 Afirmación: "Cero alucinaciones lógicas"

**El problema:** Las alucinaciones en LLMs no surgen únicamente de la probabilidad difusa. Surgen porque el espacio de representación carece de una fuente de verdad verificable. El hecho de que un sistema use geodésicas en un grupo de Lie **no garantiza que esas geodésicas estén ancladas a hechos reales**. Si el manifold se construye desde datos de texto (igual que los LLMs), hereda los mismos sesgos y lagunas factuales.

**Veredcito:** Promesa infundada en la v1. Reducible, no eliminable, sin memoria externa verificable.

### 1.2 Afirmación: "Memoria O(1) absoluta mediante holografía"

**El problema:** Los sistemas de memoria holográfica (Holographic Reduced Representations — HRR) son reales y funcionan. Pero tienen una limitación crítica conocida: **la capacidad de recuperación degrada con el número de ítems superpuestos**. Con `n` conceptos codificados en un vector de dimensión `d`, la señal-ruido cae aproximadamente como `√(n/d)`. Para mantener recuperación fiel con vocabularios de tamaño real (50k+ tokens), se necesitan dimensiones extraordinariamente altas, lo cual erosiona la ventaja O(1) al nivel práctico.

**Veredicto:** Matemáticamente real, pero la v1 omite los límites de capacidad. No es O(1) puro en práctica; es **O(log n)** con baja constante en condiciones controladas.

### 1.3 Afirmación: "Variedad de Lie como espacio semántico"

**El problema:** Investigación empírica reciente (2024-2025) muestra que los embeddings de tokens en LLMs reales **no respetan estructuras de manifold suave**. Los datos lingüísticos exhiben singularidades, polisemia masiva y estructura de "fiber bundle" (haz de fibras), no de grupo de Lie homogéneo. Forzar el lenguaje en un grupo de Lie introduce una restricción geométrica que el lenguaje natural no respeta.

**Veredicto:** Hermoso matemáticamente. Cuestionable empíricamente para lenguaje natural. Funciona bien para **dominios estructurados** (lógica, matemáticas, código).

### 1.4 Afirmación: "Hardware fotónico de consumo nulo"

**El problema:** Los chips fotónicos son hardware real y prometedor, pero están en etapa de laboratorio para aplicaciones de IA. La afirmación de "consumo energético nulo" es físicamente incorrecta: incluso los sistemas fotónicos tienen pérdidas ópticas y requieren electrónica de control. Esta afirmación daña la credibilidad del resto de la propuesta.

**Veredicto:** Dirección válida para el futuro. Eliminada como promesa en esta revisión.

### 1.5 Lo que SÍ es sólido en la v1

- ✅ La intuición central de **representación de fase compleja** (FHRR) es matemáticamente correcta y tiene respaldo en la literatura de VSA.
- ✅ La idea de **colapso en atractores** tiene paralelo real en redes de Hopfield modernas (Hopfield Networks 2.0).
- ✅ La **ligadura semántica** mediante multiplicación de números complejos es una operación reversible real y central en HDC/VSA.
- ✅ El enfoque **no-backpropagation** es alcanzable y está siendo investigado activamente.

---

## 2. CHFT v2: El Paradigma Reformulado con Precisión

El paradigma CHFT v2 abandona las hipérboles y se ancla en tres campos científicos consolidados que se fusionan de forma novedosa:

```
CHFT v2 = FHRR (Memoria de Fase Compleja)
         + Hopfield Moderno (Dinámica de Atractores)
         + Geometría Hiperbólica Discreta (Jerarquía Semántica)
```

### 2.1 Pilar 1: FHRR — Fourier Holographic Reduced Representation

En lugar del término "holografía cuántica" de la v1, usamos el nombre técnico correcto: **Fourier Holographic Reduced Representation (FHRR)**. Es un subtipo de Vector Symbolic Architecture (VSA) donde:

- Cada concepto `c` se codifica como un vector complejo de dimensión `d`: `v_c ∈ ℂ^d`, con `|v_c[i]| = 1` (vector en la hipersuperficie unitaria).
- La **ligadura (binding)** de dos conceptos `A` y `B` es: `A ⊛ B = IFFT(FFT(A) ⊙ FFT(B))` — multiplicación en el dominio de la frecuencia.
- Esta operación es **conmutativa, asociativa y reversible** (desligadura via conjugación compleja).
- La **superposición (bundling)** de conceptos es suma elemento a elemento: `A + B + C`, seguido de normalización a la esfera unitaria.

**Diferencia con v1:** No afirmamos "colapso cuántico". Es resonancia de correlación cruzada clásica. Más honesto, igualmente potente.

### 2.2 Pilar 2: Hopfield Moderno — Dinámica de Atractores Real

Las **redes de Hopfield modernas** (Ramsauer et al., 2020 — NeurIPS) son el mecanismo de "atractores" que la v1 intuía pero no definía. Formalmente:

- La energía del sistema es: `E = -lse(β, Xᵀξ) + ½ ξᵀξ + cte`
- La actualización sincrónica converge en **un solo paso** para patrones bien separados.
- La capacidad de almacenamiento escala **exponencialmente** con la dimensión (en contraste con la Hopfield clásica que escala linealmente).
- El mecanismo de recuperación es exactamente el **attention mechanism** de los Transformers — lo que significa que CHFT v2 puede ser interpretado como un Transformer sin pesos aprendidos por backpropagation.

**Conexión con v1:** Los "valles de energía mínima" y los "atractores" de la v1 son precisamente los patrones memorizados en la energía de Hopfield.

### 2.3 Pilar 3: Geometría Hiperbólica (en lugar de Lie)

En lugar de la costosa geometría de grupos de Lie, CHFT v2 adopta **embeddings hiperbólicos** (modelo de Poincaré o modelo del hiperboloide) para representar jerarquías semánticas:

- El espacio hiperbólico tiene **capacidad exponencial** para representar jerarquías en baja dimensión.
- La distancia hiperbólica respeta relaciones "es-un" (taxonomías) naturalmente.
- Se implementa con precisión en Python via `geoopt` (librería de optimización en variedades Riemannianas sobre PyTorch).

**Por qué es mejor que Lie para lenguaje:** Las jerarquías lingüísticas (hiperónimos, holónimos) son árboles con ramificación exponencial. El espacio hiperbólico es la geometría natural de los árboles. Los grupos de Lie están diseñados para simetrías continuas (rotaciones, traslaciones), no para jerarquías.

---

## 3. Arquitectura del Prototipo CHFT v2

```
┌─────────────────────────────────────────────────────┐
│                  ENTRADA (Texto)                    │
└──────────────────────┬──────────────────────────────┘
                       │ Tokenización
                       ▼
┌─────────────────────────────────────────────────────┐
│        CAPA 1: Codificador FHRR                     │
│  token → vector de fase compleja ∈ ℂ^d             │
│  Binding de tokens adyacentes (n-gramas)            │
│  Superposición de contexto (ventana deslizante)     │
└──────────────────────┬──────────────────────────────┘
                       │ Vector de estado Ψ ∈ ℂ^d
                       ▼
┌─────────────────────────────────────────────────────┐
│        CAPA 2: Manifold Hiperbólico                 │
│  Proyección al modelo de Poincaré                   │
│  Búsqueda de vecinos semánticos en ℍ^n             │
│  Representa jerarquía y especificidad               │
└──────────────────────┬──────────────────────────────┘
                       │ Coordenadas hiperbólicas
                       ▼
┌─────────────────────────────────────────────────────┐
│        CAPA 3: Memoria de Hopfield Moderna          │
│  Patrones almacenados = conceptos del vocabulario   │
│  Recuperación en 1 paso (sin iteración)             │
│  Salida = token más coherente (colapso en atractor) │
└──────────────────────┬──────────────────────────────┘
                       │ Token generado
                       ▼
┌─────────────────────────────────────────────────────┐
│            ACTUALIZACIÓN DE ESTADO                  │
│  Nuevo token altera Ψ via binding FHRR              │
│  El ciclo se repite (generación autoregresiva)      │
└─────────────────────────────────────────────────────┘
```

**Lo que NO existe en este modelo:**
- ❌ Backpropagation durante inferencia
- ❌ Atención cuadrática (la memoria de Hopfield opera en O(n·d))
- ❌ Matrices de pesos de miles de millones de parámetros

**Lo que SÍ aprende:**
- Los vectores de fase FHRR se pueden **refinar** mediante un proceso de descenso de gradiente en el espacio de fases (no en pesos sinápticos), lo cual es el proceso de "entrenamiento" CHFT.

---

## 4. Stack de Librerías Python Recomendado

### Núcleo del Paradigma

| Librería | Versión | Propósito | Instalación |
|---|---|---|---|
| **torchhd** | ≥ 0.3 | Motor FHRR/VSA con backend PyTorch, soporte GPU | `uv pip install torchhd` |
| **geoopt** | ≥ 0.5 | Optimización en variedades Riemannianas (Poincaré, Lorentz) | `uv pip install geoopt` |
| **torch** | ≥ 2.2 | Backend tensorial, memoria Hopfield moderna | `uv pip install torch` |

### Soporte Científico

| Librería | Propósito |
|---|---|
| **numpy** | Operaciones FFT, validación numérica |
| **scipy** | Análisis espectral, métricas de coherencia |
| **matplotlib / plotly** | Visualización del espacio de fases y atractores |
| **datasets (HuggingFace)** | Descarga de datasets estándar |
| **tiktoken / tokenizers** | Tokenización eficiente |

### Instalación completa del entorno

```bash
uv init chft-prototype
cd chft-prototype
uv add torch torchhd geoopt numpy scipy matplotlib datasets tiktoken plotly
```

> **Nota:** `torchhd` es la librería de referencia para VSA/HDC en Python (mantenida activamente, con soporte para FHRR que es exactamente el modelo de fases complejas de CHFT). Documentación: https://torchhd.readthedocs.io

---

## 5. Dataset Recomendado para Prueba Local

### Opción A — Para prueba de concepto rápida (< 30 minutos de preparación)

**Penn Treebank (PTB) — Porción de test**
- Tamaño: ~5 MB (texto)
- Vocabulario: ~10,000 tokens únicos
- Descarga: via HuggingFace Datasets
- Tarea: Modelado de lenguaje a nivel de palabra, clasificación de POS
- Ideal para: Validar que FHRR puede recuperar tokens y secuencias correctamente

```python
from datasets import load_dataset
ds = load_dataset("ptb_text_only", split="train[:10%]")  # ~1MB, suficiente para prototipo
```

### Opción B — Para prueba semántica (evalúa el pilar de jerarquía hiperbólica)

**WordNet Noun Hierarchy**
- Tamaño: < 10 MB
- Estructura: árbol de hiperónimos (exactamente lo que el espacio hiperbólico captura)
- Descarga: via `nltk.corpus.wordnet`
- Tarea: Verificar que el embedding hiperbólico preserva distancias taxonómicas

```python
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
```

### Opción C — Para prueba end-to-end de generación (recomendada)

**TinyStories** (Microsoft Research, 2023)
- Tamaño: ~2 GB completo / ~50 MB para subconjunto de 10k historias
- Vocabulario: ~4,000 tokens únicos (muy manejable)
- Descarga: `roneneldan/TinyStories` en HuggingFace
- Tarea: Generación de texto corto y coherente
- **Por qué es perfecto para CHFT:** Las historias cortas tienen estructura semántica simple, vocabulario limitado y coherencia temática — condiciones ideales para validar el mecanismo de atractores.

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train[:5000]")  # ~20MB
```

---

## 6. Estimación de Tiempo de Entrenamiento (PC Local)

### Configuración de referencia (PC típica de desarrollo)

- CPU: Intel i7 / Ryzen 7 (8 núcleos)
- RAM: 16 GB
- GPU: NVIDIA RTX 3060 / 4060 (8-12 GB VRAM) ← recomendada
- Sin GPU: Solo CPU (significativamente más lento)

### Fase 1: Construcción del Codebook FHRR

| Tarea | Tiempo (CPU) | Tiempo (GPU) |
|---|---|---|
| Codificar vocabulario PTB (10k tokens) en FHRR d=4096 | ~2 min | ~15 seg |
| Codificar vocabulario TinyStories (4k tokens) en FHRR d=4096 | ~45 seg | ~8 seg |
| Construir índice de Hopfield para recuperación | ~5 min | ~1 min |

### Fase 2: "Entrenamiento" FHRR — Optimización de Fases

El "entrenamiento" en CHFT no es backpropagation clásico. Es un proceso de **ajuste de fases** por descenso de gradiente en el espacio de fases complejas. Mucho más liviano:

| Dataset | Épocas | Tiempo (CPU) | Tiempo (GPU) |
|---|---|---|---|
| PTB 10% (~1MB) | 10 | ~15 min | ~3 min |
| TinyStories 5k historias | 5 | ~45 min | ~8 min |
| TinyStories 5k historias | 20 | ~3 horas | ~30 min |

### Fase 3: Embedding Hiperbólico (Poincaré)

| Dataset | Tiempo (CPU) | Tiempo (GPU) |
|---|---|---|
| WordNet (10k nodos) | ~20 min | ~4 min |
| Vocabulario TinyStories | ~10 min | ~2 min |

### **Total estimado para prototipo funcional completo:**

| Escenario | Tiempo Total |
|---|---|
| Solo CPU, PTB, prototipo mínimo | **2-3 horas** |
| CPU + GPU, TinyStories, prototipo completo | **45 min - 1.5 horas** |
| Solo CPU, TinyStories, prototipo completo | **4-6 horas** |

> ⚠️ **Importante:** Estos tiempos son para el **prototipo de prueba de paradigma**, no para un modelo competitivo con LLMs. El objetivo es demostrar que los mecanismos FHRR + Hopfield + Hiperbólico pueden generar texto coherente en dominios restringidos, SIN backpropagation masiva.

---

## 7. Métricas para Validar el Paradigma

Para saber si CHFT v2 "funciona", definimos métricas concretas y honestas:

| Métrica | Qué mide | Éxito mínimo |
|---|---|---|
| **Precisión de recuperación FHRR** | % de tokens recuperados correctamente del codebook tras binding/unbinding | > 85% con d=4096 |
| **Coherencia de atractor** | % de veces que Hopfield converge al token más probable (vs. baseline random) | > 70% top-5 |
| **Preservación de jerarquía** | Correlación de Spearman entre distancias WN y distancias hiperbólicas | > 0.75 |
| **Perplexity en TinyStories** | Calidad de generación (menor = mejor) | < 150 (baseline 4-gram ≈ 200) |
| **Memoria de contexto** | Recuperación de información de N tokens atrás | > 60% a N=50 |

---

## 8. Limitaciones Reales de CHFT v2 (Honestidad Científica)

| Limitación | Severidad | Mitigación en v2 |
|---|---|---|
| Degradación de recuperación con vocabulario grande | Alta | Usar sub-vocabularios o codebooks jerárquicos |
| Generación no tan fluida como LLM en lenguaje libre | Media | CHFT excela en dominios estructurados; no compite en texto abierto en v1.0 |
| El manifold hiperbólico no captura polisemia bien | Media | Múltiples embeddings por token (multi-sense) |
| La capacidad de Hopfield moderna requiere d alto para capacidades reales | Baja | d=4096-8192 es alcanzable en CPU/GPU modernas |
| No hay "cero alucinaciones" sin fuente de verdad externa | Alta | Anclar el codebook a una base de conocimiento estructurada (KG) en v3 |

---

## 9. Hoja de Ruta de Implementación (Fases)

```
FASE 0 — Validación Matemática (Esta semana, 1-2 días)
  └── Implementar FHRR puro con torchhd
  └── Verificar binding/unbinding en vocabulario PTB
  └── Medir precisión de recuperación
  
FASE 1 — Prototipo de Atractores (Semana 2, 3-5 días)  
  └── Implementar memoria de Hopfield moderna
  └── Cargar TinyStories, codificar oraciones
  └── Generar texto simple (completar oraciones)
  
FASE 2 — Integración Hiperbólica (Semana 3-4, 5-7 días)
  └── Entrenar embedding de Poincaré en WordNet
  └── Proyectar FHRR al espacio hiperbólico
  └── Medir mejora en coherencia semántica
  
FASE 3 — Evaluación y Paper (Semana 5-6)
  └── Benchmarks comparativos vs. n-gram baseline
  └── Análisis de capacidad de memoria
  └── Documentación del paradigma verificado
```

---

## 10. Conclusión: ¿Vale la Pena?

La propuesta v1 tiene el problema de ser una **metáfora física elevada a promesa técnica**. La propuesta v2 hace el movimiento inverso: **toma la intuición poética correcta y la traduce a matemáticas implementables**.

El paradigma CHFT v2 **sí es original** en su combinación específica:
- FHRR ya existe, Hopfield moderna ya existe, embeddings hiperbólicos ya existen.
- La **fusión de los tres como arquitectura de lenguaje sin backpropagation**, con semántica continua y recuperación por resonancia, **no tiene un paper publicado que lo implemente de esta forma exacta**.

Este es el espacio de originalidad real y alcanzable.

> **Próximo paso recomendado:** Ejecutar el experimento de Fase 0. Si la recuperación FHRR supera el 85% en PTB, el paradigma tiene sustento empírico y vale la pena continuar.

---

*Propuesta v2 — CHFT: Campos Holográficos de Fase Topológica (Reformulado)*  
*Paradigma de fusión: FHRR + Hopfield Moderna + Geometría Hiperbólica*
