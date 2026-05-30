# Propuesta de Mejoras: CHFT v4 (Cruzando la meta del 35.0%)

Para cerrar la brecha de **-2.30pp en Accuracy** y, sobre todo, para reducir la **Perplejidad de 33.58 a <10.0** (el target del Transformer es 8.00), debemos atacar el ruido de interferencia de fondo (crosstalk) en el hipervector de contexto $\Psi$. 

Proponemos las siguientes 4 líneas de mejora para la versión 4:

---

## 1. Escalar Dimensión e Historias Localmente (RTX 3060)
Como ya tienes configurada la ejecución local en tu GPU RTX 3060, podemos realizar experimentos que en Colab tomarían demasiado tiempo o causarían desconexiones:
* **Dimensión $D = 16,384$**: El espacio ortogonal en representaciones simbólicas (VSA) crece de forma no lineal con la dimensión. Duplicar la dimensión reduce el ruido de fondo (crosstalk) por un factor de $\sqrt{2}$, lo que bajará la perplejidad de forma directa.
* **Historias = 10,000**: Entrenar con más datos le permitirá a los pesos de atención posicional y a los fasores del vocabulario generalizar patrones gramaticales complejos.

---

## 2. Beta de Hopfield Aprendible (Trainable Scale $\beta$)
Actualmente, la temperatura inversa de la memoria de Hopfield está hardcodeada a un valor estático de $\beta = 16.0$ durante el cálculo de la entropía cruzada:
$$\mathcal{L} = \text{CrossEntropy}(\text{Logits} \times \beta, \text{Target})$$
* **Propuesta**: Hacer que $\beta$ sea un parámetro entrenable (`nn.Parameter`) inicializado en $16.0$ pero que el optimizador pueda ajustar libremente.
* **Impacto**: Si la escala es demasiado baja, el softmax distribuye probabilidad a tokens incorrectos (subiendo la perplejidad). Si es demasiado alta, satura el gradiente. Un $\beta$ dinámico autocalibrará la certidumbre del modelo época a época.

---

## 3. Capas de Normalización Compleja (Complex LayerNorm)
En arquitecturas de aprendizaje profundo, la normalización de capas (LayerNorm) estabiliza la dinámica de entrenamiento y reduce el ruido. 
* **Propuesta**: Añadir una capa de normalización de amplitud después de la suma de fasores y antes de cada paso de refinamiento Hopfield:
$$\Psi_{\text{norm}} = \text{LayerNorm}(\text{Abs}(\Psi)) \cdot e^{i \cdot \text{Angle}(\Psi)}$$
* **Impacto**: Estandariza la variabilidad del query state, permitiendo que la similitud de coseno en la memoria de Hopfield sea mucho más nítida y selectiva.

---

## 4. Multi-capa Hopfield Refinement (Deep Hopfield Recovery)
El modelo actual recupera el siguiente token en un solo bloque. Los Transformers obtienen su poder procesando la información a través de múltiples capas (ej. 6 a 12 capas).
* **Propuesta**: En lugar de solo refinar el query state en una capa, podemos encadenar 2 o 3 capas Hopfield secuenciales donde la salida de la primera capa se proyecta hacia atrás para refinar el contexto de entrada de la segunda capa.
* **Impacto**: Emula el razonamiento jerárquico multicapa del Transformer sin usar matrices de atención tradicionales.
