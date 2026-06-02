# 📋 CHFT Experiment Checklist & Results Tracker (`checklisttest.md`)

This document serves as the historical record of all architectural experiments, modifications, and validations conducted during the development of **Complex Holographic Field Theory (CHFT)**. It tracks which ideas successfully improved model accuracy/perplexity and which ones failed due to high-dimensional VSA (Vector Symbolic Architecture) properties or hardware constraints.

---

## 📊 1. Complete Model Evolution & Metrics

Below is a consolidated summary of all evaluated models, baseline standards, and experiment results on the **TinyStories** dataset.

| Version / Setup | Key Features | Acc@1 | Perplexity (PPL) | Diversity (TTR) | Peak VRAM | Train Time | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untrained Baseline** | Random weights, unnormalized logits | 0.61% | 97,451,090.90 | 7.0% | - | - | **Saturated** |
| **CHFT v2 (Flat Pos)** | Scale normalization, flat positional shifts | 8.29% | 236.87 | 34.0% | ~810 MB | 250s (10 ep) | **Retired** |
| **CHFT v2 (Orthogonal)** | Deterministic random frequencies + Exponential Decay | 24.62% | 93.06 | 36.2% | ~810 MB | 245s (10 ep) | **Success** |
| **CHFT v3 (First Run)** | Trainable Positional Attention + 1-Hop Hopfield + $D=8192$ | 32.70% | 33.58 | 47.6% | ~4.0 GB | 142.3m (10 ep) | **Success** |
| **CHFT v3 (Born Rule)** | QuantumNorm + Born Rule Similarity + Trainable Positional Frequencies | 5.57% | 456.10 | 33.0% | 1,480 MB | 13.1m (5 ep) | **Severe Failure** |
| **CHFT v3 (Restored)** | Reverted to fixed frequencies + ComplexLayerNorm | 28.49% | 51.95 | 80.8% | 810.2 MB | 2.7m (5 ep) | **Success** |
| **CHFT v4 (Path 1)** | Phase Unbinding (Reconstructing last token) | 26.64% | - | 53.2% | ~815 MB | ~3.0m (5 ep) | **Failure** |
| **CHFT v4 (Full)** | Salience Gating + Multi-Scale + Complex Low-Rank Transform | 27.85% | 49.98 | 69.0% | 1005.6 MB | 3.3m (5 ep) | **Success** |
| **CHFT v6 (Plano Champion)** | Pure Phase (restored champion v6, Dim=768, 1K stories) | **28.17%** | **48.69** | **48.9%** | **1124.6 MB** | **3.7m (5 ep)** | **CHAMPION** 🏆 |
| **CHFT v6 (Plano scaled)** | Dim=1024, 1,000 stories | 26.33% | - | - | - | 5.0m | **Success** |
| **CHFT v6 (Plano max)** | Dim=1024, 1,500 stories | 28.04% | - | - | 1.56 GB | 7.1m | **Success** |
| **CHFT v7.1 (HHC v1)** | Hierarchical Holographic Chunking (Dirty) | 22.96% | 81.83 | 43.5% | 1252.9 MB | 10.6m (5 ep) | **Failure** |
| **CHFT v7.2 (HHC v2)** | Nested Chunking + Micro Unbinding | 19.52% | 130.01 | 45.4% | 4472.9 MB | 69.5m (5 ep) | **Failure** |
| **CHFT v8 (CGRA)** | Complex Gated Recurrent Attractor (adds recurrency per coordinate) | 27.16% | 50.21 | 52.4% | 1124.6 MB | 252s (5 ep) | **Success** |
| **CHFT v9 (SHA)** | Spectral Holographic Attention + Dual-Path gate vs multiscale | 26.67% | 51.99 | **53.1%** | 1220.6 MB | 317s (5 ep) | **Partial** |
| **CHFT v10 (CSKA)** | Complex-Space Kernel Attention (ELU+1 kernel mapping) | 16.96% | 115.88 | 41.9% | 4005.1 MB | 700.3s (5 ep) | **Severe Failure** |
| **CHFT v11 (SA + Beta)** | Spreading Activation Energy + Adaptive Beta scaling | 27.30% | 50.73 | 53.1% | 1221.0 MB | 763.6s (5 ep) | **Success** |
| **CHFT v12 (MoE H-FFN)** | 8-Expert Sparse MoE on complex H-FFN block | 23.14% | 71.83 | 53.5% | 2229.4 MB | 460.0s (5 ep) | **Underfit / Slow** |
| **CHFT v13 (Space Fold)** | Hyperdimensional FFN Projection to 3072 dims (Ctx=64) | 24.31% | 63.13 | 50.9% | 1798.1 MB | 396.0s (5 ep) | **Partial** |
| **CHFT v13 (Ctx=128)** | Space-Folding with context scaled to 128 | 24.87% | 61.87 | 48.7% | 2773.5 MB | 536.0s (5 ep) | **Partial** |
| **CHFT v14 (MH-CGRA)** | 8-head Subspace Attractor CGRA + mix (Ctx=128) | 27.07% | 50.49 | 56.1% | 2214.8 MB | 455.2s (5 ep) | **Success** |
| **CHFT v14 (MH-CGRA Ctx=64)** | 8-head Subspace Attractor CGRA + mix (Ctx=64) | 27.96% | 48.67 | 56.5% | 1240.9 MB | 319.4s (5 ep) | **Success** |
| **CHFT v15 (UHP)** | strictly unitary Householder Projections (K=8, Ctx=64) | **28.44%** | **49.01** | **60.5%** | **1214.3 MB** | **372.0s (5 ep)** | **NEW CHAMPION** 🏆 |
| **Transformer 1L** | Causal Self-Attention Baseline (3.1M parameters) | 37.10% | 15.35 | 33.1% | 2573.6 MB | 43.9m (10 ep) | **Ref Target** |

---

## 🏆 2. What Worked (Successes)

### 1. Orthogonal Positional Binding
* **Concept**: Assigning deterministic random frequencies $\omega \in [-\pi, \pi]^D$ to each relative position index.
* **Why it worked**: Established mathematical pseudo-orthogonality between positions. It prevents different slots in the context window from bleeding into one another during summation, preserving word order.
* **Metric Boost**: Accuracy rose from **8.29%** to **24.62%**.

### 2. Exponential Context Decay
* **Concept**: Multiplying older tokens in the context window by a geometric decay factor $\gamma^{L-1-j}$ (initialized at $\gamma = 0.85$).
* **Why it worked**: Establishes a soft recency bias, reducing the interference noise generated by distant, less relevant context words on the prediction target.

### 3. Complex LayerNorm (High-Pass Centering)
* **Concept**: Subtracting the mean magnitude across dimensions and dividing by variance before applying $\sqrt{D}$ scaling.
* **Why it worked**: High-dimensional vector summations naturally accumulate a positive DC bias. Complex LayerNorm acts as a high-pass filter, centering the representation sphere and keeping semantic coordinates highly discriminative.

### 4. Content-Dependent Salience Gating (Path 2)
* **Concept**: Adding a learnable gating parameter $g_t \in [0, 1]$ based on the token phase: $g_t = \sigma(\text{Linear}(\phi_t))$.
* **Why it worked**: Successfully learned to down-weight noise from highly frequent but semantically insignificant filler words ("the", "a", "and") before they entered the superposition bundle.
* **Metric Boost**: Accuracy improved to **27.33%** and diversity stabilized.

### 5. Multi-Scale Positional Bundling (Path 4)
* **Concept**: Segmenting the context window into short-term (last 16 tokens) and long-term (remaining 48 tokens) temporal chunks, blending them independently, and scaling their contributions with learnable parameters.
* **Why it worked**: Captured different levels of grammatical context without letting long-term semantic drift overwrite the immediate syntax.
* **Metric Boost**: Accuracy rose to **27.67%**.

### 6. Complex Low-Rank Transformation (Path 3)
* **Concept**: Applying a complex projection layer of rank $r=64$ using two complex matrices $U_{\text{conj}} \in \mathbb{C}^{r \times D}$ and $V \in \mathbb{C}^{D \times r}$ to the bundle before Hopfield retrieval.
* **Why it worked**: Gave the model a parameter-efficient learnable mapping to translate context state phases into predicted target phases, acting like a projection head.
* **Metric Boost**: Achieved a record low perplexity of **49.98** (vs. 51.95 baseline) and validation loss of **3.9110**.

### 7. Seeded Dataset Splits & Normalization Alignment
* **Concept**: Seeding the train/validation permutation generator and feeding the Complex LayerNorm parameters (`ln_fn`) directly to the inference text generator loop.
* **Why it worked**: Seeded splits eliminated data leakage between checkpoint runs, providing true, uninflated accuracy metrics. Alignment of LayerNorm resolved the "word soup" generation bug, resulting in cohesive sentences.

### 8. Complex Gated Recurrent Attractor (CGRA / CHFT v8)
* **Concept**: Replacing the simple additive residual updates in Hopfield attractor hops with a dimension-wise gated recurrent update (GRU-style adapted to complex representations).
* **Why it worked**: Enabled coordinates to learn individual update, reset, and candidate phase projections, filtering high-dimensional VSA summation noise while maintaining low memory footprint and high generation diversity.
### 9. Spreading Activation Energy & Adaptive Beta (CHFT v11)
* **Concept**: Replacing positional decay with a 4-component spreading activation weight equation (incorporating depth decay, relational gate strength, precomputed token frequency, and recency bonus) and scaling the Modern Hopfield Memory temperature ($\beta$) dynamically based on query retrieval confidence.
* **Why it worked**: Filtered high-dimensional VSA cross-talk noise during summation, enabling the model to prioritise high-value context nodes. Scaling beta dynamically dynamically sharpens retrieval focus during confident matches while softening it for noisier contexts, preventing attraction to false local minima.
* **Metric Boost**: Accuracy improved from **26.67%** (v9) to **27.30%** (+0.63pp), and perplexity dropped from **51.99** to **50.73** (-1.26), rescuing the network from the CSKA (v10) collapse.

### 10. Multi-Head Subspace Attractor Routing (MH-CGRA / CHFT v14)
* **Concept**: Partitioning the context bundle during Modern Hopfield Memory attractor routing into $H=8$ parallel subspace heads ($d=96$), updating each head with its own independent recurrent CGRA gates, and performing inter-head complex linear mixing between attractor hops.
* **Why it worked**: Successfully distributed capacity into parallel subspaces, letting the model route distinct semantic features (e.g. subject, verb, context) in parallel without VSA crosstalk interference. It scaled remarkably well when context length doubled to $C=128$, maintaining the flat memory paradigm.
* **Metric Boost**:
  * At **Ctx=128**, achieved **27.07% Accuracy** and a lower perplexity of **50.49** (vs. 50.73 in v11 at Ctx=64), outperforming the space-folding projection models while using 550 MB less VRAM.
  * At **Ctx=64**, achieved a record low perplexity of **48.67** (beating the champion v6's 48.69) and **27.96% Accuracy** (bridging the accuracy gap to just 0.21pp), while accelerating training time to **5.3 minutes** due to `torch.einsum` optimizations.

### 11. Unitary Householder Projections (UHP / CHFT v15)
* **Concept**: Replacing non-unitary projections (in both the Q-projection head and inter-head mixing) with sequences of $K=8$ strictly unitary complex Householder reflections.
* **Why it worked**: Guaranteed mathematically lossless rotations in complex phase space. By eliminating phase angle distortion and magnitude degradation during projections, it preserved FHRR coordinate coherence, allowing the Hopfield attractor dynamics to cleanly classification target tokens.
* **Metric Boost**: Achieved a new **undisputed record Accuracy@1 of 28.44%** (outperforming champion v6's 28.17%) and a low perplexity of **49.01**, while using the least VRAM (1214.3 MB) and training parameters (13.6M) among advanced variants.

---

## ❌ 3. What Failed (Failures & Defeats)

### 1. Phase Interference Unbinding (Path 1)
* **Concept**: Isolating the last token ($C-1$) and multiplying it by the conjugate of its positional rotation to unbind it (extracting it from the position vector space).
* **Why it failed**: Reconstructing raw coordinates in Hopfield memory created a severe **self-similarity loop bias**. The query vector was dominated by the identity of the last token itself, causing the text generator to repeat words indefinitely.
* **Impact**: Accuracy dropped to **26.64%** and diversity collapsed.

### 2. Quantum Wavefunction Normalization (QuantumNorm)
* **Concept**: Normalizing the global vector by its L2 norm: $\Psi_{\text{norm}} = \Psi / \|\Psi\|_2$, removing component-wise LayerNorm.
* **Why it failed**: Without component-wise centering, high-dimensional summation accumulated a severe positive DC bias. All semantic hypervectors shifted towards the same quadrant, destroying their pseudo-orthogonality and collapsing retrieval.
* **Impact**: Accuracy plummeted to **5.57%** (near random chance).

### 3. Learnable Positional Frequencies ($\omega$)
* **Concept**: Making position rotation frequencies `omega` learnable parameters updated via gradient descent.
* **Why it failed**: The optimizer collapsed the positional frequencies towards a single value to minimize immediate training loss. This destroyed position orthogonality, merging all sequence indices into a blurred bag-of-words soup.
* **Impact**: Contributed to the **5.57%** accuracy collapse in CHFT v3.

### 4. Hierarchical Holographic Chunking (HHC) (v7.1 & v7.2)
* **Concept**: Organizing the context window recursively into chunks (words $\rightarrow$ sentences $\rightarrow$ sequence) using nested bindings.
* **Why it failed**:
  1. **Double Unbinding Noise**: VSA binding and unbinding operations are mathematically approximate. Applying nested unbindings ($e^{-i\Theta} \cdot e^{-i\theta}$) multiplied the reconstruction crosstalk noise, dispersing the signal.
  2. **Tensor Expansion Bottleneck**: Cleaning attractors jerárquicamente required expanding intermediate tensors to `[B, M, K, V]`, creating 90M elements per Hopfield step. This exploded VRAM usage to **4.4 GB** (a 400% increase) and slowed training by **18x** (69.5 min vs. 3.7 min).
* **Impact**: Accuracy collapsed to **19.52%** and VRAM exploded.

### 5. Complex-Space Kernel Attention (CSKA) (v10)
* **Concept**: Replacing dual-path context bundling with kernel-based linear attention using feature mapping $\phi(x) = \text{ELU}(x) + 1.0$ and value projection mapped to an accumulated matrix $Z \in \mathbb{C}^{M \times D}$.
* **Why it failed**:
  1. **Loss of Holographic Superposition Fidelity**: Projecting complex phasors into a real-valued kernel projection space and computing $Z$ destroyed the precise phase-relations necessary for FHRR binding.
  2. **VRAM and Compute Explosion**: Expanding to intermediate feature maps and accumulation tensors exploded Peak VRAM to **4.0 GB** (a 230% increase) and training time to **11.7 min** (a 120% increase) without any scaling advantages.
* **Impact**: Accuracy plummeted to **16.96%** (down from 26.67%) and perplexity degraded to **115.88** (up from 51.99).

### 6. Sparse Mixture of Experts (MoE) on H-FFN (v12)
* **Concept**: Scaling the model's capacity by replacing the dense `HolographicMLP` with an 8-expert Sparse MoE block routing to Top-2 experts.
* **Why it regressed/underperformed**:
  1. **Data Scaling & Overfitting**: Expanding parameters from 13.8M to 80M created too much capacity for a small 1,000-story dataset (200k samples).
  2. **Sparse Gradient Updates**: Since only 2 of 8 experts are updated per token, convergence is much slower, requiring significantly more epochs or data to balance expert routing and stabilize weights.
* **Impact**: Perplexity degraded to **71.83** and Accuracy dropped to **23.14%** (under 5 epochs), with VRAM increasing to **2.2 GB**.

### 7. Hyperdimensional Space-Folding Projections (v13)
* **Concept**: Projecting the flat 768-dimensional phasor $\Psi$ into a wide 3072-dimensional space using trainable complex weights, applying dense H-FFN, and folding back to 768 dimensions.
* **Why it regressed (compared to v11)**: While it significantly outperformed the MoE model by utilizing dense updates, standard random initialization of projection matrices degrades phasor coherence (FHRR phase angles). Without unitary or orthogonal constraints on the projection weights, mapping between spaces distorts the phase relationships, introducing noise that degrades Hopfield retrieval.
* **Impact**: Perplexity improved to **63.13** (down from 71.83 in MoE) and Accuracy improved to **24.31%** (up from 23.14%), but did not reach the v11 baseline of **27.30% Acc**. VRAM was extremely efficient at **1.8 GB**.

---


## 🔮 4. Future Test Candidates (Ideas to Try)

- [x] **Complex Gated Recurrent Attractor (CGRA)**: Inserting a light recurrent phase transition gate between Hopfield hops to update the query dynamically without exploding VRAM. *(Implemented in CHFT v8)*
- [x] **SHA + Dual-Path**: Correct holographic attention on individual token phasors + gated blend with multiscale. PPL 51.99, Acc 26.67% — no collapse, high diversity, but didn’t beat v6. Gate blending likely diluted SHA’s advantage. *(Implemented in CHFT v9)*
- [ ] **SHA on Content-Only Phasors (v9.1)**: Compute attention scores using RAW phase vectors (before positional binding) for clean semantic similarity. Use position-bound H for weighted sum. Separates “What to attend to” (content) from “what to retrieve” (positional context).
- [ ] **Dual-Path Phase Space**: Combining a fast pure-phase direct mapping (CHFT v6 Plano) and a slow low-rank complex projection path in parallel.
- [ ] **Dynamic Temperature Scaling**: Modifying $\beta$ dynamically per token during Hopfield retrieval based on context entropy.
