# CHFT Paradigm and Roadmap: Towards v3, v4, and v5

This document details the recent mathematical shift in our **CHFT (Complex Holographic Field Theory)** paradigm and analyzes three proposed research directions to close the performance gap with standard Transformer baselines.

---

## 🌌 1. The CHFT v3 Paradigm: Quantum Physical Fidelity

In **CHFT v3**, we transitioned from standard deep learning normalization heuritsics to equations derived from quantum mechanics and wave propagation. This represents a fundamental shift in our representation paradigm:

### A. Quantum Wavefunction Normalization (QuantumNorm)
* **Old (v2)**: Component-wise `ComplexLayerNorm` which normalized magnitude elements independently to mean 0 and variance 1 across dimensions.
* **New (v3)**: Global L2 Hilbert space normalization:
  $$\Psi_{\text{norm}} = \frac{\Psi}{\|\Psi\|_2}$$
* **Why it matters**: In quantum mechanics, the total probability must be conserved ($\|\Psi\|_2 = 1$). Normalizing components independently destroyed the spatial phase coherence. `QuantumNorm` preserves exact relative interference amplitudes (constructive and destructive), significantly improving the Signal-to-Noise Ratio (SNR) in the phase space.

### B. Born Rule Similarity (U(1) Gauge Invariance)
* **Old (v2)**: Similarity measured by projecting onto the real axis: $\text{sim}_v = \text{Re}(\langle K_v, \Psi \rangle)$.
* **New (v3)**: Similarity measured via the Born rule projection magnitude:
  $$\text{sim}_v = \frac{|\langle K_v, \Psi \rangle|}{\sqrt{D}}$$
* **Why it matters**: Measuring projection only on the real axis breaks phase-rotation gauge symmetry. In v3, using the complex modulus ($|\cdot|$) guarantees **U(1) Gauge Invariance** (local and global phase rotation invariance). The model is now robust against global phase shifts of the context state, utilizing the full complex plane for semantic representation.

### C. Dispersive Positional Wave Propagation
* **Old (v2)**: Deterministic, linear positional binding: $\theta_p = p \cdot \omega$.
* **New (v3)**: Learnable dispersive quadratic binding:
  $$\theta_p = p \cdot \omega_1 + p^2 \cdot \omega_2$$
  where $\omega_1, \omega_2$ are trainable parameters.
* **Why it matters**: Mimicking a physical dispersive wave medium (where different frequencies propagate at different velocities), the quadratic term allows the model to learn to dispersively spread or focus phase information across the sequence length, enabling selective attention decay.

---

## 🛠️ 2. Analysis of the Three Proposals (CHFT v4 & v5)

To bridge the remaining **-9.20pp Accuracy@1 gap** against the 1-Layer Transformer, we evaluate three potential research directions:

### 🚀 Proposal A: Sparse Selective Bundling (Salience-Gated Superposition)
Instead of summing all context tokens blindly, we introduce a gating mechanism that dynamically scales the phase contribution of each token before bundling:
$$\Psi = \sum_{p=0}^{C-1} g_p W_p \cdot e^{i (\phi_p + \theta_p)}$$
where $g_p = \sigma(\text{ComplexLinear}(e^{i \phi_p}))$ is a trainable complex gate.

* **Physical Analogy**: State preparation under a selective local field. It acts as a filter that drops background crosstalk noise from semantically insignificant tokens (like punctuation or common filler words).
* **Novelty**: Extremely high. Combining Holographic Reduced Representations (HRR) with content-dependent gating retains $O(1)$ context memory footprint while achieving attention-like selectivity.
* **Impact**: Directly increases the SNR of the superposition vector $\Psi$, lowering Perplexity.

### 🔍 Proposal B: Phase Interference Unbinding (Slot-Addressable Querying)
Instead of querying the Modern Hopfield Memory blindly with the raw bundle $\Psi$, we query it with a sequence of target-unbound vectors. In VSA, applying the inverse rotation $e^{-i k \omega}$ reconstructs the token at position $k$ with signal strength $D$, leaving all other context tokens as noise:
$$\Psi_{\text{unbind}}(k) = \Psi \cdot e^{-i k \omega}$$
During training/inference, the query is sent to Hopfield as:
$$\text{state}_{\text{query}} = \Psi \cdot e^{-i (C-1) \omega} \quad \text{(extracting the predecessor token)}$$

* **Physical Analogy**: Quantum state teleportation or hologram readout using the reference beam.
* **Novelty**: High. Reconstructs cross-attention mechanisms at constant memory cost by treating the context bundle $\Psi$ as a hardware key-value store, queryable by position offsets.
* **Impact**: Will dramatically improve Accuracy@1 for syntax and sequential grammars, as the model can query specific "slots" in its past.

### 🎛️ Proposal C: Learnable Complex Attention (Hybrid Phase-Space Gates)
Keep the FHRR codebook but introduce a complex-valued linear operator $W_A \in \mathbb{C}^{D \times D}$ that acts on the context vector $\Psi$ in the frequency/phase domain:
$$\Psi' = W_A \Psi$$
* **Physical Analogy**: A unitary quantum gate (like a Hadamard or phase-shift matrix) acting on the wavefunction.
* **Novelty**: Genuinely novel. Applying complex linear algebra transformations to VSA representations prior to Hopfield retrieval allows mixing of phase frequencies to capture multi-token grammatical structures (e.g., noun-verb relations) without relying on token-space sequence matrices.
* **Impact**: Increases model capacity linearly while preserving $O(1)$ scaling over context length.

---

## 📈 3. Implementation Roadmap for CHFT v4

If we decide to implement these, we recommend the following integration order:

| Phase | Feature | Target Metic | Complexity | Status |
| :--- | :--- | :---: | :---: | :---: |
| **CHFT v3** | Born Rule + QuantumNorm + Dispersive Binding | Robust Phase Coherence | Medium | **Implemented & Validated** |
| **CHFT v4** | Proposal A (Sparse Selective Bundling) | Perplexity < 20.0 | Medium | *Candidate for next step* |
| **CHFT v4.5** | Proposal B (Phase Interference Unbinding) | Accuracy@1 > 38.0% | High | *Candidate* |
| **CHFT v5** | Proposal C (Frequency-Domain Complex Gates) | Accuracy@1 > 42.0% | High | *Candidate* |
