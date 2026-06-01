# CHFT: Complex Holographic Field Theory

An alternative neural network architecture utilizing complex-valued Vector Symbolic Architectures (VSA / Holographic Reduced Representations) and Continuous Associative Memory (Modern Hopfield Networks) to achieve constant $O(1)$ memory context scaling.

---

## 📄 Technical Papers & Research Results

Read our detailed technical reports analyzing the mathematical foundations, benchmarks, and target use cases:
* **English Version:** [CHFT Technical Paper v1 (EN)](file:///i:/01-Master_Code/Test-Labs/01-CHFT/CHFT_Technical_Paper_v1_EN.md)
* **Spanish Version:** [CHFT Technical Paper v1 (ES)](file:///i:/01-Master_Code/Test-Labs/01-CHFT/CHFT_Technical_Paper_v1.md)

---

## 📊 Core Benchmarks (CHFT v6 vs. 1-Layer Transformer)

Under identical training conditions (**1,000 TinyStories, 64 context size, 768 dimensions, 5 epochs on an RTX 3060 GPU**):

* **VRAM Peak Memory:** 1124.6 MB (CHFT v6) vs. 2239.1 MB (Transformer) — **~50% reduction** 🚀
* **Training Speed:** 3.7 minutes (CHFT v6) vs. 13.5 minutes (Transformer) — **3.6x faster** 🚀
* **Generation Diversity:** 48.9% (CHFT v6) vs. 28.9% (Transformer) — **Resolves repetitive loops natively** 🚀
* **Accuracy@1:** 28.17% (CHFT v6) vs. 37.10% (Transformer).

---

## 🛠️ Getting Started & Reproduction

### 1. Checkout the Paradigm v1 Commit (CHFT v6)
To reproduce the exact experiments, metrics, and logs of the first paradigm, switch to the specific Git commit:
```bash
git checkout 68c850679e25cb3b385785bcffeb9c7a431a4043
```

### 2. Environment Setup
The project uses `uv` for package and runtime management. Ensure `uv` is installed, then run the commands directly.

### 3. Run the Training Script
Run the entry point script using `uv` to train and evaluate the model under the benchmark parameters:
```bash
uv run main.py --stories 1000 --epochs 5 --reset
```

#### Parameters:
* `--stories 1000`: Trains on the first 1,000 stories of the TinyStories dataset.
* `--epochs 5`: Limits training to 5 epochs (matches the benchmark).
* `--reset`: Deletes any previous checkpoint to start training from scratch.
* `--dim 768`: Sets the complex phasor embedding dimension.
* `--context 64`: Context sequence window length.
* `--batch_size 256`: Number of training examples per batch.
