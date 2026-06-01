# Summary of Key Learnings & Decisions

This document consolidates and compacts the key theoretical insights, technical adjustments, and architectural decisions made during our development of the **CHFT (Complex Holographic Field Theory)** paradigm.

---

## 1. PyTorch Loading warning (`weights_only`)
* **Issue:** PyTorch throws a `FutureWarning` because `torch.load` defaults to `weights_only=False`, utilizing the Python `pickle` module, which poses security risks (untrusted models can execute arbitrary code).
* **Solution:** Explicitly pass `weights_only=True` when loading state dicts containing safe primitives (lists, dicts, tensors, numbers).
  ```python
  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
  ```

---

## 2. Context Window & Memory Complexity ($O(1)$ Footprint)
* **Standard LLMs:** Require storing separate Key/Value vectors for every token in the context window. Memory consumption (KV-cache) grows linearly with context size, leading to high VRAM overhead at scale.
* **CHFT Paradigm:** Compresses (bundles) the entire context history into a **single vector** ($\psi$) of fixed dimensionality. Regardless of context length (e.g. 8 or 1,000,000 tokens), the history representation memory size remains constant (e.g., 131 KB for `DIMENSION = 16,384`).

---

## 3. Parameter-to-Data Compression Efficiency
* In CHFT, $99.9\%$ of the parameters are contained within the **Vocabulary Phase Embeddings** (`vocab_size * DIMENSION`). Because update steps are sparse and use additions/rotations, training is exceptionally fast compared to standard Transformers.
* A **Nano-scale (5.8M parameters)** model trained on only **3,000 stories** (0.14% of Microsoft's dataset) achieved **30.27% Accuracy** on unseen validation sequences. This demonstrates massive structural efficiency in learning language patterns from limited samples.

---

## 4. Multi-Layer Scaling Strategies in CHFT
To scale reasoning and syntax depth without standard Transformer scaling overhead:
1. **Attractor Refinement Layers:** Increase multi-hop steps inside the Hopfield memory. (Adds 0 parameters, uses 0 extra VRAM).
2. **Hierarchical Context Layers:** Algebraically group tokens into sentences, and sentences into paragraph-level vectors. (Adds 0 parameters when using fixed VSA operators).
3. **Translation/FFN Layers:** Insert feed-forward layers after Hopfield retrieval to translate/clean VSA crosstalk noise and learn complex rules.

---

## 5. Eliminating Data Leakage
* **Bug:** Shuffling without a fixed seed caused training and validation sets to split differently when resuming training, leaking previously trained data into the validation loop and skewing metrics.
* **Fix:** Initialized a deterministic generator seed for train/val splits:
  ```python
  g = torch.Generator().manual_seed(42)
  perm = torch.randperm(num_total, generator=g)
  ```

---

## 6. Story Concatenation (Token Packing)
To support large context sizes (e.g. `CONTEXT_LEN = 256`) without losing short stories, we pack all tokenized narratives into a continuous stream separated by `<|endoftext|>` before creating sliding windows, preventing dataset depletion.

---

## 7. Metrics & Resource Monitoring
* **Peak VRAM Monitoring:** Added tracking via `torch.cuda.max_memory_allocated()` to confirm that VRAM stays flat during scaling tests.
* **Benchmark Targets:**
  * **Transformer 1-Layer:** ~10 PPL, ~18-25% Acc.
  * **Transformer 2-Layer:** ~6.5-8.0 PPL, ~35-40% Acc.
  * **CHFT (Nano Target):** Aim for < 6.0 PPL and > 50% Acc.