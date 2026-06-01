# Fast Benchmarks Reference: 1,000 Stories & 5 Epochs
*Date: June 1, 2026*

This document serves as our rapid iteration benchmark log. We use a reduced slice of **1,000 stories** and **5 epochs** to test architectural changes in minutes instead of hours.

---

## 📊 Comparative Table: Restored CHFT v2 vs. Transformer 1-Layer

| Metric | Baseline Freq | CHFT v2 (Fast Run) | Local Transformer 1-Layer (Fast Run) | Gap (CHFT vs. Transformer) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy@1** | 6.71% | 28.49% | **37.10%** | **-8.61pp** (CHFT is behind) |
| **Perplexity (PPL)** | 5,425.00 | 51.95 | **23.27** | **+28.68** (CHFT has higher uncertainty) |
| **Diversity Score** | — | **80.8%** 🚀 | 28.9% | **+51.90pp** (Transformer loops infinitely) |
| **Peak VRAM (GPU)** | — | **810.2 MB** 🚀 | 2239.1 MB | **-1428.9 MB** (2.76x less GPU VRAM!) |
| **Train Time** | — | **2.7 min** 🚀 | 13.5 min | **-10.8 min** (5x faster training!) |
| **Parameters** | — | 4.2M | 11.3M | -7.1M parameters |

---

## 🔍 Key Comparative Insights

1. **The Performance Gap**:
   * The **Transformer** achieves **37.10%** accuracy, which is **8.61pp** higher than **CHFT's 28.49%**.
   * The **Transformer** has lower perplexity (**23.27** vs **51.95**), meaning it is more confident in its token predictions.
   * However, the Transformer uses **11.3M parameters** at $D=768$, whereas CHFT uses only **4.2M parameters** (nearly 2.7x smaller parameter capacity).

2. **The Diversity Crisis**:
   * The **Transformer** has a diversity score of only **28.9%**, meaning it suffers from severe repetition. For example, in its output:
     * *"Once upon a time there a little little girl named queen a little girl named little girl named girl girl girl named..."*
   * **CHFT** maintains **80.8% diversity** and generates rich, non-looping sentences:
     * *"Once upon a time Sarah quickly jumped and threw the lesson, I carry this huge food as high..."*

3. **Compute Efficiency (The 5x Advantage)**:
   * **CHFT** trains in **2.7 minutes** on the RTX 3060, while the **Transformer** takes **13.5 minutes** (5 times slower!).
   * **CHFT** uses only **810.2 MB** of VRAM compared to the Transformer's **2.24 GB** (2.76x less memory footprint).
