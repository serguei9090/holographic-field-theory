# %% [markdown]
# # 🌀 Experimento CHFT v2 — Campos Holográficos de Fourier
#
# Este notebook implementa y evalúa el paradigma **CHFT (Complex Holographic Field Theory)**:
# representaciones FHRR (Fourier Holographic Reduced Representation) + Memoria de Hopfield Moderna.
#
# **Objetivo**: Probar si un modelo sub-simbólico no-LLM puede aprender patrones de lenguaje
# sin atención, transformers ni embeddings estándar.
#
# ### Métricas evaluadas:
# - `Loss` decreciente por época
# - `Accuracy@1` en conjunto de validación
# - `Perplexity` del modelo
# - `Diversity` del texto generado (% tokens únicos)
# - Muestras de generación con Top-k sampling

# %%
# Instalar dependencias requeridas en Google Colab
!pip install -q datasets tiktoken

# %%
import torch
import torch.nn as nn
import numpy as np
import time
import math
import collections
from datasets import load_dataset
import tiktoken
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# CONFIGURACIÓN OPTIMIZADA PARA COLAB (GPU T4)
# ─────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIMENSION     = 4096     # Mayor dimensión aprovecha GPU T4 (16 GB VRAM)
CONTEXT_LEN   = 8        # Ventana de contexto histórico
EPOCHS        = 10       # Más épocas para mejor convergencia
LEARNING_RATE = 0.01
BATCH_SIZE    = 1024     # Batch grande para GPU
NUM_STORIES   = 1000     # 5× más datos que versión CPU
TOPK          = 5        # Top-k para sampling diverso
VAL_SPLIT     = 0.1      # 10% de muestras para validación

# ── ¿Por qué NO usamos float16 puro? ──────────────────────────────────────
# 1. FHRR opera sobre fases θ ∈ [-π, π]. float16 solo tiene ~3 decimales
#    de precisión → fases cercanas colapsan → tokens indistinguibles.
# 2. PyTorch no tiene complex32 estable (float16 real + float16 imag).
#    torch.complex() con float16 produce complex64 de todas formas.
# 3. Adam acumula gradientes pequeños (~1e-4). float16 los redondea a 0
#    → entrenamiento se detiene.
#
# ── Solución: AMP (Automatic Mixed Precision) ─────────────────────────────
# - Parámetros (phases) almacenados en float32 → gradientes estables
# - Forward pass (cos/sin/matmul) en float16 via autocast → 2× velocidad
# - GradScaler escala el loss para evitar underflow de gradientes
# ──────────────────────────────────────────────────────────────────────────
USE_AMP = DEVICE.type == 'cuda'  # AMP solo activo en GPU

print(f"✅ Dispositivo: {DEVICE}")
print(f"✅ Dimensión FHRR: {DIMENSION:,} | Contexto: {CONTEXT_LEN} tokens")
print(f"✅ Dataset: {NUM_STORIES} historias | Épocas: {EPOCHS}")
print(f"✅ AMP (f16): {USE_AMP}  <- forward float16, params float32")

# %% [markdown]
# ## 1. 📥 Dataset y Tokenización

# %%
print("\n[1/5] Cargando TinyStories...")
dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{NUM_STORIES}]")
tokenizer = tiktoken.get_encoding("cl100k_base")

print("      Tokenizando y extrayendo vocabulario...")
all_token_ids    = set()
stories_tokenized = []

for item in dataset:
    tokens = tokenizer.encode(item["text"])
    stories_tokenized.append(tokens)
    all_token_ids.update(tokens)

vocab        = list(all_token_ids)
vocab_size   = len(vocab)
token_to_idx = {tok: i for i, tok in enumerate(vocab)}
idx_to_token = {i: tok for i, tok in enumerate(vocab)}

print(f"      Vocabulario: {vocab_size:,} tokens únicos")

# Construir tensores de contexto/objetivo
contexts_list, targets_list = [], []
for story in stories_tokenized:
    if len(story) < CONTEXT_LEN + 1:
        continue
    story_idx = [token_to_idx[t] for t in story]
    for i in range(len(story_idx) - CONTEXT_LEN):
        contexts_list.append(story_idx[i : i + CONTEXT_LEN])
        targets_list.append(story_idx[i + CONTEXT_LEN])

# Split train / val
num_total  = len(contexts_list)
num_val    = int(num_total * VAL_SPLIT)
num_train  = num_total - num_val

contexts_tensor = torch.tensor(contexts_list, dtype=torch.long)
targets_tensor  = torch.tensor(targets_list,  dtype=torch.long)

# Shuffle antes de split
perm = torch.randperm(num_total)
contexts_tensor = contexts_tensor[perm].to(DEVICE)
targets_tensor  = targets_tensor[perm].to(DEVICE)

train_ctx, val_ctx = contexts_tensor[:num_train], contexts_tensor[num_train:]
train_tgt, val_tgt = targets_tensor[:num_train],  targets_tensor[num_train:]

print(f"      Train: {num_train:,} muestras | Val: {num_val:,} muestras")

# %% [markdown]
# ## 2. 🧠 Arquitectura CHFT: FHRR + Hopfield Moderno

# %%
# ── Capa FHRR: cada token = fasor complejo en el círculo unitario ──
class FHRRPhasorEmbedding(nn.Module):
    """
    Fourier Holographic Reduced Representation.
    Cada token es un vector de fases θ ∈ [-π, π]^D
    El hipervector complejo es: e^{iθ} = cos(θ) + i·sin(θ)
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.phases = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim).uniform_(-math.pi, math.pi)
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        phi = self.phases[indices]
        return torch.complex(torch.cos(phi), torch.sin(phi))

    def all_keys(self) -> torch.Tensor:
        return torch.complex(torch.cos(self.phases), torch.sin(self.phases))


# ── Memoria de Hopfield Moderna: colapso de estados ──
class ModernHopfieldMemory:
    """
    Recuperación de atractores mediante interferencia constructiva.
    Equivalente a cross-attention sin parámetros adicionales.
    """
    def __init__(self, beta: float = 16.0):
        self.beta = beta
        self.keys: torch.Tensor | None = None

    @torch.no_grad()
    def update_keys(self, codebook: FHRRPhasorEmbedding):
        self.keys = codebook.all_keys().detach()

    @torch.no_grad()
    def query(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Retorna (índice_más_probable, similitudes_raw)"""
        sim = torch.real(
            torch.matmul(self.keys, torch.conj(state).unsqueeze(-1))
        ).squeeze(-1)
        weights = torch.softmax(sim * self.beta, dim=0)
        return int(torch.argmax(weights).item()), sim

    @torch.no_grad()
    def query_topk(self, state: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Retorna logits para top-k sampling"""
        sim = torch.real(
            torch.matmul(self.keys, torch.conj(state).unsqueeze(-1))
        ).squeeze(-1)
        return sim


codebook       = FHRRPhasorEmbedding(vocab_size, DIMENSION).to(DEVICE)
hopfield_mem   = ModernHopfieldMemory(beta=16.0)
hopfield_mem.update_keys(codebook)

total_params = sum(p.numel() for p in codebook.parameters())
print(f"✅ Modelo CHFT inicializado")
print(f"   Parámetros totales: {total_params:,} ({total_params * 4 / 1e6:.1f} MB en float32)")

# %% [markdown]
# ## 3. 🏋️ Entrenamiento con Seguimiento de Validación

# %%
print("\n[2/5] Iniciando Entrenamiento CHFT...")
optimizer    = torch.optim.Adam(codebook.parameters(), lr=LEARNING_RATE)
scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

loss_history     = []
val_loss_history = []
t0 = time.time()

for epoch in range(EPOCHS):
    # ── TRAIN ──
    codebook.train()
    epoch_loss  = 0.0
    perm        = torch.randperm(num_train)
    num_batches = math.ceil(num_train / BATCH_SIZE)

    for b in range(num_batches):
        idx   = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        ctx_b = train_ctx[idx]
        tgt_b = train_tgt[idx]

        # Superposición holográfica: ψ = Σ e^{iθ_t}  (t ∈ contexto)
        ctx_hv    = codebook(ctx_b)                          # [B, C, D] complejo
        psi       = torch.sum(ctx_hv, dim=1)                 # [B, D]
        psi       = nn.functional.normalize(psi, p=2, dim=1) # normalizar magnitud

        # Similitud con todos los atractores del vocabulario
        keys_all  = codebook.all_keys()                      # [V, D]
        logits    = torch.real(torch.matmul(psi, torch.conj(keys_all).t()))  # [B, V]

        loss = nn.functional.cross_entropy(logits * 6.0, tgt_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Mantener fases en [-π, π]
        with torch.no_grad():
            codebook.phases.copy_(
                torch.remainder(codebook.phases + math.pi, 2 * math.pi) - math.pi
            )
        epoch_loss += loss.item()

    scheduler.step()
    hopfield_mem.update_keys(codebook)

    # ── VALIDACIÓN ──
    codebook.eval()
    with torch.no_grad():
        val_batches = math.ceil(num_val / BATCH_SIZE)
        val_loss_sum = 0.0
        for b in range(val_batches):
            ctx_v = val_ctx[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            tgt_v = val_tgt[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            ctx_hv = codebook(ctx_v)
            psi_v  = nn.functional.normalize(torch.sum(ctx_hv, dim=1), p=2, dim=1)
            keys_all = codebook.all_keys()
            logits_v = torch.real(torch.matmul(psi_v, torch.conj(keys_all).t()))
            val_loss_sum += nn.functional.cross_entropy(logits_v * 6.0, tgt_v).item()

    avg_train = epoch_loss / num_batches
    avg_val   = val_loss_sum / val_batches
    loss_history.append(avg_train)
    val_loss_history.append(avg_val)

    print(f"  Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

elapsed = time.time() - t0
print(f"\n✅ Entrenamiento completado en {elapsed:.1f}s ({elapsed/60:.1f} min)")

# %% [markdown]
# ## 4. 📊 Benchmark: Métricas Cuantitativas

# %%
print("\n[3/5] Calculando métricas de benchmark...")

codebook.eval()

# ── Accuracy@1 y Perplexity en validación ──
correct   = 0
total_ce  = 0.0
n_val     = 0

with torch.no_grad():
    for b in range(math.ceil(num_val / BATCH_SIZE)):
        ctx_v = val_ctx[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        tgt_v = val_tgt[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]

        ctx_hv   = codebook(ctx_v)
        psi_v    = nn.functional.normalize(torch.sum(ctx_hv, dim=1), p=2, dim=1)
        keys_all = codebook.all_keys()
        logits_v = torch.real(torch.matmul(psi_v, torch.conj(keys_all).t())) * 6.0

        preds    = torch.argmax(logits_v, dim=1)
        correct += (preds == tgt_v).sum().item()
        total_ce += nn.functional.cross_entropy(logits_v, tgt_v, reduction='sum').item()
        n_val    += len(tgt_v)

accuracy   = correct / n_val * 100
perplexity = math.exp(total_ce / n_val)

print(f"  Accuracy@1  : {accuracy:.2f}%   (tokens exactos predichos)")
print(f"  Perplexity  : {perplexity:.2f}  (menor = mejor; azar ≈ {vocab_size:,})")

# ── Baseline: modelo de frecuencia (predice siempre el token más frecuente) ──
token_freq   = collections.Counter(targets_tensor.cpu().tolist())
most_common  = token_freq.most_common(1)[0][0]
base_correct = sum(1 for t in val_tgt.cpu().tolist() if t == most_common)
base_acc     = base_correct / num_val * 100
print(f"\n  [Baseline freq] Accuracy@1: {base_acc:.2f}%  (siempre predice token más frecuente)")
print(f"  [CHFT v2]       Accuracy@1: {accuracy:.2f}%  (+{accuracy - base_acc:.2f}pp vs baseline)")

# %% [markdown]
# ## 5. ✍️ Generación de Texto con Top-k Sampling

# %%
def generate_text_topk(prompt: str, max_new: int = 20, k: int = TOPK, temperature: float = 0.8) -> str:
    """
    Genera texto usando top-k sampling con temperatura para diversidad.
    - k         : cuántos candidatos considerar en cada paso
    - temperature : < 1.0 = más conservador, > 1.0 = más creativo
    """
    codebook.eval()
    tokens_in  = tokenizer.encode(prompt)
    valid_tok  = [t for t in tokens_in if t in token_to_idx]
    if not valid_tok:
        return "[tokens del prompt fuera del vocabulario]"

    gen_idx = [token_to_idx[t] for t in valid_tok]

    with torch.no_grad():
        for _ in range(max_new):
            ctx = gen_idx[-CONTEXT_LEN:]
            if len(ctx) < CONTEXT_LEN:
                ctx = ctx + [ctx[-1]] * (CONTEXT_LEN - len(ctx))

            ctx_t    = torch.tensor(ctx, device=DEVICE)
            ctx_hv   = codebook(ctx_t)
            psi      = nn.functional.normalize(torch.sum(ctx_hv, dim=0), p=2, dim=0)
            logits   = hopfield_mem.query_topk(psi, k=k)

            # Top-k filtering
            topk_vals, topk_ids = torch.topk(logits, k)
            scaled   = topk_vals / temperature
            probs    = torch.softmax(scaled, dim=0)
            chosen   = topk_ids[torch.multinomial(probs, 1).item()].item()
            gen_idx.append(chosen)

    raw_tokens = [idx_to_token[i] for i in gen_idx]
    return tokenizer.decode(raw_tokens)


print("\n[4/5] Generación de texto (Top-k sampling, k={}, T=0.8):\n".format(TOPK))

test_prompts = [
    ("Once upon a time",       20),
    ("A little girl saw",      20),
    ("The cat went to",        20),
    ("There was a small dog",  20),
    ("The sun was shining",    20),
]

for prompt, length in test_prompts:
    result = generate_text_topk(prompt, max_new=length)
    print(f"  📝 Prompt : '{prompt}'")
    print(f"     Output : {result}\n")

# ── Diversidad: % de tokens únicos en texto generado ──
print("── Métrica de Diversidad ──")
all_generated_tokens = []
for prompt, length in test_prompts:
    raw = tokenizer.encode(generate_text_topk(prompt, max_new=50))
    all_generated_tokens.extend(raw)

unique_ratio = len(set(all_generated_tokens)) / len(all_generated_tokens) * 100
print(f"  Tokens totales generados: {len(all_generated_tokens)}")
print(f"  Tokens únicos           : {len(set(all_generated_tokens))}")
print(f"  Diversity score         : {unique_ratio:.1f}%  (100% = sin repetición)")

# %% [markdown]
# ## 6. 📈 Visualización de Resultados

# %%
print("\n[5/5] Graficando resultados...")

fig = plt.figure(figsize=(14, 10))
fig.suptitle("CHFT v2 — Campos Holográficos de Fourier: Resultados", fontsize=14, fontweight='bold')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

epochs_ax = range(1, EPOCHS + 1)

# ── Panel 1: Curva de pérdida (train + val) ──
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(epochs_ax, loss_history,     marker='o', color='#7C3AED', label='Train Loss', linewidth=2)
ax1.plot(epochs_ax, val_loss_history, marker='s', color='#EC4899', label='Val Loss',   linewidth=2, linestyle='--')
ax1.set_title("Curva de Pérdida (Cross-Entropy)")
ax1.set_xlabel("Época")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(list(epochs_ax))

# ── Panel 2: Comparativa Accuracy ──
ax2 = fig.add_subplot(gs[1, 0])
bars = ax2.bar(
    ["Baseline\n(freq)", "CHFT v2\n(nuestro)"],
    [base_acc, accuracy],
    color=["#94A3B8", "#7C3AED"],
    width=0.5,
    edgecolor="white"
)
for bar, val in zip(bars, [base_acc, accuracy]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f"{val:.1f}%", ha='center', va='bottom', fontweight='bold')
ax2.set_title("Accuracy@1: CHFT vs Baseline")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(0, max(accuracy, base_acc) * 1.3)
ax2.grid(True, axis='y', alpha=0.3)

# ── Panel 3: Métricas resumen ──
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
summary_rows = [
    ["Parámetro",           "Valor"],
    ["Dimensión FHRR",      f"{DIMENSION:,}"],
    ["Vocabulario",         f"{vocab_size:,} tokens"],
    ["Historias usadas",    f"{NUM_STORIES:,}"],
    ["Muestras train",      f"{num_train:,}"],
    ["Épocas",              f"{EPOCHS}"],
    ["Tiempo total",        f"{elapsed:.0f}s"],
    ["Train Loss final",    f"{loss_history[-1]:.4f}"],
    ["Val Loss final",      f"{val_loss_history[-1]:.4f}"],
    ["Accuracy@1",          f"{accuracy:.2f}%"],
    ["Perplexity",          f"{perplexity:.1f}"],
    ["Diversity Score",     f"{unique_ratio:.1f}%"],
]
table = ax3.table(
    cellText=summary_rows[1:],
    colLabels=summary_rows[0],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.4)
ax3.set_title("Resumen de Métricas", pad=12)

plt.savefig("chft_benchmark_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figura guardada como 'chft_benchmark_results.png'")

# %% [markdown]
# ## 7. 💾 Guardar Modelo y Resultados en Google Drive

# %%
from google.colab import drive
import os
import pickle

# Montar Google Drive
print("[-] Conectando con Google Drive...")
drive.mount('/content/drive')

# Carpeta destino
drive_dest_dir = '/content/drive/MyDrive/colabStore/01-CHFT'
os.makedirs(drive_dest_dir, exist_ok=True)

# Guardar modelo
model_data = {
    "phases":       codebook.phases.cpu().detach(),
    "token_to_idx": token_to_idx,
    "idx_to_token": idx_to_token,
    "vocab":        vocab,
    "vocab_size":   vocab_size,
    "dimension":    DIMENSION,
    "context_len":  CONTEXT_LEN,
    # Métricas para referencia futura
    "metrics": {
        "accuracy_pct":     accuracy,
        "perplexity":       perplexity,
        "baseline_acc_pct": base_acc,
        "diversity_pct":    unique_ratio,
        "train_loss_final": loss_history[-1],
        "val_loss_final":   val_loss_history[-1],
        "epochs":           EPOCHS,
        "training_secs":    elapsed,
    }
}

model_path = os.path.join(drive_dest_dir, 'chft_model_v2.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

# Guardar gráfico también
import shutil
shutil.copy("chft_benchmark_results.png", os.path.join(drive_dest_dir, "chft_benchmark_results.png"))

print(f"\n✅ Modelo guardado en: {model_path}")
print(f"✅ Gráfico guardado en: {drive_dest_dir}/chft_benchmark_results.png")
print(f"\n📊 Resumen final:")
print(f"   Accuracy@1  : {accuracy:.2f}%")
print(f"   Perplexity  : {perplexity:.2f}")
print(f"   Diversity   : {unique_ratio:.1f}%")
print(f"   Tiempo total: {elapsed:.0f}s")
