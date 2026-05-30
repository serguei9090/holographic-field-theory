import math
import collections
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def evaluate_model(
    codebook: nn.Module,
    val_ctx: torch.Tensor,
    val_tgt: torch.Tensor,
    batch_size: int,
    vocab_size: int,
    targets_tensor: torch.Tensor,
    val_split: float
):
    print("\nCalculando métricas de benchmark...")
    codebook.eval()

    correct = 0
    total_ce = 0.0
    n_val = 0
    num_val = len(val_ctx)

    with torch.no_grad():
        keys_r = torch.cos(codebook.phases)
        keys_i = torch.sin(codebook.phases)
        # Precomputar rotación posicional para evaluación
        context_len = val_ctx.shape[1]
        pos_angles = torch.arange(context_len, device=val_ctx.device).unsqueeze(1) * (math.pi / context_len)
        pos_rotation = torch.complex(torch.cos(pos_angles), torch.sin(pos_angles)) # [C, D]

        for b in range(math.ceil(num_val / batch_size)):
            ctx_v = val_ctx[b * batch_size : (b + 1) * batch_size]
            tgt_v = val_tgt[b * batch_size : (b + 1) * batch_size]

            ctx_hv = codebook(ctx_v) * pos_rotation
            psi_v = torch.sum(ctx_hv, dim=1)
            
            norm_v = torch.clamp(torch.norm(psi_v, p=2, dim=1, keepdim=True), min=1e-12)
            psi_vr = psi_v.real / norm_v
            psi_vi = psi_v.imag / norm_v
            
            # Dividir por sqrt(D) para normalizar similitud
            logits_v = (torch.matmul(psi_vr, keys_r.t()) + torch.matmul(psi_vi, keys_i.t())) / math.sqrt(codebook.phases.shape[1])

            preds = torch.argmax(logits_v, dim=1)
            correct += (preds == tgt_v).sum().item()
            total_ce += nn.functional.cross_entropy(logits_v * 16.0, tgt_v, reduction='sum').item()
            n_val += len(tgt_v)

    accuracy = correct / n_val * 100
    perplexity = math.exp(total_ce / n_val)

    print(f"  Accuracy@1  : {accuracy:.2f}%   (tokens exactos predichos)")
    print(f"  Perplexity  : {perplexity:.2f}  (menor = mejor; azar ≈ {vocab_size:,})")

    # Baseline
    token_freq = collections.Counter(targets_tensor.cpu().tolist())
    most_common = token_freq.most_common(1)[0][0]
    base_correct = sum(1 for t in val_tgt.cpu().tolist() if t == most_common)
    base_acc = base_correct / num_val * 100
    print(f"\n  [Baseline freq] Accuracy@1: {base_acc:.2f}%  (siempre predice token más frecuente)")
    print(f"  [CHFT v2]       Accuracy@1: {accuracy:.2f}%  (+{accuracy - base_acc:.2f}pp vs baseline)")

    return accuracy, perplexity, base_acc

def plot_and_save_results(
    epochs: int,
    loss_history: list[float],
    val_loss_history: list[float],
    base_acc: float,
    accuracy: float,
    dimension: int,
    vocab_size: int,
    num_stories: int,
    num_train: int,
    elapsed: float,
    unique_ratio: float,
    perplexity: float,
    filename: str = "chft_benchmark_results.png"
):
    print("\nGraficando resultados...")
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("CHFT v2 — Campos Holográficos de Fourier: Resultados", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    epochs_ax = range(1, epochs + 1)

    # ── Panel 1: Curva de pérdida ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs_ax, loss_history, marker='o', color='#7C3AED', label='Train Loss', linewidth=2)
    ax1.plot(epochs_ax, val_loss_history, marker='s', color='#EC4899', label='Val Loss', linewidth=2, linestyle='--')
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
        ["Dimensión FHRR",      f"{dimension:,}"],
        ["Vocabulario",         f"{vocab_size:,} tokens"],
        ["Historias usadas",    f"{num_stories:,}"],
        ["Muestras train",      f"{num_train:,}"],
        ["Épocas",              f"{epochs}"],
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

    plt.show()
    print("✅ Figura mostrada en pantalla.")
