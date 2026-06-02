import os
import math
import time
import torch
import torch.nn as nn
import tiktoken
from datasets import load_dataset
from model import FHRRPhasorEmbedding, ModernHopfieldMemory


def prepare_dataset(
    num_stories: int, context_len: int, val_split: float, device: torch.device
):
    print("      Cargando TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{num_stories}]")
    tokenizer = tiktoken.get_encoding("cl100k_base")

    print("      Tokenizando y extrayendo vocabulario...")
    all_token_ids = set()
    stories_tokenized = []
    eot_token = tokenizer.eot_token
    all_token_ids.add(eot_token)

    for item in dataset:
        tokens = tokenizer.encode(item["text"])
        stories_tokenized.append(tokens)
        all_token_ids.update(tokens)

    vocab = sorted(list(all_token_ids))
    vocab_size = len(vocab)
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}
    idx_to_token = {i: tok for i, tok in enumerate(vocab)}

    print(f"      Vocabulario: {vocab_size:,} tokens únicos")

    # Concatenar todas las historias en una secuencia larga separadas por EOT
    flat_tokens = []
    for story in stories_tokenized:
        flat_tokens.extend(story)
        flat_tokens.append(eot_token)

    # Convertir a índices de vocabulario
    flat_indices = [token_to_idx[t] for t in flat_tokens]

    # Construir tensores de contexto/objetivo con ventana deslizante
    contexts_list, targets_list = [], []
    for i in range(len(flat_indices) - context_len):
        contexts_list.append(flat_indices[i : i + context_len])
        targets_list.append(flat_indices[i + context_len])

    num_total = len(contexts_list)
    num_val = int(num_total * val_split)
    num_train = num_total - num_val

    contexts_tensor = torch.tensor(contexts_list, dtype=torch.long)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)

    # Shuffle antes de split con semilla fija para consistencia al reanudar
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(num_total, generator=g)
    contexts_tensor = contexts_tensor[perm].to(device)
    targets_tensor = targets_tensor[perm].to(device)

    train_ctx, val_ctx = contexts_tensor[:num_train], contexts_tensor[num_train:]
    train_tgt, val_tgt = targets_tensor[:num_train], targets_tensor[num_train:]

    print(f"      Train: {num_train:,} muestras | Val: {num_val:,} muestras")

    # Guardar vocabulario en JSON para interact.py sin dependencias pesadas
    import json

    try:
        with open("vocab.json", "w") as f:
            json.dump(
                {
                    "token_to_idx": {int(k): int(v) for k, v in token_to_idx.items()},
                    "idx_to_token": {int(k): int(v) for k, v in idx_to_token.items()},
                },
                f,
            )
        print("      Vocabulario guardado en 'vocab.json' con éxito.")
    except Exception as e:
        print(f"      ⚠️ No se pudo guardar el vocabulario: {e}")

    return (
        train_ctx,
        train_tgt,
        val_ctx,
        val_tgt,
        token_to_idx,
        idx_to_token,
        vocab,
        vocab_size,
        tokenizer,
        targets_tensor,
    )


def run_training_loop(
    codebook: FHRRPhasorEmbedding,
    hopfield_mem: ModernHopfieldMemory,
    train_ctx: torch.Tensor,
    train_tgt: torch.Tensor,
    val_ctx: torch.Tensor,
    val_tgt: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_path: str = "chft_checkpoint.pth",
    reset_checkpoint: bool = False,
    patience: int = 2,
):
    print("\nIniciando Entrenamiento CHFT v5...")
    # Optimizar conjuntamente fases, pesos posicionales, MLP, proyecciones QK y beta
    optimizer = torch.optim.Adam(
        list(codebook.parameters()) + list(hopfield_mem.parameters()), lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    loss_history = []
    val_loss_history = []
    t0 = time.time()

    if reset_checkpoint and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print("  🗑️ Checkpoint anterior eliminado por solicitud de --reset.")
        except Exception as e:
            print(f"  ⚠️ No se pudo eliminar el checkpoint: {e}")

    if os.path.exists(checkpoint_path):
        print(
            f"  ⏳ Cargando punto de control de entrenamiento desde: {checkpoint_path}..."
        )
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            codebook.load_state_dict(checkpoint["codebook_state_dict"])
            hopfield_mem.load_state_dict(checkpoint["hopfield_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            loss_history = checkpoint.get("loss_history", [])
            val_loss_history = checkpoint.get("val_loss_history", [])
            print(
                f"  ✅ Checkpoint cargado con éxito. Reanudando desde la Época {start_epoch + 1:02d}..."
            )
        except Exception as e:
            print(
                f"  ⚠️ Advertencia al cargar checkpoint ({e}). Iniciando desde cero..."
            )
            start_epoch = 0
            loss_history = []
            val_loss_history = []

    num_train = len(train_ctx)
    num_val = len(val_ctx)

    for epoch in range(start_epoch, epochs):
        # ── TRAIN ──
        codebook.train()
        hopfield_mem.train()
        epoch_loss = 0.0
        perm = torch.randperm(num_train)
        num_batches = math.ceil(num_train / batch_size)

        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            ctx_b = train_ctx[idx]
            tgt_b = train_tgt[idx]

            # 1. Holographic Binding with Multi-Head Gates
            ctx_hv = codebook(ctx_b)  # [B, C, D]

            # 2. Multi-Scale Bundling
            psi = codebook.bundle_multiscale(ctx_hv)  # [B, D]

            # 3. Holographic FFN (Complex Non-Linear + Transform)
            psi = codebook.process_bundle(psi)  # [B, D]

            # 4. Multi-Hop Refinement with QK projections
            logits = hopfield_mem.refine_and_predict(psi, codebook, steps=2)

            # Optimización
            loss = nn.functional.cross_entropy(logits * hopfield_mem.beta, tgt_b)
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

        # ── VALIDACIÓN ──
        codebook.eval()
        hopfield_mem.eval()
        with torch.no_grad():
            val_batches = math.ceil(num_val / batch_size)
            val_loss_sum = 0.0

            for b in range(val_batches):
                ctx_v = val_ctx[b * batch_size : (b + 1) * batch_size]
                tgt_v = val_tgt[b * batch_size : (b + 1) * batch_size]

                ctx_hv = codebook(ctx_v)
                psi_v = codebook.bundle_multiscale(ctx_hv)
                psi_v = codebook.process_bundle(psi_v)

                logits_v = hopfield_mem.refine_and_predict(psi_v, codebook, steps=2)

                val_loss_sum += nn.functional.cross_entropy(
                    logits_v * hopfield_mem.beta, tgt_v
                ).item()

        avg_train = epoch_loss / num_batches
        avg_val = val_loss_sum / val_batches
        loss_history.append(avg_train)
        val_loss_history.append(avg_val)

        print(
            f"  Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Beta: {hopfield_mem.beta.item():.4f}"
        )

        try:
            checkpoint_data = {
                "epoch": epoch,
                "codebook_state_dict": codebook.state_dict(),
                "hopfield_state_dict": hopfield_mem.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_history": loss_history,
                "val_loss_history": val_loss_history,
            }
            torch.save(checkpoint_data, checkpoint_path)

            best_val_loss = min(val_loss_history)
            if avg_val <= best_val_loss:
                torch.save(checkpoint_data, "best_" + checkpoint_path)
                print(f"  🏆 Nuevo mejor modelo guardado (Val Loss: {avg_val:.4f})")
        except Exception as e:
            print(f"  ⚠️ Error al guardar checkpoint: {e}")

        best_val_loss = min(val_loss_history)
        best_epoch_idx = val_loss_history.index(best_val_loss)
        epochs_no_improve = len(val_loss_history) - 1 - best_epoch_idx
        if epochs_no_improve >= patience:
            print(
                f"\n  🛑 Early stopping activado: La pérdida de validación no ha mejorado durante {patience} épocas. Deteniendo entrenamiento."
            )
            break

    elapsed = time.time() - t0
    print(f"\n✅ Entrenamiento completado en {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return loss_history, val_loss_history, elapsed
