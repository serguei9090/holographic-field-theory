import os
import math
import time
import torch
import torch.nn as nn
import tiktoken
from datasets import load_dataset
from model import FHRRPhasorEmbedding, ModernHopfieldMemory

def prepare_dataset(num_stories: int, context_len: int, val_split: float, device: torch.device):
    print("      Cargando TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{num_stories}]")
    tokenizer = tiktoken.get_encoding("cl100k_base")

    print("      Tokenizando y extrayendo vocabulario...")
    all_token_ids = set()
    stories_tokenized = []

    for item in dataset:
        tokens = tokenizer.encode(item["text"])
        stories_tokenized.append(tokens)
        all_token_ids.update(tokens)

    vocab = list(all_token_ids)
    vocab_size = len(vocab)
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}
    idx_to_token = {i: tok for i, tok in enumerate(vocab)}

    print(f"      Vocabulario: {vocab_size:,} tokens únicos")

    # Construir tensores de contexto/objetivo
    contexts_list, targets_list = [], []
    for story in stories_tokenized:
        if len(story) < context_len + 1:
            continue
        story_idx = [token_to_idx[t] for t in story]
        for i in range(len(story_idx) - context_len):
            contexts_list.append(story_idx[i : i + context_len])
            targets_list.append(story_idx[i + context_len])

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
            json.dump({
                "token_to_idx": {int(k): int(v) for k, v in token_to_idx.items()},
                "idx_to_token": {int(k): int(v) for k, v in idx_to_token.items()}
            }, f)
        print("      Vocabulario guardado en 'vocab.json' con éxito.")
    except Exception as e:
        print(f"      ⚠️ No se pudo guardar el vocabulario: {e}")
    
    return (
        train_ctx, train_tgt, 
        val_ctx, val_tgt, 
        token_to_idx, idx_to_token, 
        vocab, vocab_size, 
        tokenizer, targets_tensor
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
    patience: int = 2
):
    print("\nIniciando Entrenamiento CHFT...")
    # Optimizar conjuntamente fases, pesos posicionales y el parámetro beta
    optimizer = torch.optim.Adam(
        list(codebook.parameters()) + list(hopfield_mem.parameters()), 
        lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    loss_history = []
    val_loss_history = []
    t0 = time.time()
    
    # Manejar reinicio forzado del checkpoint
    if reset_checkpoint and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"  🗑️ Checkpoint anterior eliminado por solicitud de --reset.")
        except Exception as e:
            print(f"  ⚠️ No se pudo eliminar el checkpoint: {e}")

    # Intentar cargar checkpoint existente
    if os.path.exists(checkpoint_path):
        print(f"  ⏳ Cargando punto de control de entrenamiento desde: {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Cargar pesos
            codebook.load_state_dict(checkpoint['codebook_state_dict'])
            hopfield_mem.load_state_dict(checkpoint['hopfield_state_dict'])
            # Cargar optimizador y planificador
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            loss_history = checkpoint.get('loss_history', [])
            val_loss_history = checkpoint.get('val_loss_history', [])
            print(f"  ✅ Checkpoint cargado con éxito. Reanudando desde la Época {start_epoch + 1:02d}...")
        except Exception as e:
            print(f"  ⚠️ Advertencia al cargar checkpoint ({e}). Iniciando desde cero...")
            start_epoch = 0
            loss_history = []
            val_loss_history = []

    num_train = len(train_ctx)
    num_val = len(val_ctx)
    D = codebook.phases.shape[1]

    # Actualizar la memoria de Hopfield con las llaves iniciales del codebook cargado
    hopfield_mem.update_keys(codebook)

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

            # Superposición holográfica posicional (Binding interno + weights)
            ctx_hv = codebook(ctx_b)                          # [B, C, D]
            psi = torch.sum(ctx_hv, dim=1)                 # [B, D]
            
            # Normalización compleja estandarizada usando Complex LayerNorm
            psi = codebook.ln(psi) / math.sqrt(D)

            # Multi-Hop Refinement diferenciable en entrenamiento (2 pasos)
            keys_all = codebook.all_keys()
            for _ in range(2):
                sim_ref = torch.real(torch.matmul(psi, torch.conj(keys_all).t())) / math.sqrt(D)
                weights_ref = torch.softmax(sim_ref * hopfield_mem.beta, dim=-1)
                retrieved = torch.matmul(weights_ref.to(keys_all.dtype), keys_all)
                psi = psi + retrieved
                psi = codebook.ln(psi) / math.sqrt(D)

            # Proyección final para obtener logits
            psi_r = psi.real
            psi_i = psi.imag
            keys_r = keys_all.real
            keys_i = keys_all.imag
            logits = (torch.matmul(psi_r, keys_r.t()) + torch.matmul(psi_i, keys_i.t())) / math.sqrt(D)

            # Usar la escala beta entrenable
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
        hopfield_mem.update_keys(codebook)

        # ── VALIDACIÓN ──
        codebook.eval()
        hopfield_mem.eval()
        with torch.no_grad():
            val_batches = math.ceil(num_val / batch_size)
            val_loss_sum = 0.0
            keys_all = codebook.all_keys()
            
            keys_r = keys_all.real
            keys_i = keys_all.imag
            for b in range(val_batches):
                ctx_v = val_ctx[b * batch_size : (b + 1) * batch_size]
                tgt_v = val_tgt[b * batch_size : (b + 1) * batch_size]
                
                ctx_hv = codebook(ctx_v)
                psi_v = torch.sum(ctx_hv, dim=1)
                
                psi_v = codebook.ln(psi_v) / math.sqrt(D)
                
                # Refinamiento en validación (2 pasos)
                for _ in range(2):
                    sim_ref = torch.real(torch.matmul(psi_v, torch.conj(keys_all).t())) / math.sqrt(D)
                    weights_ref = torch.softmax(sim_ref * hopfield_mem.beta, dim=-1)
                    retrieved = torch.matmul(weights_ref.to(keys_all.dtype), keys_all)
                    psi_v = psi_v + retrieved
                    psi_v = codebook.ln(psi_v) / math.sqrt(D)
                
                psi_vr = psi_v.real
                psi_vi = psi_v.imag
                
                logits_v = (torch.matmul(psi_vr, keys_r.t()) + torch.matmul(psi_vi, keys_i.t())) / math.sqrt(D)
                val_loss_sum += nn.functional.cross_entropy(logits_v * hopfield_mem.beta, tgt_v).item()

        avg_train = epoch_loss / num_batches
        avg_val = val_loss_sum / val_batches
        loss_history.append(avg_train)
        val_loss_history.append(avg_val)

        print(f"  Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Beta: {hopfield_mem.beta.item():.4f}")

        # Guardar checkpoint
        try:
            checkpoint_data = {
                'epoch': epoch,
                'codebook_state_dict': codebook.state_dict(),
                'hopfield_state_dict': hopfield_mem.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_history': loss_history,
                'val_loss_history': val_loss_history
            }
            torch.save(checkpoint_data, checkpoint_path)
            
            # Guardar el mejor checkpoint si la pérdida de validación mejoró
            best_val_loss = min(val_loss_history)
            if avg_val <= best_val_loss:
                torch.save(checkpoint_data, "best_" + checkpoint_path)
                print(f"  🏆 Nuevo mejor modelo guardado (Val Loss: {avg_val:.4f})")
        except Exception as e:
            print(f"  ⚠️ Error al guardar checkpoint: {e}")

        # Verificar Early Stopping (Parada Temprana)
        best_val_loss = min(val_loss_history)
        best_epoch_idx = val_loss_history.index(best_val_loss)
        epochs_no_improve = len(val_loss_history) - 1 - best_epoch_idx
        if epochs_no_improve >= patience:
            print(f"\n  🛑 Early stopping activado: La pérdida de validación no ha mejorado durante {patience} épocas. Deteniendo entrenamiento.")
            break

    elapsed = time.time() - t0
    print(f"\n✅ Entrenamiento completado en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return loss_history, val_loss_history, elapsed


