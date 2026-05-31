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

    # Shuffle antes de split
    perm = torch.randperm(num_total)
    contexts_tensor = contexts_tensor[perm].to(device)
    targets_tensor = targets_tensor[perm].to(device)

    train_ctx, val_ctx = contexts_tensor[:num_train], contexts_tensor[num_train:]
    train_tgt, val_tgt = targets_tensor[:num_train], targets_tensor[num_train:]

    print(f"      Train: {num_train:,} muestras | Val: {num_val:,} muestras")
    
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
    device: torch.device
):
    print("\nIniciando Entrenamiento CHFT...")
    # Optimizar conjuntamente fases, pesos posicionales y el parámetro beta
    optimizer = torch.optim.Adam(
        list(codebook.parameters()) + list(hopfield_mem.parameters()), 
        lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    val_loss_history = []
    t0 = time.time()
    
    num_train = len(train_ctx)
    num_val = len(val_ctx)
    D = codebook.phases.shape[1]

    for epoch in range(epochs):
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

            # Superposición holográfica posicional (Binding interno + Learned weights)
            ctx_hv = codebook(ctx_b)                          # [B, C, D] complejo posicional
            psi = torch.sum(ctx_hv, dim=1)                 # [B, D] complejo
            
            # Normalización estandarizada usando Complex LayerNorm
            psi = codebook.ln(psi) / math.sqrt(D)

            # Multi-Hop Refinement diferenciable en entrenamiento (2 pasos de refinación)
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

    elapsed = time.time() - t0
    print(f"\n✅ Entrenamiento completado en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return loss_history, val_loss_history, elapsed

