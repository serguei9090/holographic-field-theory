import math
import torch
import torch.nn as nn
from model import FHRRPhasorEmbedding, ModernHopfieldMemory

def generate_text_topk(
    prompt: str,
    codebook: FHRRPhasorEmbedding,
    hopfield_mem: ModernHopfieldMemory,
    tokenizer,
    token_to_idx: dict[int, int],
    idx_to_token: dict[int, int],
    context_len: int,
    device: torch.device,
    max_new: int = 20,
    k: int = 5,
    temperature: float = 0.8
) -> str:
    """
    Genera texto usando top-k sampling con temperatura para diversidad.
    """
    codebook.eval()
    tokens_in = tokenizer.encode(prompt)
    valid_tok = [t for t in tokens_in if t in token_to_idx]
    if not valid_tok:
        return "[tokens del prompt fuera del vocabulario]"

    gen_idx = [token_to_idx[t] for t in valid_tok]

    with torch.no_grad():
        # Precomputar rotación posicional para generación con decay
        D_gen = codebook.phases.shape[1]
        g_gen = torch.Generator(device=device).manual_seed(42)
        omega_gen = torch.empty(D_gen, device=device).uniform_(-math.pi, math.pi, generator=g_gen)
        
        pos_angles = torch.arange(context_len, device=device).unsqueeze(1) * omega_gen.unsqueeze(0)
        pos_rotation = torch.complex(torch.cos(pos_angles), torch.sin(pos_angles)) # [C, D]
        
        decay_gen = (0.85 ** torch.arange(context_len, device=device).flip(0)).unsqueeze(1)
        pos_rotation = pos_rotation * decay_gen

        for _ in range(max_new):
            ctx = gen_idx[-context_len:]
            if len(ctx) < context_len:
                ctx = ctx + [ctx[-1]] * (context_len - len(ctx))

            ctx_t = torch.tensor(ctx, device=device)
            ctx_hv = codebook(ctx_t) * pos_rotation
            psi = nn.functional.normalize(torch.sum(ctx_hv, dim=0), p=2, dim=0)
            logits = hopfield_mem.query_topk(psi, k=k)
            # Clonar logits para aplicar la penalización sin modificar el objeto original
            logits = logits.clone()
            
            # Penalizar tokens que ya están en el contexto para romper bucles repetitivos
            for tok_idx in set(ctx):
                logits[tok_idx] -= 35.0  # Penalización proporcional a la escala de similitud (~64.0 máx)

            # Top-k filtering
            topk_vals, topk_ids = torch.topk(logits, k)
            scaled = topk_vals / temperature
            probs = torch.softmax(scaled, dim=0)
            chosen = topk_ids[torch.multinomial(probs, 1).item()].item()
            gen_idx.append(chosen)

    raw_tokens = [idx_to_token[i] for i in gen_idx]
    return tokenizer.decode(raw_tokens)

def calculate_diversity(
    prompts: list[tuple[str, int]],
    codebook: FHRRPhasorEmbedding,
    hopfield_mem: ModernHopfieldMemory,
    tokenizer,
    token_to_idx: dict[int, int],
    idx_to_token: dict[int, int],
    context_len: int,
    device: torch.device,
    k: int = 5,
    temperature: float = 0.8
) -> float:
    print("── Métrica de Diversidad ──")
    all_generated_tokens = []
    for prompt, length in prompts:
        generated_text = generate_text_topk(
            prompt, codebook, hopfield_mem, tokenizer,
            token_to_idx, idx_to_token, context_len, device,
            max_new=50, k=k, temperature=temperature
        )
        raw = tokenizer.encode(generated_text)
        all_generated_tokens.extend(raw)

    if not all_generated_tokens:
        return 0.0

    unique_ratio = len(set(all_generated_tokens)) / len(all_generated_tokens) * 100
    print(f"  Tokens totales generados: {len(all_generated_tokens)}")
    print(f"  Tokens únicos           : {len(set(all_generated_tokens))}")
    print(f"  Diversity score         : {unique_ratio:.1f}%  (100% = sin repetición)")
    return unique_ratio
