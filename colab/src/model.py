import math
import torch
import torch.nn as nn


class ComplexLayerNorm(nn.Module):
    """
    Normalización de amplitud en variables complejas.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(embedding_dim, eps=eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
        phase = torch.angle(z)
        mag_norm = self.ln(mag)
        return torch.complex(mag_norm * torch.cos(phase), mag_norm * torch.sin(phase))


class HolographicMLP(nn.Module):
    """
    Feed-Forward Network (H-FFN) para representaciones complejas.
    Trata las partes real e imaginaria como características conjuntas.
    """

    def __init__(self, embedding_dim: int, expansion_factor: int = 4):
        super().__init__()
        in_features = embedding_dim * 2
        hidden_features = in_features * expansion_factor

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

        # Inicialización tipo Transformer para residuales
        nn.init.normal_(self.fc2.weight, std=0.02 / math.sqrt(2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z.real, z.imag], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        r, i = torch.chunk(x, 2, dim=-1)
        return torch.complex(r, i)


class UnitaryHouseholderProjection(nn.Module):
    """
    Capas de proyección complejas estrictamente unitarias mediante Householder reflections.
    Conserva magnitudes y ángulos de fase, eliminando distorsiones.
    """

    def __init__(self, dim: int, K: int = 8):
        super().__init__()
        self.dim = dim
        self.K = K
        self.v_real = nn.Parameter(torch.empty(K, dim).normal_(mean=0.0, std=0.02))
        self.v_imag = nn.Parameter(torch.empty(K, dim).normal_(mean=0.0, std=0.02))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D] complex
        orig_shape = x.shape
        x_flat = x.view(-1, self.dim)

        v = torch.complex(self.v_real, self.v_imag)
        for k in range(self.K):
            vk = v[k]
            sq_norm = torch.sum(torch.abs(vk) ** 2) + 1e-8
            proj = torch.sum(x_flat * torch.conj(vk), dim=-1, keepdim=True)
            x_flat = x_flat - 2.0 * proj * vk / sq_norm

        return x_flat.view(orig_shape)


class FHRRPhasorEmbedding(nn.Module):
    """
    Fourier Holographic Reduced Representation con Multi-Head Gating y H-FFN.
    v11: Spreading Activation Energy replaces flat pos_weights.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, context_len: int, n_head: int = 8
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_len = context_len
        self.n_head = n_head
        assert embedding_dim % n_head == 0, (
            "embedding_dim debe ser divisible por n_head"
        )
        self.head_dim = embedding_dim // n_head

        self.phases = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim).uniform_(-math.pi, math.pi)
        )

        g = torch.Generator().manual_seed(42)
        omega = torch.empty(embedding_dim).uniform_(-math.pi, math.pi, generator=g)
        self.register_buffer("omega", omega)

        pos_angles = torch.arange(context_len).unsqueeze(1) * omega.unsqueeze(0)
        self.register_buffer("pos_rotation_real", torch.cos(pos_angles))
        self.register_buffer("pos_rotation_imag", torch.sin(pos_angles))

        self.ln_1 = ComplexLayerNorm(embedding_dim)
        self.ln_2 = ComplexLayerNorm(embedding_dim)

        # Multi-Head Saliency Gating
        self.gate_proj = nn.Linear(embedding_dim, n_head)
        nn.init.constant_(self.gate_proj.bias, 1.5)
        nn.init.normal_(self.gate_proj.weight, std=0.02)

        self.scale_short = nn.Parameter(torch.tensor(1.0))
        self.scale_long = nn.Parameter(torch.tensor(0.5))

        # H-FFN
        self.mlp = HolographicMLP(embedding_dim, expansion_factor=2)

        # Proyección complex unitaria Q (Householder - v15)
        self.q_proj = UnitaryHouseholderProjection(embedding_dim, K=8)

        # Parámetros SHA (Spectral Holographic Attention) — v9
        # beta_sha: temperatura de softmax para la atención espectral
        self.sha_beta = nn.Parameter(torch.tensor(8.0))
        # alpha_sha: escala del vector atendido al fusionar con query
        self.sha_alpha = nn.Parameter(torch.tensor(1.0))
        # recency_lambda: decaimiento ALiBi-style sobre distancias posicionales
        self.sha_recency_lambda = nn.Parameter(torch.tensor(0.05))
        # gate_sha: compuerta dual-path SHA vs multi-escala, inicializada a 0.5
        self.gate_sha = nn.Parameter(torch.tensor(0.5))

        # ── Spreading Activation Energy Weights (v11) ──────────────────────────
        # E(p) = w_d*alpha^d  +  w_r*S(r)  +  w_f*(1-exp(-k*freq))  +  w_rec*exp(-lam*d)
        # d = distance from the query token (0 = newest position, C-1 = oldest)
        # S(r) = relational strength supplied by the existing multi-head gate
        # freq = normalised token-occurrence frequency (precomputed buffer, range [0,1])
        self.sa_w_d = nn.Parameter(torch.tensor(0.25))  # depth-decay coefficient
        self.sa_alpha = nn.Parameter(torch.tensor(0.85))  # decay base (init = 0.85)
        self.sa_w_r = nn.Parameter(torch.tensor(0.25))  # relational-strength weight
        self.sa_w_f = nn.Parameter(torch.tensor(0.25))  # frequency-reward weight
        self.sa_k = nn.Parameter(torch.tensor(3.0))  # frequency saturation rate
        self.sa_w_rec = nn.Parameter(torch.tensor(0.25))  # recency-bonus weight
        self.sa_lam = nn.Parameter(torch.tensor(0.05))  # recency decay rate
        # token frequency buffer — set by train.py; default: uniform
        self.register_buffer(
            "token_freq",
            torch.zeros(num_embeddings),  # filled during dataset prep
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        phi = self.phases[indices]
        x_real = torch.cos(phi)
        x_imag = torch.sin(phi)

        pos_rot_real = self.pos_rotation_real
        pos_rot_imag = self.pos_rotation_imag

        bound_real = x_real * pos_rot_real - x_imag * pos_rot_imag
        bound_imag = x_real * pos_rot_imag + x_imag * pos_rot_real

        # ── Multi-Head Saliency Gate (S(r) component) ──────────────────────────
        gates = torch.sigmoid(self.gate_proj(phi))  # [..., C, n_head]
        # Mean gate per position → scalar relational strength S(r)  [..., C]
        gate_strength = gates.mean(dim=-1)  # [..., C]

        # ── Spreading Activation Energy (v11) ──────────────────────────────────
        # d = distance from query: position C-1 is query (d=0), position 0 is oldest (d=C-1)
        C = indices.shape[-1]  # context length
        d = torch.arange(C, device=indices.device).float().flip(0)  # [C], newest=0

        # Depth decay term: w_d * alpha^d
        alpha_clamped = torch.clamp(torch.abs(self.sa_alpha), 0.5, 0.99)
        decay_term = self.sa_w_d * (alpha_clamped**d)  # [C]

        # Relational strength term: w_r * S(r)  (gate mean already in [0,1])
        strength_term = self.sa_w_r * gate_strength  # [..., C]

        # Frequency reward term: w_f * (1 - exp(-k * freq))
        freq = self.token_freq[indices]  # [..., C]
        k_pos = torch.clamp(self.sa_k, min=0.1)
        freq_term = self.sa_w_f * (1.0 - torch.exp(-k_pos * freq))  # [..., C]

        # Recency bonus term: w_rec * exp(-lambda * d)
        lam_pos = torch.clamp(torch.abs(self.sa_lam), min=1e-4)
        recency_term = self.sa_w_rec * torch.exp(-lam_pos * d)  # [C]

        # Combined energy weight per position (all components positive)
        W = decay_term + strength_term + freq_term + recency_term  # [..., C]
        W = W.unsqueeze(-1)  # [..., C, 1] — broadcast over D

        # Apply full gates (head expansion) and energy weights
        gates_exp = gates.unsqueeze(-1).expand(*gates.shape, self.head_dim)
        gates_exp = gates_exp.reshape(*phi.shape[:-1], self.embedding_dim)

        bound_real = bound_real * W * gates_exp
        bound_imag = bound_imag * W * gates_exp

        return torch.complex(bound_real, bound_imag)

    def all_keys(self) -> torch.Tensor:
        return torch.complex(torch.cos(self.phases), torch.sin(self.phases))

    def bundle_multiscale(
        self, ctx_hv: torch.Tensor, short_len: int = 16
    ) -> torch.Tensor:
        is_batched = len(ctx_hv.shape) == 3
        dim_reduce = 1 if is_batched else 0
        C = ctx_hv.shape[dim_reduce]

        if C <= short_len:
            return torch.sum(ctx_hv, dim=dim_reduce)

        if is_batched:
            short_part = ctx_hv[:, -short_len:, :]
            long_part = ctx_hv[:, :-short_len, :]
        else:
            short_part = ctx_hv[-short_len:, :]
            long_part = ctx_hv[:-short_len, :]

        psi_short = torch.sum(short_part, dim=dim_reduce)
        psi_long = torch.sum(long_part, dim=dim_reduce)

        return self.scale_short * psi_short + self.scale_long * psi_long

    def sha_bundling(self, ctx_hv: torch.Tensor) -> torch.Tensor:
        """
        SHA — Spectral Holographic Attention (CHFT v9).

        Computes attention DIRECTLY over individual token hypervectors H[B, C-1, D]
        using their native phase cosine similarity with the query q[B, D].

        This avoids the cross-talk problem of HPRA, which incorrectly computed
        attention using the lossy superposition bundle as key, injecting O(C*sqrt(D))
        noise from all tokens into every attention score.

        ctx_hv : [B, C, D] complex  (batched)  OR  [C, D] complex  (single)
        returns : [B, D] complex  (bundled context vector)
        """
        is_batched = len(ctx_hv.shape) == 3
        if not is_batched:
            ctx_hv = ctx_hv.unsqueeze(0)  # → [1, C, D]

        B, C, D = ctx_hv.shape

        # Edge case: only one token in context
        if C <= 1:
            out = ctx_hv[:, -1, :]
            if not is_batched:
                out = out.squeeze(0)
            return out

        # 1. Split: query = last token, history = all prior tokens
        q = ctx_hv[:, -1, :]  # [B, D] complex
        H = ctx_hv[:, :-1, :]  # [B, C-1, D] complex

        # 2. Phase cosine similarity: Re(<H_p, q>) / sqrt(D)
        #    = Re( H[B, C-1, D] * conj(q)[B, 1, D] ).sum(-1)
        #    This is the correct holographic inner product — computed over
        #    INDIVIDUAL token vectors, not the superposition bundle.
        q_conj = torch.conj(q).unsqueeze(1)  # [B, 1, D]
        scores = torch.real(torch.sum(H * q_conj, dim=-1)) / math.sqrt(D)  # [B, C-1]

        # 3. ALiBi-style recency bias: recent tokens get a bonus
        #    distances[0] = furthest token, distances[C-2] = most recent
        distances = torch.arange(C - 1, device=ctx_hv.device).flip(0).float()  # [C-1]
        pos_bias = -torch.abs(self.sha_recency_lambda) * distances.unsqueeze(
            0
        )  # [1, C-1]
        scores = scores + pos_bias  # [B, C-1]

        # 4. Softmax temperature-scaled attention weights
        attn_weights = torch.softmax(scores * self.sha_beta, dim=-1)  # [B, C-1]

        # 5. Weighted holographic superposition (no unbinding needed — clean signal)
        psi_attended = torch.sum(attn_weights.unsqueeze(-1) * H, dim=1)  # [B, D]

        # 6. Residual fusion with query
        psi_final = q + self.sha_alpha * psi_attended  # [B, D]

        if not is_batched:
            psi_final = psi_final.squeeze(0)
        return psi_final

    def dual_path_bundling(self, ctx_hv: torch.Tensor) -> torch.Tensor:
        """
        Dual-Path Bundling: fuses SHA (spectral attention) with the proven
        multi-scale temporal bundling via a learnable gate.

        - gate_sha → 1.0: pure SHA (full attention)
        - gate_sha → 0.0: pure multi-scale (champion v6 behaviour)

        The gate learns the optimal blend during training.
        """
        gate = torch.sigmoid(self.gate_sha)  # scalar in (0, 1)
        psi_sha = self.sha_bundling(ctx_hv)
        psi_ms = self.bundle_multiscale(ctx_hv)
        return gate * psi_sha + (1.0 - gate) * psi_ms

    def process_bundle(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Bloque Transformer Equivalente para el Vector Holográfico (H-FFN + Atención Lineal).
        """
        # 1. Proyección complex unitaria Q (Householder)
        psi = self.q_proj(psi)

        # 2. H-FFN block with LayerNorm (Pre-LN style) - SOFT RESIDUAL
        psi_norm = self.ln_1(psi)
        psi = psi + 0.15 * self.mlp(psi_norm)

        return self.ln_2(psi)


class ModernHopfieldMemory(nn.Module):
    """
    Recuperación de atractores nativa por interferencia geométrica pura con CGRA (Complex Gated Recurrent Attractor).
    """

    def __init__(self, embedding_dim: int, beta: float = 16.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.embedding_dim = embedding_dim
        # ── Adaptive Beta (Memory Decay / v11) ─────────────────────────────────
        # beta_eff = beta_base * sigmoid(beta_conf_scale * max_sim)
        # When retrieval similarity is high (confident) → sharpen focus.
        # When noisy (low similarity) → soften to avoid false attractors.
        self.beta_conf_scale = nn.Parameter(torch.tensor(4.0))

        # Configuración Multi-Head Subspace
        self.n_head = 8
        self.head_dim = embedding_dim // self.n_head
        assert embedding_dim % self.n_head == 0, (
            "embedding_dim debe ser divisible por n_head"
        )

        # Parámetros del CGRA por subespacio/cabeza (n_head, head_dim, 4, 2)
        std = 1.0 / math.sqrt(4)
        self.W_z = nn.Parameter(
            torch.empty(self.n_head, self.head_dim, 4, 2).normal_(mean=0.0, std=std)
        )
        self.b_z = nn.Parameter(torch.zeros(self.n_head, self.head_dim, 2))

        self.W_r = nn.Parameter(
            torch.empty(self.n_head, self.head_dim, 4, 2).normal_(mean=0.0, std=std)
        )
        self.b_r = nn.Parameter(torch.zeros(self.n_head, self.head_dim, 2))

        self.W_c = nn.Parameter(
            torch.empty(self.n_head, self.head_dim, 4, 2).normal_(mean=0.0, std=std)
        )
        self.b_c = nn.Parameter(torch.zeros(self.n_head, self.head_dim, 2))

        # Proyección complex unitaria para mezcla de cabezas (Householder)
        self.mix_proj = UnitaryHouseholderProjection(embedding_dim, K=8)

    def cgra_update(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # h: [B, D] complex o [D] complex
        # x: [B, D] complex o [D] complex
        is_1d = h.ndim == 1
        if is_1d:
            h = h.unsqueeze(0)
            x = x.unsqueeze(0)

        B = h.shape[0]

        # 1. Redimensionar a cabezas: [B, n_head, head_dim]
        h_heads = h.view(B, self.n_head, self.head_dim)
        x_heads = x.view(B, self.n_head, self.head_dim)

        # 2. Apilar partes reales e imaginarias: [B, n_head, head_dim, 4]
        stacked_inputs = torch.stack(
            [h_heads.real, h_heads.imag, x_heads.real, x_heads.imag], dim=-1
        )

        # 3. Computar compuertas de actualización (z) y reset (r) en paralelo
        z = torch.sigmoid(
            torch.einsum("bndf,ndfg->bndg", stacked_inputs, self.W_z) + self.b_z
        )  # [B, n_head, head_dim, 2]
        r = torch.sigmoid(
            torch.einsum("bndf,ndfg->bndg", stacked_inputs, self.W_r) + self.b_r
        )  # [B, n_head, head_dim, 2]

        # 4. Aplicar compuerta de reset al estado anterior de cada cabeza
        h_reset_real = r[..., 0] * h_heads.real
        h_reset_imag = r[..., 1] * h_heads.imag

        # 5. Computar estado candidato (tilde_h)
        I_c = torch.stack(
            [h_reset_real, h_reset_imag, x_heads.real, x_heads.imag], dim=-1
        )
        h_tilde = torch.tanh(
            torch.einsum("bndf,ndfg->bndg", I_c, self.W_c) + self.b_c
        )  # [B, n_head, head_dim, 2]

        # 6. Actualización recurrente compuerta por compuerta
        h_new_real = (1.0 - z[..., 0]) * h_heads.real + z[..., 0] * h_tilde[..., 0]
        h_new_imag = (1.0 - z[..., 1]) * h_heads.imag + z[..., 1] * h_tilde[..., 1]

        # 7. Reconstruir a tensor plano [B, D]
        out_heads = torch.complex(h_new_real, h_new_imag)
        out = out_heads.view(B, self.embedding_dim)

        if is_1d:
            out = out.squeeze(0)
        return out

    def refine_and_predict(
        self, psi: torch.Tensor, codebook: FHRRPhasorEmbedding, steps: int = 2
    ) -> torch.Tensor:
        """
        Realiza el multi-hop routing y computa los logits directamente usando cosenos nativos.
        psi: [B, D] state
        """
        D = self.embedding_dim
        keys_all = codebook.all_keys()

        state = psi
        # Refinamiento multi-hop con beta adaptativo (v11) y mezcla de cabezas (v14)
        for _ in range(steps):
            sim = torch.real(torch.matmul(state, torch.conj(keys_all).t())) / math.sqrt(
                D
            )

            # Adaptive Beta: scale sharpness by retrieval confidence
            # max_sim ∈ [0, 1] for unit phasors — high = confident match
            max_sim = sim.max(dim=-1, keepdim=True).values  # [B, 1]
            beta_eff = self.beta * torch.sigmoid(self.beta_conf_scale * max_sim)

            weights = torch.softmax(sim * beta_eff, dim=-1)
            retrieved = torch.matmul(weights.to(keys_all.dtype), keys_all)

            # CGRA actualiza el estado recurrentemente en cada subespacio de cabeza
            state = self.cgra_update(state, retrieved)

            # Proyección complex unitaria para mezcla de cabezas (Householder)
            state = self.mix_proj(state)

            state = codebook.ln_2(state) / math.sqrt(D)

        # Logits finales usando fases nativas puras
        state_r = state.real
        state_i = state.imag
        keys_r = keys_all.real
        keys_i = keys_all.imag
        logits = (
            torch.matmul(state_r, keys_r.t()) + torch.matmul(state_i, keys_i.t())
        ) / math.sqrt(D)

        return logits

    def update_keys(self, codebook: FHRRPhasorEmbedding):
        pass
