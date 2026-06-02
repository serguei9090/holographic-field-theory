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


class FHRRPhasorEmbedding(nn.Module):
    """
    Fourier Holographic Reduced Representation con Multi-Head Gating y H-FFN.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, context_len: int, n_head: int = 8
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        assert embedding_dim % n_head == 0, (
            "embedding_dim debe ser divisible por n_head"
        )
        self.head_dim = embedding_dim // n_head

        self.phases = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim).uniform_(-math.pi, math.pi)
        )

        init_weights = torch.ones(context_len)
        self.pos_weights = nn.Parameter(init_weights)

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

        # Proyección lineal compleja asimétrica Q (Path 3 evolucionado a QKV)
        rank = 128  # Aumentamos el rango de transformación
        std = 1.0 / math.sqrt(embedding_dim)
        U_real = torch.empty(rank, embedding_dim).normal_(mean=0.0, std=std)
        U_imag = torch.empty(rank, embedding_dim).normal_(mean=0.0, std=std)
        self.complex_U_conj = nn.Parameter(torch.complex(U_real, U_imag))

        V_real = torch.empty(embedding_dim, rank).normal_(mean=0.0, std=std)
        V_imag = torch.empty(embedding_dim, rank).normal_(mean=0.0, std=std)
        self.complex_V = nn.Parameter(torch.complex(V_real, V_imag))

        # Parámetros SHA (Spectral Holographic Attention) — v9
        # beta_sha: temperatura de softmax para la atención espectral
        self.sha_beta = nn.Parameter(torch.tensor(8.0))
        # alpha_sha: escala del vector atendido al fusionar con query
        self.sha_alpha = nn.Parameter(torch.tensor(1.0))
        # recency_lambda: decaimiento ALiBi-style sobre distancias posicionales
        self.sha_recency_lambda = nn.Parameter(torch.tensor(0.05))
        # gate_sha: compuerta dual-path SHA vs multi-escala, inicializada a 0.5
        self.gate_sha = nn.Parameter(torch.tensor(0.5))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        phi = self.phases[indices]
        x_real = torch.cos(phi)
        x_imag = torch.sin(phi)

        pos_rot_real = self.pos_rotation_real
        pos_rot_imag = self.pos_rotation_imag

        bound_real = x_real * pos_rot_real - x_imag * pos_rot_imag
        bound_imag = x_real * pos_rot_imag + x_imag * pos_rot_real

        # Multi-Head Gating
        is_batched = len(indices.shape) == 2
        if not is_batched:
            W = self.pos_weights.view(-1, 1)
        else:
            W = self.pos_weights.view(1, -1, 1)

        gates = torch.sigmoid(self.gate_proj(phi))  # [..., C, n_head]
        gates = gates.unsqueeze(-1).expand(*gates.shape, self.head_dim)
        gates = gates.reshape(*phi.shape[:-1], self.embedding_dim)

        bound_real = bound_real * W * gates
        bound_imag = bound_imag * W * gates

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
        q = ctx_hv[:, -1, :]   # [B, D] complex
        H = ctx_hv[:, :-1, :]  # [B, C-1, D] complex

        # 2. Phase cosine similarity: Re(<H_p, q>) / sqrt(D)
        #    = Re( H[B, C-1, D] * conj(q)[B, 1, D] ).sum(-1)
        #    This is the correct holographic inner product — computed over
        #    INDIVIDUAL token vectors, not the superposition bundle.
        q_conj = torch.conj(q).unsqueeze(1)          # [B, 1, D]
        scores = torch.real(torch.sum(H * q_conj, dim=-1)) / math.sqrt(D)  # [B, C-1]

        # 3. ALiBi-style recency bias: recent tokens get a bonus
        #    distances[0] = furthest token, distances[C-2] = most recent
        distances = torch.arange(C - 1, device=ctx_hv.device).flip(0).float()  # [C-1]
        pos_bias = -torch.abs(self.sha_recency_lambda) * distances.unsqueeze(0)  # [1, C-1]
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
        psi_ms  = self.bundle_multiscale(ctx_hv)
        return gate * psi_sha + (1.0 - gate) * psi_ms

    def process_bundle(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Bloque Transformer Equivalente para el Vector Holográfico (H-FFN + Atención Lineal).
        """
        # 1. Proyección lineal compleja (Q-Transformation)
        temp = torch.matmul(psi, self.complex_V)
        proj = torch.matmul(temp, self.complex_U_conj)
        psi = psi + proj

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

        # Parámetros del CGRA (dimensión por dimensión, mapeo 4 entradas -> 2 salidas)
        # Entradas: [h.real, h.imag, x.real, x.imag]
        # Salidas: [real, imag] para compuertas y candidato
        std = 1.0 / math.sqrt(4)
        self.W_z = nn.Parameter(
            torch.empty(embedding_dim, 4, 2).normal_(mean=0.0, std=std)
        )
        self.b_z = nn.Parameter(torch.zeros(embedding_dim, 2))

        self.W_r = nn.Parameter(
            torch.empty(embedding_dim, 4, 2).normal_(mean=0.0, std=std)
        )
        self.b_r = nn.Parameter(torch.zeros(embedding_dim, 2))

        self.W_c = nn.Parameter(
            torch.empty(embedding_dim, 4, 2).normal_(mean=0.0, std=std)
        )
        self.b_c = nn.Parameter(torch.zeros(embedding_dim, 2))

    def cgra_update(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # h: [B, D] complex o [D] complex
        # x: [B, D] complex o [D] complex
        is_1d = h.ndim == 1
        if is_1d:
            h = h.unsqueeze(0)
            x = x.unsqueeze(0)

        # 1. Apilar partes reales e imaginarias
        stacked_inputs = torch.stack(
            [h.real, h.imag, x.real, x.imag], dim=-1
        )  # [B, D, 4]

        # 2. Computar compuertas de actualización (z) y reset (r)
        z = torch.sigmoid(
            torch.einsum("bdf,dfg->bdg", stacked_inputs, self.W_z) + self.b_z
        )  # [B, D, 2]
        r = torch.sigmoid(
            torch.einsum("bdf,dfg->bdg", stacked_inputs, self.W_r) + self.b_r
        )  # [B, D, 2]

        # 3. Aplicar compuerta de reset al estado anterior
        h_reset_real = r[..., 0] * h.real
        h_reset_imag = r[..., 1] * h.imag

        # 4. Computar estado candidato (tilde_h)
        I_c = torch.stack([h_reset_real, h_reset_imag, x.real, x.imag], dim=-1)
        h_tilde = torch.tanh(
            torch.einsum("bdf,dfg->bdg", I_c, self.W_c) + self.b_c
        )  # [B, D, 2]

        # 5. Actualización recurrente compuerta por compuerta
        h_new_real = (1.0 - z[..., 0]) * h.real + z[..., 0] * h_tilde[..., 0]
        h_new_imag = (1.0 - z[..., 1]) * h.imag + z[..., 1] * h_tilde[..., 1]

        out = torch.complex(h_new_real, h_new_imag)
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
        # Refinamiento multi-hop por interferencia
        for _ in range(steps):
            sim = torch.real(torch.matmul(state, torch.conj(keys_all).t())) / math.sqrt(
                D
            )

            weights = torch.softmax(sim * self.beta, dim=-1)
            retrieved = torch.matmul(weights.to(keys_all.dtype), keys_all)

            # CGRA actualiza el estado recurrentemente en vez de sumarle directamente
            state = self.cgra_update(state, retrieved)
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
