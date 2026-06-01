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
    def __init__(self, num_embeddings: int, embedding_dim: int, context_len: int, n_head: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        assert embedding_dim % n_head == 0, "embedding_dim debe ser divisible por n_head"
        self.head_dim = embedding_dim // n_head

        self.phases = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim).uniform_(-math.pi, math.pi)
        )
        
        init_weights = 0.85 ** torch.arange(context_len).flip(0)
        self.pos_weights = nn.Parameter(init_weights)
        
        g = torch.Generator().manual_seed(42)
        omega = torch.empty(embedding_dim).uniform_(-math.pi, math.pi, generator=g)
        self.register_buffer('omega', omega)
        
        pos_angles = torch.arange(context_len).unsqueeze(1) * omega.unsqueeze(0)
        self.register_buffer('pos_rotation_real', torch.cos(pos_angles))
        self.register_buffer('pos_rotation_imag', torch.sin(pos_angles))

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
        rank = 128 # Aumentamos el rango de transformación
        std = 1.0 / math.sqrt(embedding_dim)
        U_real = torch.empty(rank, embedding_dim).normal_(mean=0.0, std=std)
        U_imag = torch.empty(rank, embedding_dim).normal_(mean=0.0, std=std)
        self.complex_U_conj = nn.Parameter(torch.complex(U_real, U_imag))
        
        V_real = torch.empty(embedding_dim, rank).normal_(mean=0.0, std=std)
        V_imag = torch.empty(embedding_dim, rank).normal_(mean=0.0, std=std)
        self.complex_V = nn.Parameter(torch.complex(V_real, V_imag))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        phi = self.phases[indices]
        x_real = torch.cos(phi)
        x_imag = torch.sin(phi)
        
        R = self.pos_rotation_real
        I = self.pos_rotation_imag
        
        bound_real = x_real * R - x_imag * I
        bound_imag = x_real * I + x_imag * R
        
        # Multi-Head Gating
        is_batched = len(indices.shape) == 2
        if not is_batched:
            W = self.pos_weights.view(-1, 1)
        else:
            W = self.pos_weights.view(1, -1, 1)
            
        gates = torch.sigmoid(self.gate_proj(phi)) # [..., C, n_head]
        gates = gates.unsqueeze(-1).expand(*gates.shape, self.head_dim)
        gates = gates.reshape(*phi.shape[:-1], self.embedding_dim)
            
        bound_real = bound_real * W * gates
        bound_imag = bound_imag * W * gates
        
        return torch.complex(bound_real, bound_imag)

    def all_keys(self) -> torch.Tensor:
        return torch.complex(torch.cos(self.phases), torch.sin(self.phases))

    def bundle_multiscale(self, ctx_hv: torch.Tensor, short_len: int = 16) -> torch.Tensor:
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
    Recuperación de atractores nativa por interferencia geométrica pura.
    """
    def __init__(self, embedding_dim: int, beta: float = 16.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.embedding_dim = embedding_dim

    def refine_and_predict(self, psi: torch.Tensor, codebook: FHRRPhasorEmbedding, steps: int = 2) -> torch.Tensor:
        """
        Realiza el multi-hop routing y computa los logits directamente usando cosenos nativos.
        psi: [B, D] state
        """
        D = self.embedding_dim
        keys_all = codebook.all_keys()
        
        state = psi
        # Refinamiento multi-hop por interferencia
        for _ in range(steps):
            sim = torch.real(
                torch.matmul(state, torch.conj(keys_all).t())
            ) / math.sqrt(D)
            
            weights = torch.softmax(sim * self.beta, dim=-1)
            retrieved = torch.matmul(weights.to(keys_all.dtype), keys_all)
            
            state = state + retrieved
            state = codebook.ln_2(state) / math.sqrt(D)
        
        # Logits finales usando fases nativas puras
        state_r = state.real
        state_i = state.imag
        keys_r = keys_all.real
        keys_i = keys_all.imag
        logits = (torch.matmul(state_r, keys_r.t()) + torch.matmul(state_i, keys_i.t())) / math.sqrt(D)
        
        return logits
