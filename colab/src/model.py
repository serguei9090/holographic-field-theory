import math
import torch
import torch.nn as nn

class FHRRPhasorEmbedding(nn.Module):
    """
    Fourier Holographic Reduced Representation.
    Cada token es un vector de fases θ ∈ [-π, π]^D.
    El hipervector complejo es: e^{iθ} = cos(θ) + i·sin(θ).
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, context_len: int):
        super().__init__()
        self.phases = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim).uniform_(-math.pi, math.pi)
        )
        # Trainable positional attention weights
        # Inicializado con decaimiento de 0.85
        init_weights = 0.85 ** torch.arange(context_len).flip(0)
        self.pos_weights = nn.Parameter(init_weights)
        
        # Frecuencias de fase posicionales deterministas (True Positional Binding)
        g = torch.Generator().manual_seed(42)
        omega = torch.empty(embedding_dim).uniform_(-math.pi, math.pi, generator=g)
        self.register_buffer('omega', omega)
        
        # Precomputar rotaciones para acelerar el forward pass
        pos_angles = torch.arange(context_len).unsqueeze(1) * omega.unsqueeze(0)
        self.register_buffer('pos_rotation_real', torch.cos(pos_angles))
        self.register_buffer('pos_rotation_imag', torch.sin(pos_angles))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [B, C] o [C]
        phi = self.phases[indices] # [B, C, D] o [C, D]
        x_real = torch.cos(phi)
        x_imag = torch.sin(phi)
        
        # Aplicar rotación posicional: x * pos_rotation
        R = self.pos_rotation_real
        I = self.pos_rotation_imag
        
        bound_real = x_real * R - x_imag * I
        bound_imag = x_real * I + x_imag * R
        
        # Aplicar pesos posicionales aprendidos
        if len(indices.shape) == 1:
            W = self.pos_weights.view(-1, 1)
        else:
            W = self.pos_weights.view(1, -1, 1)
            
        bound_real = bound_real * W
        bound_imag = bound_imag * W
        
        return torch.complex(bound_real, bound_imag)

    def all_keys(self) -> torch.Tensor:
        return torch.complex(torch.cos(self.phases), torch.sin(self.phases))


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
    def query(self, state: torch.Tensor, refine_steps: int = 1) -> tuple[int, torch.Tensor]:
        """Retorna (índice_más_probable, similitudes_raw)"""
        if self.keys is None:
            raise ValueError("Keys not initialized. Call update_keys() first.")
        
        # Refinamiento multi-hop del estado
        state = self.refine_query(state, steps=refine_steps)
        
        sim = torch.real(
            torch.matmul(self.keys, torch.conj(state).unsqueeze(-1))
        ).squeeze(-1) / math.sqrt(self.keys.shape[-1])
        weights = torch.softmax(sim * self.beta, dim=0)
        return int(torch.argmax(weights).item()), sim

    @torch.no_grad()
    def query_topk(self, state: torch.Tensor, k: int = 5, refine_steps: int = 1) -> torch.Tensor:
        """Retorna logits para top-k sampling"""
        if self.keys is None:
            raise ValueError("Keys not initialized. Call update_keys() first.")
        
        # Refinamiento multi-hop del estado
        state = self.refine_query(state, steps=refine_steps)
        
        sim = torch.real(
            torch.matmul(self.keys, torch.conj(state).unsqueeze(-1))
        ).squeeze(-1) / math.sqrt(self.keys.shape[-1])
        return sim

    @torch.no_grad()
    def refine_query(self, state: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Refina el vector de consulta state usando recuperación iterativa residual (Multi-Hop).
        state: [B, D] complejo o [D] complejo
        """
        if self.keys is None:
            raise ValueError("Keys not initialized.")
        if steps <= 0:
            return state
            
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(0) # [1, D]
            
        refined = state.clone()
        D = self.keys.shape[-1]
        
        for _ in range(steps):
            # Similitud de coseno compleja
            sim = torch.real(
                torch.matmul(refined, torch.conj(self.keys).t())
            ) / math.sqrt(D) # [B, V]
            
            weights = torch.softmax(sim * self.beta, dim=-1) # [B, V]
            retrieved = torch.matmul(weights.to(self.keys.dtype), self.keys) # [B, D] complejo
            
            # Conexión residual y normalización
            refined = refined + retrieved
            norm = torch.clamp(torch.norm(refined, p=2, dim=-1, keepdim=True), min=1e-12)
            refined = refined / norm
            
        if is_single:
            refined = refined.squeeze(0)
        return refined
