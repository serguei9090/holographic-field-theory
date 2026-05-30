import math
import torch
import torch.nn as nn

class FHRRPhasorEmbedding(nn.Module):
    """
    Fourier Holographic Reduced Representation.
    Cada token es un vector de fases θ ∈ [-π, π]^D.
    El hipervector complejo es: e^{iθ} = cos(θ) + i·sin(θ).
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
        if self.keys is None:
            raise ValueError("Keys not initialized. Call update_keys() first.")
        # Dividir por sqrt(D) para normalizar similitud
        sim = torch.real(
            torch.matmul(self.keys, torch.conj(state).unsqueeze(-1))
        ).squeeze(-1) / math.sqrt(self.keys.shape[-1])
        weights = torch.softmax(sim * self.beta, dim=0)
        return int(torch.argmax(weights).item()), sim

    @torch.no_grad()
    def query_topk(self, state: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Retorna logits para top-k sampling"""
        if self.keys is None:
            raise ValueError("Keys not initialized. Call update_keys() first.")
        # Dividir por sqrt(D) para normalizar similitud
        sim = torch.real(
            torch.matmul(self.keys, torch.conj(state).unsqueeze(-1))
        ).squeeze(-1) / math.sqrt(self.keys.shape[-1])
        return sim
