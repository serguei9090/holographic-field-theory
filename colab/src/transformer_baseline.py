import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    """
    Standard Multi-Head Causal Self-Attention block.
    """
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by head count"
        
        # Key, query, value projections as a single linear layer
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        
        # Causal mask register buffer
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, sequence length, embedding dimension
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        
        # Attention dropout (optional)
        if self.dropout > 0.0:
            att = nn.functional.dropout(att, p=self.dropout, training=self.training)
            
        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Reassemble all head outputs side-by-side
        
        return self.c_proj(y)


class MLP(nn.Module):
    """
    Standard Feed-Forward network block using GELU.
    """
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        if self.dropout > 0.0:
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class Block(nn.Module):
    """
    Standard pre-LayerNorm Transformer block.
    """
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections with Pre-LN setup
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    """
    Decoder-only language model using standard transformer blocks.
    """
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(max_seq_len, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, max_seq_len, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))
        
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying: share embedding weights with linear output layer
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to projection layers in residual blocks
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.max_seq_len, f"Cannot forward sequence of length {t}, max sequence length is {self.max_seq_len}"
        
        # Position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # [1, T]
        
        # Embed tokens and positions
        tok_emb = self.transformer.wte(idx) # [B, T, n_embd]
        pos_emb = self.transformer.wpe(pos) # [1, T, n_embd]
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Pass through attention/MLP blocks
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        # We only care about predicting the next token at the final position during sliding window evaluation
        # But during standard sequence training/generation, we can calculate logits for all steps
        logits = self.lm_head(x) # [B, T, vocab_size]
        
        loss = None
        if targets is not None:
            # Shift logits/targets for autoregressive learning if training the whole window
            # Or compute cross entropy on the final position only, depending on how our dataset is structured.
            # In our prepare_dataset, targets_tensor represents the next token immediately AFTER train_ctx (which has length CONTEXT_LEN).
            # So the target matches the logit at the very last sequence index (T-1).
            # Let's support both: if targets is 1D (matching final prediction only), we take logits[:, -1, :].
            if len(targets.shape) == 1:
                logits_final = logits[:, -1, :]
                loss = nn.functional.cross_entropy(logits_final, targets)
            else:
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                
        return logits, loss
