import os
import sys
import time
import math
import argparse
import collections
import torch
import torch.nn as nn
from dotenv import load_dotenv

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Parse console arguments
parser = argparse.ArgumentParser(description="Decoder-only Transformer Baseline Training")
parser.add_argument("--layers", type=int, default=1, choices=[1, 2], help="Number of transformer layers")
parser.add_argument("--dim", type=int, default=0, help="Embedding/hidden dimension (0 = auto-set based on layers)")
parser.add_argument("--reset", action="store_true", help="Ignore and delete prior checkpoint")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (default 5e-4 for stable training)")
args = parser.parse_args()

# Enforce GPU
if not torch.cuda.is_available():
    sys.exit("❌ ERROR: CUDA (GPU) is not available. Stopping execution to prevent slow CPU training.")

DEVICE = torch.device("cuda")
print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")

# Load environment variables
load_dotenv()

# Add colab/src to path
sys.path.insert(0, "./colab/src")
from train import prepare_dataset
from transformer_baseline import TransformerLM

# Set configuration parameters matching current CHFT configs
CONTEXT_LEN   = 64
EPOCHS        = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE    = 256
NUM_STORIES   = 5000
VAL_SPLIT     = 0.1
TOPK          = 5

# Auto-set hidden dimension based on layers to match official configurations if not overridden
if args.dim > 0:
    DIMENSION = args.dim
else:
    DIMENSION = 256 if args.layers == 1 else 512

N_HEADS = 8 if DIMENSION % 8 == 0 else 4

print(f"🚀 Starting Local Transformer Baseline Training ({args.layers}-Layer)")
print(f"🔧 Config: Layers={args.layers}, Dim={DIMENSION}, Context={CONTEXT_LEN}, Batch={BATCH_SIZE}, LR={LEARNING_RATE}")

# 1. Prepare Dataset (Reuses exact same seeding split to avoid data leakage)
train_ctx, train_tgt, val_ctx, val_tgt, token_to_idx, idx_to_token, vocab, vocab_size, tokenizer, targets_tensor = prepare_dataset(
    num_stories=NUM_STORIES,
    context_len=CONTEXT_LEN,
    val_split=VAL_SPLIT,
    device=DEVICE
)

# 2. Initialize Transformer Baseline Model
model = TransformerLM(
    vocab_size=vocab_size,
    n_embd=DIMENSION,
    n_head=N_HEADS,
    n_layer=args.layers,
    max_seq_len=CONTEXT_LEN,
    dropout=0.1
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model Parameters: {total_params:,} ({total_params * 4 / 1e6:.2f} MB on disk)")

# 3. Train Loop with Early Stopping
checkpoint_path = f"transformer_checkpoint_{args.layers}L.pth"
best_checkpoint_path = f"best_{checkpoint_path}"

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

if args.reset and os.path.exists(checkpoint_path):
    try:
        os.remove(checkpoint_path)
        print("  🗑️ Deleted previous checkpoint file.")
    except Exception as e:
        print(f"  ⚠️ Error deleting checkpoint: {e}")

start_epoch = 0
loss_history = []
val_loss_history = []
t0 = time.time()

if os.path.exists(checkpoint_path):
    print(f"  ⏳ Loading checkpoint: {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint.get('loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        print(f"  ✅ Resuming from Epoch {start_epoch + 1:02d}...")
    except Exception as e:
        print(f"  ⚠️ Error loading checkpoint: {e}. Starting from scratch...")
        start_epoch = 0

num_train = len(train_ctx)
num_val = len(val_ctx)
patience = 2

for epoch in range(start_epoch, EPOCHS):
    # ── TRAINING ──
    model.train()
    epoch_loss = 0.0
    perm = torch.randperm(num_train)
    num_batches = math.ceil(num_train / BATCH_SIZE)
    
    for b in range(num_batches):
        idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        ctx_b = train_ctx[idx]
        tgt_b = train_tgt[idx]
        
        _, loss = model(ctx_b, tgt_b)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    scheduler.step()
    
    # ── VALIDATION ──
    model.eval()
    val_loss_sum = 0.0
    val_batches = math.ceil(num_val / BATCH_SIZE)
    with torch.no_grad():
        for b in range(val_batches):
            ctx_v = val_ctx[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            tgt_v = val_tgt[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            _, loss = model(ctx_v, tgt_v)
            val_loss_sum += loss.item()
            
    avg_train = epoch_loss / num_batches
    avg_val = val_loss_sum / val_batches
    loss_history.append(avg_train)
    val_loss_history.append(avg_val)
    
    print(f"  Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
    
    # Save checkpoint
    try:
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_history': loss_history,
            'val_loss_history': val_loss_history
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        best_val = min(val_loss_history)
        if avg_val <= best_val:
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"  🏆 New best model saved (Val Loss: {avg_val:.4f})")
    except Exception as e:
        print(f"  ⚠️ Error saving checkpoint: {e}")
        
    # Check early stopping
    best_val = min(val_loss_history)
    best_epoch_idx = val_loss_history.index(best_val)
    if (len(val_loss_history) - 1 - best_epoch_idx) >= patience:
        print(f"\n  🛑 Early stopping: Val Loss did not improve for {patience} epochs.")
        break

elapsed = time.time() - t0
print(f"\n✅ Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Reload best checkpoint for evaluation
if os.path.exists(best_checkpoint_path):
    print(f"🏆 Loading best checkpoint found: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

# 4. Evaluate Metrics
print("\nCalculating benchmark metrics...")
model.eval()
correct = 0
total_ce = 0.0
n_val = 0

with torch.no_grad():
    for b in range(math.ceil(num_val / BATCH_SIZE)):
        ctx_v = val_ctx[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        tgt_v = val_tgt[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        
        logits, _ = model(ctx_v)
        logits_final = logits[:, -1, :] # Predict next token at final position
        
        preds = torch.argmax(logits_final, dim=1)
        correct += (preds == tgt_v).sum().item()
        total_ce += nn.functional.cross_entropy(logits_final, tgt_v, reduction='sum').item()
        n_val += len(tgt_v)

accuracy = correct / n_val * 100
perplexity = math.exp(total_ce / n_val)

print(f"  Accuracy@1  : {accuracy:.2f}%")
print(f"  Perplexity  : {perplexity:.2f}")

# Calculate frequency baseline
token_freq = collections.Counter(targets_tensor.cpu().tolist())
most_common = token_freq.most_common(1)[0][0]
base_correct = sum(1 for t in val_tgt.cpu().tolist() if t == most_common)
base_acc = base_correct / num_val * 100
print(f"  [Baseline freq] Accuracy@1: {base_acc:.2f}%")
print(f"  [Transformer]   Accuracy@1: {accuracy:.2f}%  (+{accuracy - base_acc:.2f}pp vs baseline)")

# 5. Generate Test Outputs
test_prompts = [
    ("Once upon a time",       20),
    ("A little girl saw",      20),
    ("The cat went to",        20),
    ("There was a small dog",  20),
    ("The sun was shining",    20),
]

def generate_text_topk(model, prompt, tokenizer, token_to_idx, idx_to_token, max_seq_len, device, max_new=20, k=5, temperature=0.8):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = [token_to_idx[t] for t in tokens if t in token_to_idx]
    if not input_ids:
        input_ids = [token_to_idx[tokenizer.eot_token]]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_new):
            cond = input_tensor[:, -max_seq_len:]
            logits, _ = model(cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-K
            v, idxs = torch.topk(logits, k, dim=-1)
            probs = torch.softmax(v, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            next_token_idx = idxs.gather(-1, next_idx)
            
            input_tensor = torch.cat((input_tensor, next_token_idx), dim=1)
            
    out_indices = input_tensor[0].cpu().tolist()
    out_tokens = [idx_to_token[i] for i in out_indices]
    return tokenizer.decode(out_tokens)

print("\n✍️ Generating test samples...")
all_generated_tokens = []
for prompt, length in test_prompts:
    res = generate_text_topk(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        token_to_idx=token_to_idx,
        idx_to_token=idx_to_token,
        max_seq_len=CONTEXT_LEN,
        device=DEVICE,
        max_new=length,
        k=TOPK,
        temperature=0.8
    )
    print(f"  📝 Prompt : '{prompt}'")
    print(f"     Output : {res}\n")
    
    all_generated_tokens.extend(tokenizer.encode(res))

# Diversity metric
unique_toks = len(set(all_generated_tokens))
div_score = unique_toks / len(all_generated_tokens) * 100 if all_generated_tokens else 0.0
print(f"── Diversity Score: {div_score:.1f}%")

# Peak VRAM
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

# 6. Copy-Pasteable console log summary
print("\n" + "="*50)
print(f"📊 SUMMARY OF METRICS (Transformer {args.layers}L Baseline)")
print("="*50)
print(f"Dimension          : {DIMENSION}")
print(f"Vocab Size         : {vocab_size:,} tokens")
print(f"Stories Used       : {NUM_STORIES}")
print(f"Train Samples      : {num_train:,}")
print(f"Actual Epochs      : {len(loss_history)} / {EPOCHS}")
print(f"Total Time         : {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Train Loss Final   : {loss_history[-1]:.4f}")
print(f"Val Loss Final     : {val_loss_history[-1]:.4f}")
print(f"Accuracy@1         : {accuracy:.2f}%")
print(f"Accuracy@1 (Base)  : {base_acc:.2f}%")
print(f"Perplexity         : {perplexity:.2f}")
print(f"Diversity Score    : {div_score:.1f}%")
print(f"Peak VRAM (GPU)    : {peak_vram:.1f} MB")
print("="*50 + "\n")
