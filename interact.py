import os
import sys
import time
import json
import torch
import tiktoken

# Reconfigure stdout/stderr to UTF-8 on Windows for emoji printing support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add src to system path
sys.path.insert(0, "./colab/src")
from model import FHRRPhasorEmbedding, ModernHopfieldMemory
from generate import generate_text_topk

def main():
    print("🚀 CHFT Interactive Inference Console")
    print("=" * 45)
    
    # 1. Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 2. Check for vocab.json and checkpoints
    vocab_path = "vocab.json"
    checkpoint_path = "best_chft_checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "chft_checkpoint.pth"
        
    if not os.path.exists(vocab_path):
        print(f"❌ ERROR: Vocab file '{vocab_path}' not found.")
        print("Please run 'main.py' at least once to generate 'vocab.json' from the dataset.")
        sys.exit(1)
        
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: No checkpoint file found (neither 'best_chft_checkpoint.pth' nor 'chft_checkpoint.pth').")
        print("Please train the model first using 'main.py'.")
        sys.exit(1)
        
    # 3. Load Vocab
    print(f"⏳ Loading vocabulary from '{vocab_path}'...")
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
        
    # Convert keys back to integers (JSON stores dict keys as strings)
    token_to_idx = {int(k): int(v) for k, v in vocab_data["token_to_idx"].items()}
    idx_to_token = {int(k): int(v) for k, v in vocab_data["idx_to_token"].items()}
    vocab_size = len(token_to_idx)
    print(f"✅ Loaded vocabulary of {vocab_size:,} tokens.")
    
    # 4. Config variables (must match training run)
    DIMENSION = 16384
    CONTEXT_LEN = 8
    
    # 5. Build and Load Model
    print(f"⏳ Initializing model with Dim={DIMENSION}, Context={CONTEXT_LEN}...")
    codebook = FHRRPhasorEmbedding(vocab_size, DIMENSION, CONTEXT_LEN).to(device)
    hopfield_mem = ModernHopfieldMemory().to(device)
    
    print(f"⏳ Loading weights from '{checkpoint_path}'...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        codebook.load_state_dict(checkpoint['codebook_state_dict'])
        hopfield_mem.load_state_dict(checkpoint['hopfield_state_dict'])
        hopfield_mem.update_keys(codebook)
        print(f"✅ Model loaded successfully (saved at Epoch {checkpoint['epoch'] + 1}).")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        sys.exit(1)
        
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # 6. Interactive loop
    print("\nModel is ready! Type 'exit' to quit.")
    print("=" * 45)
    
    while True:
        try:
            prompt = input("\n📝 Prompt > ").strip()
            if not prompt:
                continue
            if prompt.lower() == 'exit':
                break
                
            try:
                num_tokens = int(input("🔢 Tokens to generate (default 20) > ") or 20)
            except ValueError:
                num_tokens = 20
                
            print("⚡ Generating...")
            t0 = time.time()
            
            # Generate text
            output = generate_text_topk(
                prompt=prompt,
                codebook=codebook,
                hopfield_mem=hopfield_mem,
                tokenizer=tokenizer,
                token_to_idx=token_to_idx,
                idx_to_token=idx_to_token,
                context_len=CONTEXT_LEN,
                device=device,
                max_new=num_tokens,
                k=5,
                temperature=0.8,
                refine_steps=1
            )
            
            elapsed = time.time() - t0
            tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0
            
            print("-" * 45)
            print(f"📖 Output: {output}")
            print("-" * 45)
            print(f"⏱️ Time taken: {elapsed:.3f}s | Speed: {tok_per_sec:.1f} tokens/second")
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"⚠️ Error during generation: {e}")

if __name__ == "__main__":
    main()
