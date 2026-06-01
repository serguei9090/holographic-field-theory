import os
import sys
import torch
import argparse
from dotenv import load_dotenv

# Reconfigure stdout/stderr to UTF-8 on Windows for emoji printing support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Parse console arguments
parser = argparse.ArgumentParser(description="CHFT Local Training Entry Point")
parser.add_argument("--reset", action="store_true", help="Ignora y elimina el checkpoint de entrenamiento anterior para empezar desde cero")
parser.add_argument("--stories", type=int, default=5000, help="Cantidad de historias a usar")
parser.add_argument("--epochs", type=int, default=10, help="Épocas de entrenamiento")
parser.add_argument("--dim", type=int, default=768, help="Dimensión de los fasores complejos")
parser.add_argument("--context", type=int, default=64, help="Longitud de contexto")
parser.add_argument("--batch_size", type=int, default=256, help="Tamaño de batch")
args = parser.parse_args()

# 1. Enforce GPU (RTX 3060) to prevent running on CPU
if not torch.cuda.is_available():
    sys.exit("❌ ERROR: CUDA (GPU) is not available. Stopping execution to prevent slow CPU training.")

DEVICE = torch.device("cuda")
print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")

# 2. Load environment variables (HF_TOKEN)
load_dotenv()
if "HF_TOKEN" in os.environ:
    print("✅ HF_TOKEN loaded from environment.")

# 3. Add colab/src directory to path to reuse all module code
sys.path.insert(0, "./colab/src")

from model import FHRRPhasorEmbedding, ModernHopfieldMemory
from train import prepare_dataset, run_training_loop
from evaluate import evaluate_model, plot_and_save_results
from generate import generate_text_topk, calculate_diversity

# 4. Configurar Parámetros (Sync con Colab V4)
DIMENSION     = args.dim      # Dimensión de los fasores complejos (VSA de fase)
CONTEXT_LEN   = args.context  # Longitud de contexto (ventana)
EPOCHS        = args.epochs   # Épocas de entrenamiento
LEARNING_RATE = 0.01
BATCH_SIZE    = args.batch_size # Batch size
NUM_STORIES   = args.stories  # Cantidad de historias TinyStories
TOPK          = 5        # Parámetro K para sampling de texto
VAL_SPLIT     = 0.1      # 10% para validación

print(f"🚀 Iniciando entrenamiento local de CHFT en: {torch.cuda.get_device_name(0)}")
print(f"🔧 Configuración: Dim={DIMENSION}, Contexto={CONTEXT_LEN}, Batch={BATCH_SIZE}, Historias={NUM_STORIES}")

# 5. Cargar y Preparar Datos
train_ctx, train_tgt, val_ctx, val_tgt, token_to_idx, idx_to_token, vocab, vocab_size, tokenizer, targets_tensor = prepare_dataset(
    num_stories=NUM_STORIES,
    context_len=CONTEXT_LEN,
    val_split=VAL_SPLIT,
    device=DEVICE
)

# 6. Inicializar Componentes de CHFT
codebook = FHRRPhasorEmbedding(vocab_size, DIMENSION, CONTEXT_LEN).to(DEVICE)
hopfield_mem = ModernHopfieldMemory(DIMENSION, beta=16.0).to(DEVICE)

total_params = sum(p.numel() for p in codebook.parameters()) + sum(p.numel() for p in hopfield_mem.parameters())
print(f"✅ Parámetros del Modelo: {total_params:,} ({total_params * 4 / 1e6:.2f} MB en float32)")

# 7. Entrenar el Modelo
loss_history, val_loss_history, elapsed = run_training_loop(
    codebook=codebook,
    hopfield_mem=hopfield_mem,
    train_ctx=train_ctx,
    train_tgt=train_tgt,
    val_ctx=val_ctx,
    val_tgt=val_tgt,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
    checkpoint_path="chft_checkpoint.pth",
    reset_checkpoint=args.reset
)

# Cargar el mejor checkpoint guardado para evaluación e inferencia
best_checkpoint_path = "best_chft_checkpoint.pth"
if os.path.exists(best_checkpoint_path):
    print(f"\n🏆 Cargando el mejor punto de control encontrado: {best_checkpoint_path}")
    try:
        checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE, weights_only=True)
        codebook.load_state_dict(checkpoint['codebook_state_dict'])
        hopfield_mem.load_state_dict(checkpoint['hopfield_state_dict'])
    except Exception as e:
        print(f"  ⚠️ Error al cargar el mejor checkpoint: {e}")

# 8. Evaluar Métricas
accuracy, perplexity, base_acc = evaluate_model(
    codebook=codebook,
    hopfield_mem=hopfield_mem,
    val_ctx=val_ctx,
    val_tgt=val_tgt,
    batch_size=BATCH_SIZE,
    vocab_size=vocab_size,
    targets_tensor=targets_tensor,
    val_split=VAL_SPLIT
)


# 9. Generar Texto de Prueba
test_prompts = [
    ("Once upon a time",       20),
    ("A little girl saw",      20),
    ("The cat went to",        20),
    ("There was a small dog",  20),
    ("The sun was shining",    20),
]

print("\n✍️ Generando texto de prueba...")
for prompt, length in test_prompts:
    result = generate_text_topk(
        prompt=prompt,
        codebook=codebook,
        hopfield_mem=hopfield_mem,
        tokenizer=tokenizer,
        token_to_idx=token_to_idx,
        idx_to_token=idx_to_token,
        context_len=CONTEXT_LEN,
        device=DEVICE,
        max_new=length,
        k=TOPK,
        temperature=0.8,
        refine_steps=1  # 1 paso de refinamiento de atractores
    )
    print(f"  📝 Prompt : '{prompt}'")
    print(f"     Output : {result}\n")

# 10. Calcular Métrica de Diversidad
unique_ratio = calculate_diversity(
    prompts=test_prompts,
    codebook=codebook,
    hopfield_mem=hopfield_mem,
    tokenizer=tokenizer,
    token_to_idx=token_to_idx,
    idx_to_token=idx_to_token,
    context_len=CONTEXT_LEN,
    device=DEVICE,
    k=TOPK,
    temperature=0.8,
    refine_steps=1
)

# 11. Mostrar y Graficar Resultados
num_train = len(train_ctx)
plot_and_save_results(
    epochs=EPOCHS,
    loss_history=loss_history,
    val_loss_history=val_loss_history,
    base_acc=base_acc,
    accuracy=accuracy,
    dimension=DIMENSION,
    vocab_size=vocab_size,
    num_stories=NUM_STORIES,
    num_train=num_train,
    elapsed=elapsed,
    unique_ratio=unique_ratio,
    perplexity=perplexity
)
