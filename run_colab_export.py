# %% [markdown]
# # Experimento de Campos Holográficos (CHFT v2)
# 
# Este notebook implementa un experimento de **Campos Holográficos (CHFT)** utilizando representaciones holográficas reducidas de Fourier (FHRR) y una Memoria de Hopfield Moderna para tareas de predicción y generación de texto en un subset del dataset `TinyStories`.

# %%
# Instalar dependencias requeridas en Google Colab
!pip install datasets tiktoken

# %%
import torch
import torch.nn as nn
import numpy as np
import time
from datasets import load_dataset
import tiktoken
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# CONFIGURACIÓN Y PARÁMETROS DEL MODELO
# -------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIMENSION = 2048      # Dimensión hipervectorial FHRR
CONTEXT_LEN = 8       # Longitud de contexto histórico (N)
EPOCHS = 5            # Épocas de entrenamiento
LEARNING_RATE = 0.02  # Tasa de ajuste de fase compleja
BATCH_SIZE = 512      # BATCHING VECTORIAL para acelerar entrenamiento
NUM_STORIES = 200     # Número de historias para entrenar

print(f"[-] Usando dispositivo: {DEVICE}")
print(f"[-] Dimensión hipervectorial FHRR: {DIMENSION}")

# %% [markdown]
# ## 1. Preparación del Dataset y Tokenización

# %%
print("[-] Cargando TinyStories...")
dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{NUM_STORIES}]")

tokenizer = tiktoken.get_encoding("cl100k_base")

print("[-] Analizando vocabulario...")
all_token_ids = set()
stories_tokenized = []

for item in dataset:
    story = item["text"]
    tokens = tokenizer.encode(story)
    stories_tokenized.append(tokens)
    all_token_ids.update(tokens)

vocab = list(all_token_ids)
vocab_size = len(vocab)
token_to_idx = {token: i for i, token in enumerate(vocab)}
idx_to_token = {i: token for i, token in enumerate(vocab)}

print(f"[-] Vocabulario único en subset: {vocab_size} tokens.")

# Preparar todas las secuencias en tensores para paralelización total
print("[-] Preparando batches vectoriales en PyTorch...")
contexts_list = []
targets_list = []

for story in stories_tokenized:
    if len(story) < CONTEXT_LEN + 1:
        continue
    story_idx = [token_to_idx[t] for t in story]
    for i in range(len(story_idx) - CONTEXT_LEN):
        contexts_list.append(story_idx[i:i + CONTEXT_LEN])
        targets_list.append(story_idx[i + CONTEXT_LEN])

contexts_tensor = torch.tensor(contexts_list, dtype=torch.long, device=DEVICE)
targets_tensor = torch.tensor(targets_list, dtype=torch.long, device=DEVICE)
num_samples = len(contexts_tensor)
print(f"[-] Total de ejemplos de entrenamiento: {num_samples}")

# %% [markdown]
# ## 2. Capa FHRR Manual y Memoria de Hopfield Moderna

# %%
class FHRRPhasorEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.phases = nn.Parameter(torch.empty(num_embeddings, embedding_dim).uniform_(-np.pi, np.pi))
        
    def forward(self, indices):
        selected_phases = self.phases[indices]
        return torch.complex(torch.cos(selected_phases), torch.sin(selected_phases))

codebook = FHRRPhasorEmbedding(vocab_size, DIMENSION).to(DEVICE)

class ModernHopfieldMemory:
    def __init__(self, size, dim, beta=12.0):
        self.dim = dim
        self.beta = beta
        self.keys = None
        
    def update_keys(self, codebook_module):
        with torch.no_grad():
            phases = codebook_module.phases
            self.keys = torch.complex(torch.cos(phases), torch.sin(phases)).detach()
        
    def query(self, state_vector):
        state_complex = state_vector
        keys_complex = self.keys
        similarity = torch.real(torch.matmul(keys_complex, torch.conj(state_complex).unsqueeze(-1))).squeeze(-1)
        attention_weights = torch.softmax(similarity * self.beta, dim=0)
        collapsed_idx = torch.argmax(attention_weights).item()
        return collapsed_idx, similarity

hopfield_memory = ModernHopfieldMemory(vocab_size, DIMENSION)
hopfield_memory.update_keys(codebook)

# %% [markdown]
# ## 3. Entrenamiento (Ajuste de Fases con Batching Paralelo)

# %%
print("\n[-] Iniciando Entrenamiento (Ajuste de Fases)...")
optimizer = torch.optim.Adam(codebook.parameters(), lr=LEARNING_RATE)

start_time = time.time()
loss_history = []

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    permutation = torch.randperm(num_samples)
    num_batches = int(np.ceil(num_samples / BATCH_SIZE))
    
    for b in range(num_batches):
        indices = permutation[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
        if len(indices) == 0:
            continue
            
        batch_contexts = contexts_tensor[indices]
        batch_targets = targets_tensor[indices]
        
        context_hvs = codebook(batch_contexts)
        sum_complex = torch.sum(context_hvs, dim=1)
        psi_states = torch.nn.functional.normalize(sum_complex, p=2, dim=1)
        
        phases_all = codebook.phases
        keys_complex = torch.complex(torch.cos(phases_all), torch.sin(phases_all))
        
        logits = torch.real(torch.matmul(psi_states, torch.conj(keys_complex).t()))
        loss = torch.nn.functional.cross_entropy(logits * 5.0, batch_targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            codebook.phases.copy_(torch.remainder(codebook.phases + np.pi, 2 * np.pi) - np.pi)
            
        epoch_loss += loss.item()
        
    hopfield_memory.update_keys(codebook)
    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss Promedio: {avg_loss:.4f}")

end_time = time.time()
print(f"[-] Entrenamiento completado en: {end_time - start_time:.2f} segundos.")

# %% [markdown]
# ## 4. Generación de Texto

# %%
def generate_text(prompt, max_len=15):
    prompt_tokens = tokenizer.encode(prompt)
    print(f"\nPrompt original: '{prompt}'")
    
    valid_tokens = [t for t in prompt_tokens if t in token_to_idx]
    if not valid_tokens:
        print("Error: Los tokens del prompt no están en el vocabulario.")
        return
        
    generated_indices = [token_to_idx[t] for t in valid_tokens]
    
    for _ in range(max_len):
        context_indices = generated_indices[-CONTEXT_LEN:]
        if len(context_indices) < CONTEXT_LEN:
            context_indices = context_indices + [context_indices[-1]] * (CONTEXT_LEN - len(context_indices))
            
        context_tensor = torch.tensor(context_indices, device=DEVICE)
        context_hvs = codebook(context_tensor)
        
        sum_complex = torch.sum(context_hvs, dim=0)
        psi_state = torch.nn.functional.normalize(sum_complex, p=2, dim=0)
        
        next_idx, _ = hopfield_memory.query(psi_state)
        generated_indices.append(next_idx)
        
    tokens_to_decode = [idx_to_token[idx] for idx in generated_indices]
    decoded_text = tokenizer.decode(tokens_to_decode)
    print(f"Texto generado final:\n-> {decoded_text}")

test_prompts = [
    "Once upon a time",
    "A little girl saw",
    "The cat went to"
]

for p in test_prompts:
    generate_text(p, max_len=12)

# %% [markdown]
# ## 5. Graficar Resultados de la Curva de Pérdida

# %%
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', color='purple', linestyle='dashed')
plt.title("Curva de Pérdida en Campos Holográficos (CHFT v2)")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 6. Guardar Modelo en Google Drive (Permanencia)

# %%
from google.colab import drive
import os
import pickle

# 1. Montar Google Drive
print("[-] Conectando con Google Drive...")
drive.mount('/content/drive')

# 2. Definir y crear la carpeta dedicada 'colabStore/subcarpeta'
drive_dest_dir = '/content/drive/MyDrive/colabStore/01-CHFT'
os.makedirs(drive_dest_dir, exist_ok=True)

# 3. Preparar los datos del codebook y vocabulario
model_data = {
    "phases": codebook.phases.cpu().data,
    "token_to_idx": token_to_idx,
    "idx_to_token": idx_to_token,
    "vocab": vocab,
    "vocab_size": vocab_size,
    "dimension": DIMENSION,
    "context_len": CONTEXT_LEN
}

# 4. Guardar archivo pickle de forma permanente en Google Drive
dest_file_path = os.path.join(drive_dest_dir, 'chft_model.pkl')
with open(dest_file_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"[-] ¡Modelo CHFT exportado con éxito a Google Drive en: {dest_file_path}!")


