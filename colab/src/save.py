import os
import pickle
import shutil
import torch

def save_model_and_results(
    drive_dest_dir: str,
    codebook: torch.nn.Module,
    token_to_idx: dict[int, int],
    idx_to_token: dict[int, int],
    vocab: list[int],
    vocab_size: int,
    dimension: int,
    context_len: int,
    metrics: dict,
    local_chart_path: str = "chft_benchmark_results.png"
):
    print("[-] Guardando modelo y resultados...")
    os.makedirs(drive_dest_dir, exist_ok=True)

    model_data = {
        "phases": codebook.phases.cpu().detach(),
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "vocab": vocab,
        "vocab_size": vocab_size,
        "dimension": dimension,
        "context_len": context_len,
        "metrics": metrics
    }

    model_path = os.path.join(drive_dest_dir, 'chft_model_v2.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✅ Modelo guardado en: {model_path}")

    if os.path.exists(local_chart_path):
        dest_chart_path = os.path.join(drive_dest_dir, local_chart_path)
        shutil.copy(local_chart_path, dest_chart_path)
        print(f"✅ Gráfico guardado en: {dest_chart_path}")
    else:
        print(f"⚠️ No se encontró el gráfico local '{local_chart_path}' para copiar.")
