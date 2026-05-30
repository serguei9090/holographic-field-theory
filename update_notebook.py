"""
update_notebook.py
──────────────────
Convierte run_colab_export.py  →  run_chft_experiment.ipynb

Uso:
    uv run update_notebook.py

El script:
  1. Lee run_colab_export.py (formato percent-notebook %%/# %% [markdown])
  2. Parsea manualmente cada celda
  3. Escribe un .ipynb válido que Colab puede abrir directamente

No requiere jupytext, solo la stdlib de Python.
"""

import json
import re
import pathlib

SRC  = pathlib.Path("run_colab_export.py")
DEST = pathlib.Path("run_chft_experiment.ipynb")


def parse_cells(source: str) -> list[dict]:
    """
    Parsea un archivo .py con formato percent-notebook:
      # %% [markdown]   → celda markdown
      # %%              → celda de código
    """
    cells: list[dict] = []

    # Separar por delimitadores %% (con o sin [markdown])
    pattern = re.compile(r'^# %%(.*)$', re.MULTILINE)
    splits   = pattern.split(source)

    # splits[0] = contenido antes del primer %% (ignorar)
    # splits[1], splits[2], splits[3], splits[4], ... = tipo, contenido, tipo, contenido, ...
    i = 1
    while i < len(splits) - 1:
        cell_type_tag = splits[i].strip()          # e.g. " [markdown]" o ""
        cell_content  = splits[i + 1]              # texto de la celda
        i += 2

        # Eliminar salto de línea inicial
        if cell_content.startswith('\n'):
            cell_content = cell_content[1:]
        # Eliminar salto de línea final extra
        cell_content = cell_content.rstrip('\n')

        is_markdown = '[markdown]' in cell_type_tag

        if is_markdown:
            # Quitar el prefijo "# " de cada línea de markdown
            md_lines = []
            for line in cell_content.splitlines():
                if line.startswith('# '):
                    md_lines.append(line[2:])
                elif line == '#':
                    md_lines.append('')
                else:
                    md_lines.append(line)
            source_lines = [ln + '\n' for ln in md_lines]
            # Última línea sin \n
            if source_lines:
                source_lines[-1] = source_lines[-1].rstrip('\n')

            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source_lines,
            })
        else:
            # Celda de código
            code_lines = [ln + '\n' for ln in cell_content.splitlines()]
            if code_lines:
                code_lines[-1] = code_lines[-1].rstrip('\n')

            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_lines,
            })

    return cells


def build_notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            },
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            }
        },
        "cells": cells,
    }


def main():
    print(f"[+] Leyendo {SRC}...")
    source = SRC.read_text(encoding="utf-8")

    print("[+] Parseando celdas...")
    cells = parse_cells(source)
    print(f"    -> {len(cells)} celdas encontradas "
          f"({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
          f"{sum(1 for c in cells if c['cell_type']=='code')} codigo)")

    print(f"[+] Escribiendo {DEST}...")
    nb = build_notebook(cells)
    DEST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    size_kb = DEST.stat().st_size / 1024
    print(f"[OK] Notebook actualizado: {DEST}  ({size_kb:.1f} KB)")
    print(f"     Listo para subir a Google Colab.")


if __name__ == "__main__":
    main()
