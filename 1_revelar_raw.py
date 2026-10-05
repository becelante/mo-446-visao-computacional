"""Passo 1: revela os RAW (.ARW) de cada pasta de config.RAW_SETS em PNG, com balanco de branco
e brilho comuns a todas as fotos do conjunto."""
from src import config
from src.passo1_raw_convert import convert_arw_to_png

if __name__ == "__main__":
    for raw_dir, png_dir, max_dim, bits in config.RAW_SETS:
        if not raw_dir.is_dir():
            print(f"Pasta {raw_dir.name}/ nao encontrada, pulando.")
            continue
        print(f"\n========== {raw_dir.name} ==========")
        convert_arw_to_png(raw_dir, png_dir, max_dim, bits)
