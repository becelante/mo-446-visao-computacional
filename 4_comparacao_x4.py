"""Passo 4: extra X4 no conjunto config.X4_DATASET (SIFT x SuperPoint + LightGlue na GPU).
Usa as saidas do passo 3."""
import os

from src import config
from src.passo2_pipeline import write_csv
from src.passo4_comparacao_x4 import compare_matchers

if __name__ == "__main__":
    dataset = next(d for d in config.EXTRAS if d["name"] == config.X4_DATASET)
    os.makedirs(config.X4_RESULTS.parent, exist_ok=True)

    print("[X4] SIFT x SuperPoint + LightGlue nos pares consecutivos...")
    summary = compare_matchers(dataset["png"], dataset["results"])
    for row in summary:
        print(f"  {row}")
    write_csv(config.X4_RESULTS, summary)
    print(f"Resultados em {config.X4_RESULTS}")
