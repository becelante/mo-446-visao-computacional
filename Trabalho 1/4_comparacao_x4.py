"""Passo 4: extra X4 (SuperPoint + LightGlue na GPU), em extras/x4/:
  - sift_vs_lightglue.csv: SIFT x LightGlue nos pares consecutivos de config.X4_DATASET (usa o passo 3);
  - stitcher_vs_lightglue.csv, panorama_*.png e detalhe_*.png: cv2.Stitcher x LightGlue no D1, com a
    mesma composicao (a do cv2.Stitcher, em projecao esferica)."""
import os

from src import config
from src.passo2_pipeline import write_csv
from src.passo4_comparacao_x4 import compare_matchers, compare_stitcher, load_lightglue

if __name__ == "__main__":
    os.makedirs(config.X4_RESULTS, exist_ok=True)
    lightglue = load_lightglue()

    dataset = next(d for d in config.EXTRAS if d["name"] == config.X4_DATASET)
    print(f"[X4] SIFT x SuperPoint + LightGlue nos pares consecutivos do {config.X4_DATASET}...")
    summary = compare_matchers(dataset["png"], dataset["results"], *lightglue)
    for row in summary:
        print(f"  {row}")
    write_csv(config.X4_RESULTS / "sift_vs_lightglue.csv", summary)

    print(f"\n[X4] cv2.Stitcher x SuperPoint + LightGlue no {config.BASE_INPUT.name}, mesma composicao...")
    write_csv(config.X4_RESULTS / "stitcher_vs_lightglue.csv", compare_stitcher(config.BASE_INPUT, config.X4_RESULTS, *lightglue))
    print(f"Resultados em {config.X4_RESULTS}")
