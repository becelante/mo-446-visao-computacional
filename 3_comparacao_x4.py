import csv
import os

from src import config
from src.comparacao import compare_matchers, run_stitcher

if __name__ == "__main__":
    dataset = next(d for d in config.DATASETS if d["name"] == config.X4_DATASET)
    out_dir = config.X4_RESULTS
    os.makedirs(out_dir, exist_ok=True)

    print("[X4] SIFT x SuperPoint + LightGlue nos pares consecutivos...")
    summary = compare_matchers(dataset["png"], dataset["results"], out_dir)
    for row in summary:
        print(f"  {row}")

    print("[X4] cv2.Stitcher no mesmo conjunto de fotos...")
    stitcher = run_stitcher(dataset["png"], dataset["results"], out_dir)
    print(f"  {stitcher}")

    with open(out_dir / "resumo.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metodo", "metrica", "valor"])
        for row in summary + [stitcher]:
            for key, value in row.items():
                if key != "metodo":
                    writer.writerow([row["metodo"], key, value])
    print(f"Resultados em {out_dir}")
