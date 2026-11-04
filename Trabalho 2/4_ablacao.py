"""
Etapa 5, estudo de ablacao (Variante A), em results/<detector>/ablacao/:
  - emparelhamento sequencial x sequencial com deteccao de laco x exaustivo;
  - numero de quadros (todos, 1 a cada 2, 1 a cada 4).
Cada variante roda o pipeline inteiro (SfM, escala, alinhamento) e reporta cameras, pontos,
erro de reprojecao, deriva do laco e medidas da sala. O cache de features/pares e compartilhado,
entao so os pares novos de cada variante sao casados.
"""
import argparse
import csv
import json

from src import config
from src.pipeline import metric_errors, model_measurements, reconstruct
from src.step_0_frames import load_frames

# (nome, usar 1 a cada `step` quadros, estrategia de pares)
VARIANTS = [
    ("sequencial", 1, "sequential"),
    ("sequencial + laco", 1, "sequential_loop"),
    ("sequencial + laco, 1/2 dos quadros", 2, "sequential_loop"),
    ("sequencial + laco, 1/4 dos quadros", 4, "sequential_loop"),
    ("exaustivo, 1/4 dos quadros", 4, "exhaustive"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default=config.DETECTOR)
    args = parser.parse_args()
    all_paths, all_images = load_frames(config.frames_dir())
    out = config.results_dir(args.detector) / "ablacao"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, step, pairing in VARIANTS:
        print(f"\n=== {name} ===")
        paths, images = all_paths[::step], all_images[::step]
        try:
            m = reconstruct(paths, images, method=args.detector, pairing=pairing)["metrics"]
        except Exception as exc:  # uma variante fraca pode nao achar o tabuleiro ou o par inicial
            print(f"Falhou: {exc}")
            rows.append({"variante": name, "quadros": len(paths), "erro": str(exc)})
            continue
        row = {"variante": name, "quadros": len(paths), "pares": m["n_pairs"], "pares_verificados": m["n_pairs_verified"],
               "cameras": m["n_cameras"], "pontos": m["n_points"], "reproj_media_px": m["reproj_mean"],
               "reproj_mediana_px": m["reproj_median"], "track_medio": m["mean_track_length"],
               "deriva_cm": 100 * m["drift"]["delta_m"], "deriva_pct": m["drift"]["delta_percent"],
               "largura_m": m["dimensions"]["largura"], "comprimento_m": m["dimensions"]["comprimento"],
               "dispersao_escala_pct": 100 * m["scale"]["board_scale_spread"],
               "matching_s": m["times"]["matching_s_estimated"], "sfm_s": m["times"]["sfm_s"]}
        for e in metric_errors(model_measurements(m), config.real_measurements()):
            row[f"erro_{e['medida']}_cm"] = e["erro_cm"]
        rows.append(row)
        (out / f"{name.replace(' ', '_').replace(',', '').replace('/', '-')}.json").write_text(
            json.dumps(m, indent=2, default=str))

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "ablacao.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    lines += ["| " + " | ".join(f"{r[k]:.2f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in fields) + " |"
              for r in rows]
    (out / "ablacao.md").write_text("\n".join(lines) + "\n")
    print((out / "ablacao.md").read_text())
