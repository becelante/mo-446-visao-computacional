"""Roda o baseline SIFT numa particao.

    uv run python -m baseline.run_sift val     # busca dos limiares na validacao e predicoes com os melhores
    uv run python -m baseline.run_sift test    # aplica no teste os limiares escolhidos na validacao (uma vez, no fim)

1. Extrai os pares SIFT de cada imagem (GPU) e guarda em results/baseline/sift_pares_<particao>_<lado>.pkl.
   E a parte cara; os limiares abaixo so filtram esses pares, entao a busca nao repete a extracao.
2. Na validacao: testa todas as combinacoes da grade de config.yaml, salva a tabela
   results/baseline/sift_busca_val.csv e os parametros com maior S em sift_parametros.json.
3. Salva as predicoes com os parametros escolhidos (sift_<particao>.csv) e as metricas.
"""
import itertools
import json
import os
import pickle
import sys
import time

import pandas as pd
from tqdm import tqdm

from baseline import sift
from evaluate import score_file
from pipeline import data, metrics
from pipeline.common import get_device, load_config

cfg = load_config()
bcfg = cfg["baseline_sift"]
out_dir = os.path.join(cfg["results_dir"], "baseline")
os.makedirs(out_dir, exist_ok=True)


def extract(images, max_side):
    """Pares SIFT de todas as imagens (com cache). Devolve {id: dict(gray, P, Q, ratio, shape)}."""
    cache = os.path.join(out_dir, f"sift_pares_{split}_{max_side}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    device, feats, t0 = get_device(), {}, time.time()
    for _, r in tqdm(images.iterrows(), total=len(images), desc=f"pares SIFT (lado {max_side})"):
        img = data.load_image(r["path"])
        gray, _ = sift.prepare(img, max_side)
        P, Q, ratio, _ = sift.self_matches(gray, bcfg["n_features"], bcfg["min_dist"], device)
        feats[r["id"]] = {"gray": gray, "P": P, "Q": Q, "ratio": ratio, "shape": img.shape[:2]}
    print(f"extracao: {(time.time() - t0) / len(images):.3f} s por imagem")
    pickle.dump(feats, open(cache, "wb"))
    return feats


def predict_all(feats, images, params):
    """Predicoes no formato de submissao."""
    rows = []
    for image_id in images["id"]:
        f = feats[image_id]
        masks = sift.predict(f["gray"], f["P"], f["Q"], f["ratio"], f["shape"], params)
        rows.append({"case_id": image_id, "annotation": metrics.masks_to_annotation(masks)})
    return pd.DataFrame(rows)


split = sys.argv[1]
images = data.load_split(split)
params_path = os.path.join(out_dir, "sift_parametros.json")

if split == "val":
    solution = data.solution(images)
    grid = bcfg["grid"]
    results = []
    for max_side in bcfg["max_side"]:
        feats = extract(images, max_side)
        combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
        for params in tqdm(combos, desc=f"busca (lado {max_side})"):
            m, _ = metrics.evaluate(solution, predict_all(feats, images, params))
            results.append({"max_side": max_side, **params, "S": m["S"], "precisao": m["precisao"],
                            "revocacao": m["revocacao"], "oF1_detectadas": m["oF1_forjadas_detectadas"]})
    results = pd.DataFrame(results).sort_values("S", ascending=False)
    results.to_csv(os.path.join(out_dir, "sift_busca_val.csv"), index=False)
    print(results.head(10).to_string(index=False, float_format="%.4f"))
    best = results.iloc[0].to_dict()
    best = {k: (int(v) if k in ("max_side", "min_inliers") else float(v)) for k, v in best.items()
            if k in ("max_side", *grid)}
    json.dump(best, open(params_path, "w"), indent=2)
else:
    best = json.load(open(params_path))                  # escolhidos na validacao, sem olhar o teste

print(f"\nparametros: {best}")
feats = extract(images, best["max_side"])
t0 = time.time()
pred = predict_all(feats, images, best)
print(f"predicao (sem a extracao): {(time.time() - t0) / len(images):.3f} s por imagem")
path = os.path.join(out_dir, f"sift_{split}.csv")
pred.to_csv(path, index=False)
print(f"===== {split}")
score_file(path, split)
