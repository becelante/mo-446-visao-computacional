"""Aplica a rede treinada numa particao e escolhe os limiares na validacao.

    uv run python predict.py results/unet/unet_base val    # busca dos limiares + predicoes da validacao
    uv run python predict.py results/unet/unet_base test   # limiares da validacao no teste (uma vez, no fim)

1. Roda a rede (checkpoints/melhor.ckpt) em todas as imagens e guarda os mapas de probabilidade.
2. Na validacao: testa a grade de limiares de config.yaml (unet.grid), salva busca_val.csv e os
   melhores em limiares.json. No teste: so le limiares.json.
3. Salva pred_<particao>.csv no formato de submissao e as metricas ao lado.
"""
import itertools
import json
import os
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import score_file
from pipeline import data, metrics
from pipeline.common import get_device, load_config
from pipeline.dataset import ForgeryDataset
from pipeline.segmenter import Segmenter, to_annotation

run_dir, split = sys.argv[1], sys.argv[2]
cfg = load_config()
device = get_device()
images = data.load_split(split)
model = Segmenter.load_from_checkpoint(os.path.join(run_dir, "checkpoints", "melhor.ckpt"), map_location=device)
model.eval()
c = model.cfg


@torch.no_grad()
def probabilities():
    """Mapa de probabilidade de cada imagem (so a parte util) e o tamanho original."""
    dataset = ForgeryDataset(images, c["size"], os.path.join(cfg["results_dir"], "cache"))
    out = [None] * len(dataset)
    t0 = time.time()
    for batch in tqdm(DataLoader(dataset, batch_size=c["batch_size"], num_workers=c["num_workers"]), desc="rede"):
        with torch.autocast(device.type, enabled=cfg["amp"]):
            probs = torch.sigmoid(model(batch["image"].to(device))[:, 0]).float().cpu().numpy()
        for k, i in enumerate(batch["index"].tolist()):
            h, w = batch["valid_hw"][k].tolist()
            out[i] = (probs[k, :h, :w].astype("float16"), batch["orig_hw"][k].tolist())
    print(f"rede: {(time.time() - t0) / len(dataset):.3f} s por imagem (com leitura do cache)")
    return out


def predict_all(probs, params):
    annotations = [to_annotation(p.astype("float32"), hw, **params) for p, hw in probs]
    return pd.DataFrame({"case_id": images["id"], "annotation": annotations})


probs = probabilities()
params_path = os.path.join(run_dir, "limiares.json")
if split == "val":
    solution = data.solution(images)
    grid = c["grid"]
    results = []
    for values in tqdm(list(itertools.product(*grid.values())), desc="busca dos limiares"):
        params = dict(zip(grid, values))
        m, _ = metrics.evaluate(solution, predict_all(probs, params))
        results.append({**params, "S": m["S"], "precisao": m["precisao"], "revocacao": m["revocacao"],
                        "oF1_detectadas": m["oF1_forjadas_detectadas"]})
    results = pd.DataFrame(results).sort_values("S", ascending=False)
    results.to_csv(os.path.join(run_dir, "busca_val.csv"), index=False)
    print(results.head(10).to_string(index=False, float_format="%.4f"))
    best = {k: float(results.iloc[0][k]) for k in grid}
    json.dump(best, open(params_path, "w"), indent=2)
else:
    best = json.load(open(params_path))                  # escolhidos na validacao, sem olhar o teste

print(f"\nlimiares: {best}")
path = os.path.join(run_dir, f"pred_{split}.csv")
predict_all(probs, best).to_csv(path, index=False)
print(f"===== {split}")
score_file(path, split)
