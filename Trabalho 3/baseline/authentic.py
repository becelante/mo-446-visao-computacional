"""Baseline obrigatorio: prever "authentic" para todas as imagens.

    uv run python -m baseline.authentic

Nao tem nenhum parametro, entao roda-lo no teste nao ajusta nada: o escore e so a fracao de
autenticas da particao. Salva results/baseline/authentic_{val,test}.csv e as metricas.
"""
import os

import pandas as pd

from evaluate import score_file
from pipeline import data
from pipeline.common import load_config

out_dir = os.path.join(load_config()["results_dir"], "baseline")
os.makedirs(out_dir, exist_ok=True)
for split in ("val", "test"):
    images = data.load_split(split)
    path = os.path.join(out_dir, f"authentic_{split}.csv")
    pd.DataFrame({"case_id": images["id"], "annotation": "authentic"}).to_csv(path, index=False)
    print(f"===== {split}")
    score_file(path, split)
