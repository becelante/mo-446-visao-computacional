"""Calcula S (metrica oficial do Kaggle) e as metricas complementares do relatorio.

    uv run python evaluate.py predicoes_teste.csv solucao_teste.csv

predicoes: case_id,annotation        solucao: case_id,annotation,shape
Grava <predicoes>_metricas.json e <predicoes>_por_imagem.csv ao lado das predicoes.
"""
import json
import sys

import pandas as pd

from pipeline import metrics

pred_csv, solution_csv = sys.argv[1], sys.argv[2]
read = lambda path: pd.read_csv(path, dtype=str, keep_default_na=False)

m, per_image = metrics.evaluate(read(solution_csv), read(pred_csv))
print(metrics.report(m))

out = pred_csv.removesuffix(".csv")
json.dump(m, open(out + "_metricas.json", "w"), indent=2, ensure_ascii=False)
per_image.to_csv(out + "_por_imagem.csv", index=False)
