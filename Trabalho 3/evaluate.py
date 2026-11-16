"""Calcula S (metrica oficial do Kaggle) e as metricas complementares do relatorio.

    uv run python evaluate.py predicoes_val.csv val

predicoes: CSV no formato de submissao (case_id,annotation), com case_id = id da imagem em splits/<particao>.csv
particao:  train, val ou test; a verdade e montada a partir das mascaras originais
Grava <predicoes>_metricas.json e <predicoes>_por_imagem.csv ao lado das predicoes.
"""
import json
import sys

import pandas as pd

from pipeline import data, metrics


def score_file(pred_csv, split):
    """Avalia um CSV de predicoes numa particao, imprime o relatorio e salva as metricas."""
    predictions = pd.read_csv(pred_csv, dtype=str, keep_default_na=False)
    m, per_image = metrics.evaluate(data.solution(data.load_split(split)), predictions)
    print(metrics.report(m))
    out = pred_csv.removesuffix(".csv")
    json.dump(m, open(out + "_metricas.json", "w"), indent=2, ensure_ascii=False)
    per_image.to_csv(out + "_por_imagem.csv", index=False)
    return m


if __name__ == "__main__":
    score_file(sys.argv[1], sys.argv[2])
