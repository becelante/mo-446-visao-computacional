"""Divide os dados em treino (70%), validacao (15%) e teste (15%) e salva splits/{train,val,test}.csv.

- Agrupado por case_id: todas as imagens de um case_id (autentica, forjada e forjada suplementar)
  ficam na mesma particao. A forjada e a autentica adulterada; separa-las seria vazamento.
- Estratificado pelo tipo de grupo (par autentica+forjada, so forjada, so forjada suplementar),
  o que mantem a proporcao de autenticas e forjadas igual nas tres particoes.
- Semente fixa (config.yaml), entao rodar de novo gera exatamente as mesmas listas.

    uv run python make_split.py
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.common import load_config
from pipeline.data import SPLITS_DIR, list_images

cfg = load_config()
images = list_images(cfg["data_dir"])


def group_type(g):
    if (g["classe"] == "authentic").any():
        return "autentica+forjada"
    return "so forjada (suplementar)" if (g["origem"] == "supplemental").all() else "so forjada"


groups = images.groupby("case_id").apply(group_type, include_groups=False).rename("tipo").reset_index()

# 70% treino; os 30% restantes divididos ao meio entre validacao e teste
train, rest = train_test_split(groups, test_size=0.30, stratify=groups["tipo"], random_state=cfg["seed"])
val, test = train_test_split(rest, test_size=0.50, stratify=rest["tipo"], random_state=cfg["seed"])

os.makedirs(SPLITS_DIR, exist_ok=True)
summary = []
for name, part in [("train", train), ("val", val), ("test", test)]:
    df = images[images["case_id"].isin(part["case_id"])].copy()
    for col in ("path", "mask_path"):                    # caminhos relativos a data/
        df[col] = [os.path.relpath(p, cfg["data_dir"]) if isinstance(p, str) else "" for p in df[col]]
    df = df.sort_values("id")[["id", "case_id", "classe", "origem", "path", "mask_path"]]
    df.to_csv(os.path.join(SPLITS_DIR, f"{name}.csv"), index=False)
    summary.append({"particao": name, "case_ids": len(part), "imagens": len(df),
                    "autenticas": (df["classe"] == "authentic").sum(), "forjadas": (df["classe"] == "forged").sum(),
                    "% forjadas": f"{(df['classe'] == 'forged').mean():.1%}",
                    "suplementares": (df["origem"] == "supplemental").sum()})

assert not (set(train["case_id"]) & set(val["case_id"]) or set(train["case_id"]) & set(test["case_id"])
            or set(val["case_id"]) & set(test["case_id"])), "case_id em mais de uma particao"
summary = pd.DataFrame(summary).set_index("particao")
summary["% das imagens"] = (summary["imagens"] / summary["imagens"].sum()).map("{:.1%}".format)
print(summary.to_string())
print(f"\nsalvo em {SPLITS_DIR}/")
