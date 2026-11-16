"""Leitura dos dados da competicao, particoes (splits) e solucao no formato da metrica.

Estrutura em data/:
    train_images/authentic/<case_id>.png      imagens autenticas
    train_images/forged/<case_id>.png         imagens forjadas
    train_masks/<case_id>.npy                 mascaras das forjadas, shape (N, H, W)
    supplemental_images/<case_id>.png         so forjadas, mascaras em supplemental_masks/

Cada mascara e uma instancia: uma origem e todas as copias dela.
Um case_id pode ter ate 3 imagens (autentica, forjada e forjada suplementar), entao cada imagem
tem um identificador proprio, `id`, usado nas predicoes e na metrica.
"""
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image

from pipeline.common import ROOT, load_config
from pipeline.metrics import masks_to_annotation

SPLITS_DIR = os.path.join(ROOT, "splits")


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_masks(path, shape):
    """Mascaras (N, H, W) bool de um .npy. Aceita tambem (H, W) e (H, W, N)."""
    m = np.load(path)
    if m.ndim == 2:
        m = m[None]
    if m.shape[1:] != tuple(shape) and m.shape[:2] == tuple(shape):
        m = m.transpose(2, 0, 1)
    return m > 0


def list_images(data_dir):
    """Uma linha por imagem: id, case_id, classe, origem (train/supplemental) e caminhos."""
    rows = []
    for source in ("train", "supplemental"):
        mask_dir = os.path.join(data_dir, f"{source}_masks")
        for path in sorted(glob(os.path.join(data_dir, f"{source}_images", "**", "*.png"), recursive=True)):
            case_id = os.path.splitext(os.path.basename(path))[0]
            folder = os.path.basename(os.path.dirname(path))
            mask_path = os.path.join(mask_dir, case_id + ".npy")
            if folder in ("authentic", "forged"):
                label = folder
            else:                                   # pasta sem subdivisao: classe pela existencia da mascara
                label = "forged" if os.path.exists(mask_path) else "authentic"
            image_id = f"{case_id}_{label}" + ("_supl" if source == "supplemental" else "")
            rows.append({"id": image_id, "case_id": case_id, "classe": label, "origem": source, "path": path,
                         "mask_path": mask_path if label == "forged" else None})
    return pd.DataFrame(rows)


def load_split(name, data_dir=None):
    """Imagens de uma particao (train, val ou test), com caminhos absolutos."""
    data_dir = data_dir or load_config()["data_dir"]
    df = pd.read_csv(os.path.join(SPLITS_DIR, f"{name}.csv"), dtype=str, keep_default_na=False)
    df["path"] = [os.path.join(data_dir, p) for p in df["path"]]
    df["mask_path"] = [os.path.join(data_dir, p) if p else None for p in df["mask_path"]]
    return df


def solution(images):
    """Verdade no formato da metrica oficial: case_id (= id da imagem), annotation, shape."""
    rows = []
    for _, r in images.iterrows():
        w, h = Image.open(r["path"]).size
        annotation = "authentic"
        if r["classe"] == "forged":
            annotation = masks_to_annotation(load_masks(r["mask_path"], (h, w)))
        rows.append({"case_id": r["id"], "annotation": annotation,
                     "shape": "authentic" if annotation == "authentic" else f"[{h}, {w}]"})
    return pd.DataFrame(rows)
