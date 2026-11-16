"""Dataset para treinar a rede de segmentacao (passo 0).

Cada imagem e redimensionada para que o lado maior tenha `size` px (mantendo a proporcao) e
completada com zeros ate size x size. Assim a imagem inteira entra na rede e o par origem/copia
nunca e cortado. O alvo e a uniao de todas as instancias em um canal so (0 = autentico, 1 = copiado).

Ler e redimensionar PNGs grandes (ate 14 MP) a cada epoca seria lento, entao a versao reduzida
de cada imagem fica salva em results/cache/<size>/<id>.npz na primeira leitura.
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.data import load_image, load_masks

MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # normalizacao do ImageNet (encoder pre-treinado)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def resize_and_pad(img, mask, size):
    """Lado maior -> size px, depois zeros a direita e embaixo ate size x size.
    Devolve (imagem, mascara, (h, w) da parte util)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    out_img = np.zeros((size, size, 3), np.uint8)
    out_img[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    out_mask = np.zeros((size, size), np.uint8)
    out_mask[:nh, :nw] = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    return out_img, out_mask, (nh, nw)


class ForgeryDataset(Dataset):
    """images: DataFrame de data.load_split. Cada item e um dicionario:
        image     (3, size, size) float, normalizada
        mask      (1, size, size) float, uniao das instancias
        valid_hw  (h, w) da parte util (sem o preenchimento)
        orig_hw   (H, W) da imagem original
        index     posicao da linha em `images`
    """

    def __init__(self, images, size, cache_dir):
        self.images = images.reset_index(drop=True)
        self.size = size
        self.cache_dir = os.path.join(cache_dir, str(size))
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.images)

    def load(self, i):
        """Imagem reduzida, mascara reduzida, parte util e tamanho original (do cache se ja existir)."""
        row = self.images.iloc[i]
        path = os.path.join(self.cache_dir, row["id"] + ".npz")
        if not os.path.exists(path):
            img = load_image(row["path"])
            h, w = img.shape[:2]
            mask = load_masks(row["mask_path"], (h, w)).any(0) if row["classe"] == "forged" else np.zeros((h, w), bool)
            small_img, small_mask, valid_hw = resize_and_pad(img, mask, self.size)
            with open(path + ".tmp", "wb") as f:           # escreve e so depois renomeia: se o treino
                np.savez(f, image=small_img, mask=small_mask, valid_hw=valid_hw, orig_hw=(h, w))
            os.replace(path + ".tmp", path)                 # for interrompido, nao fica arquivo pela metade
        d = np.load(path)
        return d["image"], d["mask"], tuple(d["valid_hw"]), tuple(d["orig_hw"])

    def __getitem__(self, i):
        img, mask, valid_hw, orig_hw = self.load(i)
        img = (img.astype(np.float32) / 255 - MEAN) / STD
        return {"image": torch.from_numpy(img.transpose(2, 0, 1)),
                "mask": torch.from_numpy(mask[None].astype(np.float32)),
                "valid_hw": torch.tensor(valid_hw), "orig_hw": torch.tensor(orig_hw), "index": i}
