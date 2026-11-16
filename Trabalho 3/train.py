"""Treina a rede de segmentacao no split de treino, escolhendo a melhor epoca pelo S da validacao.

    uv run python train.py --name unet_base

Salva em results/unet/<nome>/:
  checkpoints/melhor.ckpt   pesos da epoca com maior S na validacao
  events.out.tfevents.*     curvas e exemplos para o TensorBoard
Para acompanhar durante o treino:  uv run tensorboard --logdir results/unet   (abrir http://localhost:6006)
"""
import argparse
import os
from datetime import datetime

import lightning as L
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from pipeline import data
from pipeline.common import get_device, load_config
from pipeline.dataset import ForgeryDataset
from pipeline.segmenter import Segmenter

parser = argparse.ArgumentParser()
parser.add_argument("--name", default=datetime.now().strftime("unet_%Y%m%d-%H%M"), help="nome do experimento")
args = parser.parse_args()

cfg = load_config()
c = cfg["unet"]
L.seed_everything(cfg["seed"], workers=True)
get_device()                                             # liga cudnn.benchmark e TF32
cache_dir = os.path.join(cfg["results_dir"], "cache")

train_images, val_images = data.load_split("train"), data.load_split("val")
train_set = ForgeryDataset(train_images, c["size"], cache_dir)
val_set = ForgeryDataset(val_images, c["size"], cache_dir)
loader = lambda ds, shuffle: DataLoader(ds, batch_size=c["batch_size"], shuffle=shuffle, num_workers=c["num_workers"],
                                        pin_memory=True, persistent_workers=True, drop_last=shuffle)

# 4 forjadas fixas da validacao para ver a predicao evoluir no TensorBoard
forged = np.flatnonzero(val_images["classe"] == "forged")
show = np.random.default_rng(0).choice(forged, 4, replace=False).tolist()

print("montando a verdade da validacao...")
model = Segmenter(c, solution=data.solution(val_images), val_ids=list(val_images["id"]), show=show)

logger = TensorBoardLogger(os.path.join(cfg["results_dir"], "unet"), name=args.name, version="",
                           default_hp_metric=False)
trainer = L.Trainer(
    max_epochs=c["epochs"],
    precision="16-mixed" if cfg["amp"] else "32-true",
    logger=logger,
    callbacks=[ModelCheckpoint(monitor="val_S", mode="max", filename="melhor"),
               EarlyStopping(monitor="val_S", mode="max", patience=c["patience"]),
               LearningRateMonitor()],
    log_every_n_steps=20,
)
trainer.fit(model, loader(train_set, True), loader(val_set, False))
print(f"melhor S na validacao: {trainer.checkpoint_callback.best_model_score:.4f}")
print(f"checkpoint: {trainer.checkpoint_callback.best_model_path}")
