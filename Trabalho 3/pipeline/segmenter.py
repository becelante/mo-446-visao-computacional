"""Rede de segmentacao (passo 1): U-Net com encoder pre-treinado no ImageNet, em PyTorch Lightning.

A rede devolve, para cada pixel, a probabilidade de ele fazer parte de uma regiao copiada
(origem ou copia). Para virar uma predicao no formato da competicao (to_annotation):
  p = maior probabilidade da imagem      (probabilidade de a imagem ser forjada)
  p < p_thr                              -> "authentic"
  senao mascara = probabilidade > pix_thr, na resolucao original, como UMA instancia
  mascara menor que min_area da imagem   -> "authentic"
"""
import cv2
import lightning as L
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from pipeline.dataset import MEAN, STD
from pipeline.metrics import evaluate, masks_to_annotation


def to_annotation(prob, orig_hw, pix_thr, p_thr, min_area):
    """prob: (h, w) float, so a parte util (sem preenchimento). Devolve a anotacao oficial."""
    if prob.max() < p_thr:
        return "authentic"
    H, W = orig_hw
    mask = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR) > pix_thr
    if mask.mean() < min_area:
        return "authentic"
    return masks_to_annotation([mask])


def denormalize(img):
    return (img * torch.tensor(STD, device=img.device)[:, None, None]
            + torch.tensor(MEAN, device=img.device)[:, None, None]).clamp(0, 1)


class Segmenter(L.LightningModule):
    """cfg: secao `unet` do config.yaml. solution: verdade da validacao (data.solution), para o S por epoca.
    val_ids: id de cada imagem da validacao, na ordem do dataset. show: indices de exemplos para o TensorBoard."""

    def __init__(self, cfg, solution=None, val_ids=None, show=()):
        super().__init__()
        self.save_hyperparameters("cfg")                   # vai junto do checkpoint
        self.cfg = cfg
        self.net = smp.Unet(cfg["encoder"], encoder_weights="imagenet", classes=1)
        self.dice = smp.losses.DiceLoss("binary", from_logits=True)
        self.solution, self.val_ids, self.show = solution, val_ids, set(show)
        self.val_preds, self.val_examples = {}, []

    def forward(self, x):
        return self.net(x)

    def loss(self, logits, mask):
        """BCE pixel a pixel + Dice (o Dice compensa as regioes copiadas serem pequenas)."""
        return F.binary_cross_entropy_with_logits(logits, mask) + self.dice(logits, mask)

    def training_step(self, batch, _):
        loss = self.loss(self(batch["image"]), batch["mask"])
        self.log("treino/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch["image"])
        self.log("val/loss", self.loss(logits, batch["mask"]), on_epoch=True, prog_bar=True)
        probs = torch.sigmoid(logits[:, 0]).float()
        c = self.cfg
        for k, i in enumerate(batch["index"].tolist()):
            h, w = batch["valid_hw"][k].tolist()
            prob = probs[k, :h, :w].cpu().numpy()
            self.val_preds[self.val_ids[i]] = to_annotation(prob, batch["orig_hw"][k].tolist(),
                                                            c["pix_thr"], c["p_thr"], c["min_area"])
            if i in self.show:
                self.val_examples.append((i, denormalize(batch["image"][k]),
                                          batch["mask"][k].expand(3, -1, -1), probs[k][None].expand(3, -1, -1)))

    def on_validation_epoch_end(self):
        preds, examples = self.val_preds, self.val_examples
        self.val_preds, self.val_examples = {}, []
        if self.trainer.sanity_checking or self.solution is None:
            return
        submission = pd.DataFrame({"case_id": list(preds), "annotation": list(preds.values())})
        m, _ = evaluate(self.solution, submission)
        self.log_dict({"val/S": m["S"], "val/precisao": m["precisao"], "val/revocacao": m["revocacao"],
                       "val/oF1_detectadas": np.nan_to_num(m["oF1_forjadas_detectadas"])}, prog_bar=False)
        self.log("val_S", m["S"], prog_bar=True, logger=False)   # nome sem "/" para o checkpoint
        if examples and self.logger:
            # uma linha por exemplo: imagem | verdade | probabilidade prevista
            tiles = [t for _, *row in sorted(examples, key=lambda e: e[0]) for t in row]
            tiles = [F.interpolate(t[None], size=256, mode="bilinear")[0] for t in tiles]
            self.logger.experiment.add_image("val/exemplos", make_grid(tiles, nrow=3), self.current_epoch)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sched]
