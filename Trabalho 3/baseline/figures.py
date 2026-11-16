"""Figuras de cada etapa do baseline SIFT para uma imagem da validacao, para o relatorio.

    uv run python -m baseline.figures                  # forjada da validacao com maior oF1
    uv run python -m baseline.figures 22918_forged     # uma imagem especifica (id de splits/val.csv)

Usa os parametros escolhidos na validacao (results/baseline/sift_parametros.json) e salva em
results/baseline/figuras/<id>/:
  1_pontos.png        pontos SIFT detectados
  2_pares.png         pares que passam no teste de Lowe (normais e espelhados)
  3_grupos.png        pares agrupados pelo RANSAC (uma cor por transformacao) e os descartados
  4_correlacao.png    imagem levada pela transformacao e mapa de correlacao local com o limiar
  5_mascara.png       mascara prevista ao lado da verdadeira
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from baseline import sift  # noqa: E402
from pipeline import data, recodai_f1  # noqa: E402
from pipeline.common import get_device, load_config  # noqa: E402

GROUP_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

cfg = load_config()
base_dir = os.path.join(cfg["results_dir"], "baseline")
params = json.load(open(os.path.join(base_dir, "sift_parametros.json")))
images = data.load_split("val").set_index("id")

if len(sys.argv) > 1:
    image_id = sys.argv[1]
else:
    scores = pd.read_csv(os.path.join(base_dir, "sift_val_por_imagem.csv"))
    image_id = scores[scores["forjada"]].sort_values("escore").iloc[-1]["case_id"]
row = images.loc[image_id]
out_dir = os.path.join(base_dir, "figuras", image_id)
os.makedirs(out_dir, exist_ok=True)

img = data.load_image(row["path"])
gray, scale = sift.prepare(img, params["max_side"])
bcfg = cfg["baseline_sift"]


def figure(title):
    h, w = gray.shape
    fig, ax = plt.subplots(figsize=(8, 8 * h / w + 0.6))
    ax.imshow(gray, cmap="gray")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def lines(ax, P, Q, color, lw=0.8, alpha=0.8, max_lines=60):
    """Todos os pontos, mas so ate max_lines linhas (sorteadas) para a figura continuar legivel."""
    ax.scatter(*np.concatenate([P, Q]).T, s=4, color=color, alpha=alpha)
    if len(P) > max_lines:
        pick = np.random.default_rng(0).choice(len(P), max_lines, replace=False)
        P, Q = P[pick], Q[pick]
    ax.add_collection(LineCollection(np.stack([P, Q], 1), colors=color, linewidths=lw, alpha=alpha))


# 1. pontos SIFT
pts, _ = sift.sift_features(gray, bcfg["n_features"])
fig, ax = figure(f"1. Pontos SIFT detectados: {len(pts)}")
ax.scatter(pts[:, 0], pts[:, 1], s=3, color="#ffcc00", alpha=0.7)
save(fig, "1_pontos.png")

# 2. pares que passam no teste de Lowe
P, Q, ratio, mirrored = sift.self_matches(gray, bcfg["n_features"], bcfg["min_dist"], get_device())
keep = ratio < params["ratio"]
fig, ax = figure(f"2. Pares com razão de Lowe < {params['ratio']}: {keep.sum()} de {len(P)} candidatos\n"
                 f"(azul: cópia direta, laranja: cópia espelhada; no máximo 60 linhas desenhadas por cor)")
lines(ax, P[keep & ~mirrored], Q[keep & ~mirrored], "#4ea8ff")
lines(ax, P[keep & mirrored], Q[keep & mirrored], "#ff9f1c")
save(fig, "2_pares.png")

# 3. grupos do RANSAC
groups = sift.find_groups(P[keep], Q[keep], params["min_inliers"], params["ransac_px"])
fig, ax = figure(f"3. RANSAC: {len(groups)} transformação(ões) com pelo menos {params['min_inliers']} pares\n"
                 "(cinza: pares descartados; no máximo 60 linhas desenhadas por cor)")
lines(ax, P[keep], Q[keep], "#9a9a9a", lw=0.5, alpha=0.5)
for k, (T, src, dst) in enumerate(groups):
    lines(ax, src, dst, GROUP_COLORS[k % len(GROUP_COLORS)], lw=1.2, alpha=0.95)
save(fig, "3_grupos.png")

# 4. correlacao local com a transformacao do maior grupo
if groups:
    T, src, _ = groups[0]
    moved, zncc, valid = sift.similarity(gray, T)
    zncc_show = np.where(valid, zncc, np.nan)
    h, w = gray.shape
    fig, axs = plt.subplots(1, 2, figsize=(14, 7 * h / w + 1.2), layout="constrained")
    axs[0].imshow(moved, cmap="gray")
    axs[0].set_title("4a. Imagem levada pela transformação do grupo 1\n(onde há cópia, fica igual à original)", fontsize=11)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("lightgray")                            # sem textura ou fora da imagem
    im = axs[1].imshow(zncc_show, cmap=cmap, vmin=-1, vmax=1)
    axs[1].contour(np.nan_to_num(zncc_show, nan=-1) > params["corr_thr"], levels=[0.5], colors="red", linewidths=1)
    axs[1].set_title(f"4b. Correlação local entre as duas (vermelho: > {params['corr_thr']})\n"
                     "cinza: sem textura ou fora da imagem", fontsize=11)
    fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, shrink=0.5, label="correlação local")
    for a in axs:
        a.axis("off")
    fig.savefig(os.path.join(out_dir, "4_correlacao.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# 5. mascara prevista x verdadeira
masks = sift.predict(gray, P, Q, ratio, img.shape[:2], params)
truth = data.load_masks(row["mask_path"], img.shape[:2]) if row["classe"] == "forged" else []
score = (recodai_f1.oF1_score(masks, [t.astype(np.uint8) for t in truth]) if masks and len(truth)
         else float(not masks and not len(truth)))


def paint(masks_):
    out = img.astype(float).copy()
    for k, m in enumerate(masks_):
        c = np.array(matplotlib.colors.to_rgb(GROUP_COLORS[k % len(GROUP_COLORS)])) * 255
        out[m > 0] = 0.45 * out[m > 0] + 0.55 * c
    return out.astype(np.uint8)


h, w = img.shape[:2]
fig, axs = plt.subplots(1, 2, figsize=(14, 7 * h / w + 0.8))
axs[0].imshow(paint(masks))
axs[0].set_title(f"5a. Previsto: {len(masks)} instância(s)" if masks else "5a. Previsto: authentic", fontsize=11)
axs[1].imshow(paint(truth))
axs[1].set_title(f"5b. Verdade: {len(truth)} instância(s)   |   escore da imagem = {score:.2f}", fontsize=11)
for a in axs:
    a.axis("off")
save(fig, "5_mascara.png")
print(f"figuras de {image_id} em {out_dir}")
