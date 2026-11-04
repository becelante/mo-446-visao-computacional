"""
Etapa 5 (Variante A): estudo de ablacao. Cada configuracao refaz o SfM do zero (mapeador
global) e passa pela mesma pos-analise (escala, alinhamento, medidas, deriva), sem densa:

  - emparelhamento, nos quadros-chave: exaustivo x sequencial, sem e com deteccao de laco
    (o exaustivo repetido mostra tambem a variacao entre execucoes);
  - numero de quadros: N quadros espalhados uniformemente pelo video (o primeiro e o ultimo
    ficam, para a deriva do laco), com emparelhamento exaustivo.

O erro medio das dimensoes usa so as medidas automaticas (largura e comprimento): os pontos
marcados a mao nao estao em todos os conjuntos de quadros.
"""
import csv
import json
import os
import time

import matplotlib
import numpy as np
import pycolmap

from . import config, step_0_video_frames, step_4_sfm
from .room_pipeline import postprocess

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MATCHING_RUNS = [("exaustivo", "exhaustive", False), ("sequencial", "sequential", False),
                 ("seq. + laço", "sequential", True)]
BLUE, ORANGE, GRAY = "#2b7bd6", "#e8622c", "#ecede8"

def _sfm(cfg, image_dir, names, work_dir, matching, loop, features_db, log):
    info_path = os.path.join(work_dir, "sfm_info.json")
    if os.path.exists(info_path):
        return pycolmap.Reconstruction(os.path.join(work_dir, "sparse")), json.load(open(info_path))
    sfm_cfg, cam_cfg = cfg["sfm"], cfg["camera"]
    t0 = time.time()
    rec, info = step_4_sfm.run_sfm(
        image_dir, work_dir, names, matching, sfm_cfg["sequential_overlap"], loop, sfm_cfg["vocab_tree"],
        cam_cfg["model"], cam_cfg["focal_px"], sfm_cfg["max_image_size"], sfm_cfg["max_num_features"],
        sfm_cfg["use_gpu"], features_db=features_db, peak_threshold=sfm_cfg["sift_peak_threshold"],
        clahe=sfm_cfg["clahe"], mapper=sfm_cfg["mapper"], colmap_exe=step_4_sfm.find_colmap(sfm_cfg["colmap_exe"]),
        log=log)
    info["time_sfm_s"] = time.time() - t0
    json.dump(info, open(info_path, "w"), indent=1)
    os.remove(os.path.join(work_dir, "database.db"))  # o banco do exaustivo passa de 1 GB
    return rec, info

def _evaluate(cfg, image_dir, names, run_dir, matching, loop, features_db, log):
    rec, info = _sfm(cfg, image_dir, names, os.path.join(run_dir, "sfm"), matching, loop, features_db, log)
    m = postprocess(rec, info, image_dir, names, run_dir, dict(cfg, manual_points=None), do_dense=False,
                    figures=False, log=lambda *a: None)
    metric = m["metric"]
    auto = [d for d in m["dimensions"] if d["source"].startswith("auto:")]
    row = {"frames": len(names), "registered": info["n_reg_images"], "models": info["n_models"],
           "points_sparse": m["n_sparse_points_clean"], "reproj_mean_px": m["reproj_mean_px"], "metric": metric,
           "drift_cm": m["loop_drift"]["drift_cm"] if metric and m["loop_drift"] else None,
           "drift_pct": m["loop_drift"]["drift_pct"] if metric and m["loop_drift"] else None,
           "mean_dim_err_pct": float(np.mean([d["rel_err_pct"] for d in auto])) if metric else None,
           "time_matching_s": info["time_matching_s"], "time_sfm_s": info["time_sfm_s"]}
    for d in auto:
        row[f"err_{d['name']}_cm"] = d.get("abs_err_cm")
    log("  " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, ensure_ascii=False))
    return row

def run_ablation(image_dir, names, out_dir, cfg, log=print):
    os.makedirs(out_dir, exist_ok=True)
    feats_db = os.path.join(out_dir, "features_quadros_chave.db")
    if not os.path.exists(feats_db):
        log("[Ablacao] extraindo features dos quadros-chave uma vez para as tres estrategias")
        sfm_cfg, cam_cfg = cfg["sfm"], cfg["camera"]
        step_4_sfm.extract_features(feats_db, image_dir, names, cam_cfg["model"], cam_cfg["focal_px"],
                                    sfm_cfg["max_image_size"], sfm_cfg["max_num_features"], sfm_cfg["use_gpu"],
                                    sfm_cfg["sift_peak_threshold"], sfm_cfg["clahe"],
                                    step_4_sfm.find_colmap(sfm_cfg["colmap_exe"]))
    matching_rows = []
    for name, matching, loop in MATCHING_RUNS:
        log(f"[Ablacao] {name}: {len(names)} quadros-chave")
        row = _evaluate(cfg, image_dir, names, os.path.join(out_dir, name.replace(" ", "").replace("ç", "c")), matching, loop,
                        feats_db, log)
        matching_rows.append({"config": name, **row})

    uniform_dir = os.path.join(out_dir, "quadros_uniformes")
    n_max = max(cfg["ablation"]["n_frames"])
    all_uniform = step_0_video_frames.extract_uniform_frames(config.video_path(cfg), uniform_dir, n_max,
                                                             cfg["frames"]["max_dim"])
    frame_rows = []
    for n in cfg["ablation"]["n_frames"]:
        sub = [all_uniform[k] for k in np.unique(np.linspace(0, len(all_uniform) - 1, n).round().astype(int))]
        log(f"[Ablacao] {len(sub)} quadros uniformes, exaustivo")
        frame_rows.append(_evaluate(cfg, uniform_dir, sub, os.path.join(out_dir, f"uniforme_{n}"), "exhaustive",
                                    False, None, log))

    for rows, path in ((matching_rows, "emparelhamento.csv"), (frame_rows, "quadros.csv")):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out_dir, path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    _markdown(matching_rows, frame_rows, os.path.join(out_dir, "ablacao.md"))
    _plot_matching(matching_rows, os.path.join(out_dir, "ablacao_emparelhamento.png"))
    _plot_frames(frame_rows, os.path.join(out_dir, "ablacao_quadros.png"))

def _fmt(v, f="{:.1f}"):
    return "sem escala" if v is None else f.format(v)

def _markdown(matching_rows, frame_rows, path):
    dims = [k for k in matching_rows[0] if k.startswith("err_")]
    head = ["Config.", "Registradas", "Modelos", "Pontos", "Reproj. (px)", "Deriva (cm)", "Deriva (%)"] + \
           [f"erro {k[4:-3]} (cm)" for k in dims] + ["Erro medio dim. (%)", "Emparelhamento (s)", "SfM (s)"]
    L = ["## Emparelhamento (quadros-chave)", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in matching_rows:
        vals = [r["config"], f"{r['registered']}/{r['frames']}", r["models"], r["points_sparse"],
                f"{r['reproj_mean_px']:.3f}", _fmt(r["drift_cm"]), _fmt(r["drift_pct"], "{:.3f}")] + \
               [_fmt(r.get(k)) for k in dims] + [_fmt(r["mean_dim_err_pct"]), f"{r['time_matching_s']:.0f}",
                                                 f"{r['time_sfm_s']:.0f}"]
        L.append("| " + " | ".join(str(v) for v in vals) + " |")
    head = ["Quadros", "Registradas", "Modelos", "Pontos", "Reproj. (px)", "Deriva (cm)"] + \
           [f"erro {k[4:-3]} (cm)" for k in dims] + ["SfM (min)"]
    L += ["", "## Numero de quadros (uniformes no video, exaustivo)", "", "| " + " | ".join(head) + " |",
          "|" + "---|" * len(head)]
    for r in frame_rows:
        vals = [r["frames"], f"{r['registered']} ({100 * r['registered'] / r['frames']:.0f}%)", r["models"],
                r["points_sparse"], f"{r['reproj_mean_px']:.3f}", _fmt(r["drift_cm"])] + \
               [_fmt(r.get(k)) for k in dims] + [f"{r['time_sfm_s'] / 60:.1f}"]
        L.append("| " + " | ".join(str(v) for v in vals) + " |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")

def _style(ax, title):
    ax.set_title(title, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5")
    ax.set_axisbelow(True)

def _plot_matching(rows, path):
    names = [r["config"] for r in rows]
    panels = [("registered", "Câmeras registradas", "{:.0f}"), ("mean_dim_err_pct", "Erro médio dim. (%)", "{:.1f}"),
              ("drift_cm", "Deriva do laço (cm)", "{:.1f}"), ("time_matching_s", "Emparelhamento (s)", "{:.0f}")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (k, title, fmt) in zip(axes, panels):
        vals = [r[k] for r in rows]
        shown = sorted(v for v in vals if v is not None)
        top = max(shown or [1])
        if len(shown) > 1 and shown[-1] > 3 * shown[-2]:  # barra fora de escala: cortada, com o valor escrito
            top = 1.6 * shown[-2]
        ax.bar(names, [min(v or 0, top) for v in vals], color=BLUE, width=0.6)
        for i, v in enumerate(vals):
            label = "sem\nescala" if v is None else fmt.format(v) + (" ↑" if v > top else "")
            ax.text(i, min(v or 0, top) + 0.02 * top, label, ha="center", va="bottom", fontsize=10,
                    color="black" if v is not None else "#555")
        ax.set_ylim(0, 1.2 * top)
        ax.tick_params(axis="x", rotation=20)
        _style(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _plot_frames(rows, path):
    n = [r["frames"] for r in rows]
    no_scale = [r["frames"] for r in rows if not r["metric"]]
    dims = [k for k in rows[0] if k.startswith("err_")]
    panels = [("Câmeras registradas (%)", [("", [100 * r["registered"] / r["frames"] for r in rows])]),
              ("Erro de reprojeção médio (px)", [("", [r["reproj_mean_px"] for r in rows])]),
              ("Erro das dimensões $|\\hat{L} - L|$ (cm)", [(k[4:-3], [r.get(k) for r in rows]) for k in dims]),
              ("Deriva do laço (cm)", [("", [r["drift_cm"] for r in rows])]),
              ("Pontos esparsos (mil)", [("", [r["points_sparse"] / 1000 for r in rows])]),
              ("Tempo do SfM (min)", [("", [r["time_sfm_s"] / 60 for r in rows])])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, series) in zip(axes.ravel(), panels):
        if no_scale and title.startswith(("Erro das", "Deriva")):
            ax.axvspan(min(n) - 0.05 * max(n), max(no_scale) + 0.05 * max(n), color=GRAY)
            ax.text((min(n) + max(no_scale)) / 2, 0.5, "sem escala\nmétrica", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", color="#555")
        for (label, vals), color in zip(series, (BLUE, ORANGE)):
            pts = [(x, v) for x, v in zip(n, vals) if v is not None]
            if pts:
                ax.plot(*zip(*pts), "-o", color=color, lw=2.5, ms=6, mec="white")
                if label:
                    ax.annotate(label, pts[-1], xytext=(8, 0), textcoords="offset points", va="center")
        ax.set_xticks(n)
        ax.set_ylim(bottom=0)
        _style(ax, title)
    for ax in axes[1]:
        ax.set_xlabel("quadros usados")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
