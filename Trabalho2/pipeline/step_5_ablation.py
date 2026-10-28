"""
Etapa 5 (Variante A): estudo de ablacao.

Compara estrategias de emparelhamento (exaustivo x sequencial, com e sem
deteccao de laco) e o numero de quadros usados (todos, 1 de cada 2, 1 de cada 4).
Cada configuracao roda o SfM do zero e passa pela mesma pos-analise (escala,
alinhamento, medidas, deriva), sem densificacao, para comparar:
imagens registradas, pontos, erro de reprojecao, deriva do laco, erro das
dimensoes e tempo.
"""
import csv
import json
import os
import time
import traceback

import numpy as np

from . import step_4_sfm
from .room_pipeline import postprocess

DEFAULT_RUNS = [
    {"name": "exaustivo", "matching": "exhaustive"},
    {"name": "sequencial", "matching": "sequential", "loop_detection": False},
    {"name": "sequencial+laco", "matching": "sequential", "loop_detection": True},
    {"name": "exaustivo_1de2", "matching": "exhaustive", "frame_step": 2},
    {"name": "exaustivo_1de4", "matching": "exhaustive", "frame_step": 4},
]


def _subset(names, step):
    sub = names[::step]
    if sub[-1] != names[-1]:
        sub.append(names[-1])  # mantem o ultimo quadro para medir a deriva do laco
    return sub


def run_ablation(image_dir, names, out_dir, cfg, log=print):
    os.makedirs(out_dir, exist_ok=True)
    sfm_cfg = cfg.get("sfm", {})
    cam_cfg = cfg.get("camera", {})
    runs = cfg.get("ablation", {}).get("runs") or DEFAULT_RUNS
    common = dict(camera_model=cam_cfg.get("model", "SIMPLE_RADIAL"), focal_px=cam_cfg.get("focal_px"),
                  max_image_size=sfm_cfg.get("max_image_size", 1600),
                  max_num_features=sfm_cfg.get("max_num_features", 8192), use_gpu=sfm_cfg.get("use_gpu", "auto"),
                  peak_threshold=sfm_cfg.get("sift_peak_threshold"), clahe=sfm_cfg.get("clahe", False),
                  colmap_exe=step_4_sfm.find_colmap(sfm_cfg.get("colmap_exe", "auto")))

    feats_db = os.path.join(out_dir, "features_todos.db")
    if not os.path.exists(feats_db):
        log("[Ablacao] extraindo features uma vez para as configuracoes com todos os quadros")
        t0 = time.time()
        step_4_sfm.extract_features(feats_db, image_dir, names, **common)
        feat_time = time.time() - t0
    else:
        feat_time = None

    rows = []
    for run in runs:
        name = run["name"]
        step = int(run.get("frame_step", 1))
        sub = names if step == 1 else _subset(list(names), step)
        log(f"[Ablacao] {name}: {len(sub)} quadros, {run['matching']}"
            + (" + laco" if run.get("loop_detection") else ""))
        run_dir = os.path.join(out_dir, name)
        t0 = time.time()
        try:
            rec, info = step_4_sfm.run_sfm(
                image_dir, os.path.join(run_dir, "sfm"), sub, run["matching"],
                run.get("overlap", sfm_cfg.get("sequential_overlap", 15)), run.get("loop_detection", False),
                sfm_cfg.get("vocab_tree"), features_db=feats_db if step == 1 else None,
                mapper=sfm_cfg.get("mapper", "incremental"), log=log, **common)
            if step == 1 and feat_time is not None:
                info["time_features_s"] = feat_time
            m = postprocess(rec, info, image_dir, sub, run_dir, cfg, do_dense=False, figures=True, log=lambda *a: None)
            total = time.time() - t0 + ((feat_time or 0) if step == 1 else 0)
            row = {"config": name, "matching": run["matching"], "loop_detection": bool(run.get("loop_detection", False)),
                   "frames": len(sub), "registered": info["n_reg_images"], "models": info["n_models"],
                   "verified_pairs": info.get("n_verified_pairs"), "points_sparse": m["n_sparse_points_clean"],
                   "reproj_mean_px": m["reproj_mean_px"], "reproj_median_px": m["reproj_median_px"],
                   "drift_cm": m["loop_drift"]["drift_cm"] if m["loop_drift"] else None,
                   "drift_pct": m["loop_drift"]["drift_pct"] if m["loop_drift"] else None,
                   "time_matching_s": info["time_matching_s"], "time_total_s": total}
            for d in m["dimensions"]:
                row[f"err_{d['name']}_cm"] = d.get("abs_err_cm")
            errs = [d.get("rel_err_pct") for d in m["dimensions"] if d.get("rel_err_pct") is not None]
            row["mean_dim_err_pct"] = float(np.mean(errs)) if errs else None
        except Exception as e:
            log(f"  FALHOU: {e}")
            traceback.print_exc()
            row = {"config": name, "matching": run["matching"], "loop_detection": bool(run.get("loop_detection", False)),
                   "frames": len(sub), "error": str(e)[:200]}
        rows.append(row)
        log("  " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, ensure_ascii=False))

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out_dir, "ablacao.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    _markdown(rows, os.path.join(out_dir, "ablacao.md"))
    _plot(rows, os.path.join(out_dir, "ablacao.png"))
    return rows


def _fmt(v, f="{:.2f}"):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (f.format(v) if isinstance(v, float) else str(v))


def _markdown(rows, path):
    dim_cols = sorted({k for r in rows for k in r if k.startswith("err_")})
    head = ["Config.", "Quadros", "Registrados", "Pontos", "Reproj. media (px)", "Deriva (cm)", "Deriva (%)"] + \
           [c[4:-3] + " (cm)" for c in dim_cols] + ["Erro medio dim. (%)", "Tempo (s)"]
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['config']} | {r['frames']} | falhou: {r['error']} |" + " |" * (len(head) - 3))
            continue
        vals = [r["config"], r["frames"], f"{r['registered']}/{r['frames']}", r["points_sparse"],
                _fmt(r["reproj_mean_px"], "{:.3f}"), _fmt(r["drift_cm"], "{:.1f}"), _fmt(r["drift_pct"])] + \
               [_fmt(r.get(c), "{:.1f}") for c in dim_cols] + [_fmt(r["mean_dim_err_pct"]), _fmt(r["time_total_s"], "{:.0f}")]
        L.append("| " + " | ".join(str(v) for v in vals) + " |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return
    names = [r["config"] for r in ok]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    series = [("drift_cm", "Deriva do laco (cm)"), ("mean_dim_err_pct", "Erro medio das dimensoes (%)"),
              ("time_total_s", "Tempo total (s)")]
    for ax, (k, lab) in zip(axes, series):
        vals = [r.get(k) if r.get(k) is not None else 0 for r in ok]
        ax.bar(names, vals, color="#4C72B0")
        ax.set_title(lab)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
