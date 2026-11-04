"""
Figuras e tabelas das Etapas 2 e 3 a partir dos quadros-chave (modulos step_1, step_2, step_3):

  - keypoints de algumas imagens (SIFT padrao, SIFT de limiar baixo e ORB, com escala/orientacao);
  - comparacao dos detectores em pares de quadros-chave vizinhos: keypoints, tempo, aprovados no
    ratio test e na verificacao cruzada, inliers de E (RANSAC, 4 px) e pares com geometria
    (>= 15 inliers; pares sem geometria contam como zero). O SIFT do COLMAP, o do pipeline,
    entra com os keypoints e o tempo da extracao;
  - matches antes/depois, linhas epipolares e triangulacao de duas vistas do par com mais
    inliers (SIFT de limiar baixo);
  - matriz de conectividade (inliers de E) num subconjunto dos quadros.
"""
import csv
import os
import time

import cv2
import matplotlib
import numpy as np

from . import step_1_keypoints_detector as S1
from . import step_2_matching as S2
from . import step_3_epipolar as S3

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ("SIFT", "SIFT_BAIXO", "ORB")
LABELS = {"SIFT": "SIFT padrao", "SIFT_BAIXO": "SIFT limiar baixo", "ORB": "ORB"}

def compare_detectors(load, names, K, colmap_sift, n_pairs, pair_gap, out_dir, log):
    idx = np.linspace(0, len(names) - 1 - pair_gap, min(n_pairs, len(names) - pair_gap)).astype(int)
    rows, feats = [], {}
    for m in METHODS:
        for i in sorted(set(idx) | set(idx + pair_gap)):
            img = load(names[i])
            t0 = time.time()
            kps, desc = S1.detect_features(img, m)
            feats[(m, i)] = (kps, desc, time.time() - t0)
        for i in idx:
            kp1, d1, t1 = feats[(m, i)]
            kp2, d2, _ = feats[(m, i + pair_gap)]
            g = S3.two_view_geometry(kp1, d1, kp2, d2, K, m)
            rows.append({"method": m, "pair": f"{names[i]}-{names[i + pair_gap]}", "pair_idx": int(i),
                         "kps": len(kp1), "t_detect_s": t1,
                         "ratio_pass": sum(a.distance < 0.75 * b.distance for a, b in S2.knn_match(d1, d2, m)),
                         "cross_pass": g["metrics"]["n_matches"], "inliers_E": g["metrics"]["inliers_E"]})
    with open(os.path.join(out_dir, "comparacao_detectores.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    cols = [LABELS[m] for m in METHODS] + ["SIFT COLMAP"]
    md = [f"{len(idx)} pares de quadros-chave a {pair_gap} de distancia.", "",
          "| Metrica (media por par) | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    by = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    mean = lambda m, k: np.mean([r[k] for r in by[m]])
    lines = [("Keypoints por imagem", lambda m: f"{mean(m, 'kps'):.0f}", f"{colmap_sift['kps']:.0f}"),
             ("Tempo de deteccao por imagem (s)", lambda m: f"{mean(m, 't_detect_s'):.3f}", f"{colmap_sift['t']:.3f}"),
             ("Pares com geometria (>= 15 inliers)", lambda m: f"{sum(r['inliers_E'] >= 15 for r in by[m])} / {len(idx)}", "-"),
             ("Aprovados no ratio test", lambda m: f"{mean(m, 'ratio_pass'):.0f}", "-"),
             ("Aprovados na verificacao cruzada", lambda m: f"{mean(m, 'cross_pass'):.0f}", "-"),
             ("Inliers de E (RANSAC, 4 px)", lambda m: f"{mean(m, 'inliers_E'):.0f}", "-"),
             ("Inliers / aprovados no ratio test (total)",
              lambda m: f"{sum(r['inliers_E'] for r in by[m]) / max(1, sum(r['ratio_pass'] for r in by[m])):.2f}", "-")]
    for label, fn, colmap in lines:
        md.append(f"| {label} | " + " | ".join(fn(m) for m in METHODS) + f" | {colmap} |")
    with open(os.path.join(out_dir, "comparacao_detectores.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    log("  comparacao de detectores:\n    " + "\n    ".join(md))
    best = max(by["SIFT_BAIXO"], key=lambda r: r["inliers_E"])["pair_idx"]
    return best, feats

def detail_pair(load, names, i, j, K, feats, out_dir):
    img1, img2 = load(names[i]), load(names[j])
    kp1, d1, _ = feats[("SIFT_BAIXO", i)]
    kp2, d2, _ = feats[("SIFT_BAIXO", j)]
    g = S3.two_view_geometry(kp1, d1, kp2, d2, K, "SIFT_BAIXO")
    raw = [p[0] for p in S2.knn_match(d1, d2, "SIFT_BAIXO")]
    S2.draw_matches_before_after(img1, kp1, img2, kp2, raw, g["good"], g["mask_E"], os.path.join(out_dir, "matches"))
    if g["E"] is None:
        return
    pts1, pts2 = S3.matched_points(kp1, kp2, g["good"])
    S3.draw_epipolar_lines(img1, img2, pts1[g["mask_E"]], pts2[g["mask_E"]], S3.fundamental_from_essential(g["E"], K),
                           os.path.join(out_dir, "linhas_epipolares.jpg"))
    if g["points3d"] is not None and len(g["points3d"]):
        p1, p2 = pts1[g["mask_pose"]], pts2[g["mask_pose"]]
        valid = S3.triangulate(p1, p2, K, g["R"], g["t"])[1]
        cols = np.array([img1[int(y), int(x)] for x, y in p1])[valid]
        S3.save_point_cloud(g["points3d"], os.path.join(out_dir, "duas_vistas.ply"), cols)

def connectivity(load, names, K, conn_max, out_dir):
    sub = np.linspace(0, len(names) - 1, min(conn_max, len(names))).astype(int)
    kps_all, descs_all = S1.detect_features_all([load(names[k]) for k in sub], "SIFT_BAIXO")
    conn = S3.build_connectivity_matrix(kps_all, descs_all, K, "SIFT_BAIXO")
    np.savetxt(os.path.join(out_dir, "conectividade.csv"), conn, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(conn, cmap="magma")
    fig.colorbar(im, ax=ax, label="inliers de E")
    ax.set_title(f"Conectividade entre {len(sub)} quadros (SIFT limiar baixo)")
    ax.set_xlabel("quadro")
    ax.set_ylabel("quadro")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conectividade.png"), dpi=160)
    plt.close(fig)

def run(image_dir, names, out_dir, K, colmap_sift, n_pairs=30, pair_gap=1, conn_max=40, log=print):
    os.makedirs(out_dir, exist_ok=True)
    load = lambda n: cv2.imread(os.path.join(image_dir, n))
    for n in [names[i] for i in np.linspace(0, len(names) - 1, 3).astype(int)]:
        img = load(n)
        for m in METHODS:
            S1.draw_keypoints(img, S1.detect_features(img, m)[0],
                              os.path.join(out_dir, f"keypoints_{m}_{os.path.splitext(n)[0]}.jpg"))
    best, feats = compare_detectors(load, names, K, colmap_sift, n_pairs, pair_gap, out_dir, log)
    detail_pair(load, names, best, best + pair_gap, K, feats, out_dir)
    connectivity(load, names, K, conn_max, out_dir)
