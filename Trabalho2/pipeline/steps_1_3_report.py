"""
Figuras e tabelas das Etapas 2 e 3 geradas a partir dos quadros do video, usando
os modulos ja implementados (step_1, step_2, step_3):

  - keypoints de algumas imagens (SIFT e ORB, com escala/orientacao);
  - comparacao SIFT x ORB em pares consecutivos (keypoints, tempo, matches,
    aprovados no ratio test, inliers de E, erro de reprojecao);
  - matches antes/depois do ratio test + RANSAC, linhas epipolares e nuvem
    triangulada de duas vistas para um par;
  - matriz de conectividade (inliers de E) num subconjunto dos quadros;
  - (se houver SfM) erro da pose relativa de duas vistas contra a pose do COLMAP.
"""
import csv
import importlib
import os
import time

import cv2
import numpy as np


def _mod(n, name):
    pkg = __package__
    try:
        return importlib.import_module(f".step_{n}_{name}", pkg)
    except ModuleNotFoundError:
        return importlib.import_module(f".{n}_{name}", pkg)


S1 = _mod(1, "keypoints_detector")
S2 = _mod(2, "matching")
S3 = _mod(3, "epipolar")


def _rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def run(image_dir, names, out_dir, K=None, colmap_poses=None, methods=("SIFT", "ORB"),
        n_pairs=30, pair_gap=3, conn_max=40, log=print):
    """
    colmap_poses: {nome: (R, t)} do SfM, para validar a pose de duas vistas.
    Retorna um dict com o resumo (tambem salvo em CSV/Markdown em out_dir).
    """
    os.makedirs(out_dir, exist_ok=True)
    load = lambda n: cv2.imread(os.path.join(image_dir, n))
    first = load(names[0])
    h, w = first.shape[:2]
    if K is None:
        K = S3.build_intrinsics(w, h)

    # --- keypoints por imagem ---------------------------------------------------------
    for n in [names[i] for i in np.linspace(0, len(names) - 1, 3).astype(int)]:
        img = load(n)
        for m in methods:
            kps, _ = S1.detect_features(img, m)
            S1.draw_keypoints(img, kps, os.path.join(out_dir, f"keypoints_{m}_{os.path.splitext(n)[0]}.jpg"))

    # --- comparacao de detectores em pares (i, i + pair_gap) ---------------------------
    idx = np.linspace(0, len(names) - 1 - pair_gap, min(n_pairs, max(1, len(names) - pair_gap))).astype(int)
    rows = []
    feats = {}
    for m in methods:
        for i in sorted(set(idx) | set(idx + pair_gap)):
            img = load(names[i])
            t0 = time.time()
            kps, desc = S1.detect_features(img, m)
            feats[(m, i)] = (kps, desc, time.time() - t0)
        for i in idx:
            kp1, d1, t1 = feats[(m, i)]
            kp2, d2, _ = feats[(m, i + pair_gap)]
            raw = S2.knn_match(d1, d2, m)
            t0 = time.time()
            g = S3.two_view_geometry(kp1, d1, kp2, d2, K, m)
            met = g["metrics"]
            row = {"method": m, "pair": f"{names[i]}-{names[i + pair_gap]}", "kps": len(kp1),
                   "t_detect_s": t1, "raw_matches": len(raw), "ratio_pass": met["n_matches"],
                   "inliers_E": met["inliers_E"], "inliers_pose": met["inliers_pose"],
                   "points3d": met["n_points3d"], "reproj_median_px": met["reproj_median"]}
            if colmap_poses and g["R"] is not None and names[i] in colmap_poses and names[i + pair_gap] in colmap_poses:
                Ra, ta = colmap_poses[names[i]]
                Rb, tb = colmap_poses[names[i + pair_gap]]
                R_rel = Rb @ Ra.T
                t_rel = tb - R_rel @ ta
                row["rot_err_deg"] = _rot_angle_deg(g["R"].T @ R_rel)
                tn = t_rel / max(np.linalg.norm(t_rel), 1e-12)
                row["t_dir_err_deg"] = float(np.degrees(np.arccos(np.clip(float(g["t"].ravel() @ tn), -1, 1))))
            rows.append(row)

    with open(os.path.join(out_dir, "comparacao_detectores.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99))
        wr.writeheader()
        wr.writerows(rows)

    summary = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        s = {k: float(np.nanmean([r[k] for r in rs])) for k in
             ("kps", "t_detect_s", "raw_matches", "ratio_pass", "inliers_E", "points3d", "reproj_median_px")}
        s["inlier_ratio"] = float(np.mean([r["inliers_E"] / max(r["ratio_pass"], 1) for r in rs]))
        for k in ("rot_err_deg", "t_dir_err_deg"):
            v = [r[k] for r in rs if k in r]
            if v:
                s[k] = float(np.median(v))
        summary[m] = s
    md = ["| Metrica (media por par) | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
    labels = [("kps", "Keypoints por imagem", "{:.0f}"), ("t_detect_s", "Tempo de deteccao (s)", "{:.3f}"),
              ("raw_matches", "Matches brutos (k-NN)", "{:.0f}"), ("ratio_pass", "Aprovados no ratio test", "{:.0f}"),
              ("inliers_E", "Inliers de E (RANSAC)", "{:.0f}"), ("inlier_ratio", "Fracao de inliers", "{:.2f}"),
              ("points3d", "Pontos triangulados", "{:.0f}"), ("reproj_median_px", "Reprojecao mediana (px)", "{:.3f}"),
              ("rot_err_deg", "Erro de rotacao vs COLMAP (graus, mediana)", "{:.2f}"),
              ("t_dir_err_deg", "Erro da direcao de t vs COLMAP (graus, mediana)", "{:.2f}")]
    for k, lab, fmt in labels:
        if all(k in summary[m] for m in methods):
            md.append(f"| {lab} | " + " | ".join(fmt.format(summary[m][k]) for m in methods) + " |")
    with open(os.path.join(out_dir, "comparacao_detectores.md"), "w") as f:
        f.write(f"Pares ({len(idx)}) com {pair_gap} quadros-chave de distancia.\n\n" + "\n".join(md) + "\n")
    log("  comparacao de detectores:\n    " + "\n    ".join(md))

    # --- um par em detalhe (SIFT) --------------------------------------------------------
    i = int(idx[len(idx) // 2])
    j = i + pair_gap
    img1, img2 = load(names[i]), load(names[j])
    kp1, d1, _ = feats[("SIFT", i)] if ("SIFT", i) in feats else (*S1.detect_features(img1), 0)
    kp2, d2, _ = feats[("SIFT", j)] if ("SIFT", j) in feats else (*S1.detect_features(img2), 0)
    g = S3.two_view_geometry(kp1, d1, kp2, d2, K, "SIFT")
    raw = [p[0] for p in S2.knn_match(d1, d2, "SIFT")]
    S2.draw_matches_before_after(img1, kp1, img2, kp2, raw, g["good"], g["mask_E"],
                                 os.path.join(out_dir, "matches"))
    if g["E"] is not None:
        pts1, pts2 = S3.matched_points(kp1, kp2, g["good"])
        F = S3.fundamental_from_essential(g["E"], K)
        S3.draw_epipolar_lines(img1, img2, pts1[g["mask_E"]], pts2[g["mask_E"]], F,
                               os.path.join(out_dir, "linhas_epipolares.jpg"))
        if g["points3d"] is not None and len(g["points3d"]):
            p1 = pts1[g["mask_pose"]]
            cols = [img1[int(y), int(x)] for x, y in p1]
            valid = S3.triangulate(pts1[g["mask_pose"]], pts2[g["mask_pose"]], K, g["R"], g["t"])[1]
            S3.save_point_cloud(g["points3d"], os.path.join(out_dir, "duas_vistas.ply"),
                                np.array(cols)[valid])

    # --- matriz de conectividade ---------------------------------------------------------
    sub = np.linspace(0, len(names) - 1, min(conn_max, len(names))).astype(int)
    kps_all, descs_all = S1.detect_features_all([load(names[k]) for k in sub], "SIFT")
    conn, _ = S3.build_connectivity_matrix(kps_all, descs_all, K, "SIFT")
    np.savetxt(os.path.join(out_dir, "conectividade.csv"), conn, fmt="%d", delimiter=",")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(conn, cmap="magma")
    fig.colorbar(im, ax=ax, label="inliers de E")
    ax.set_title(f"Conectividade entre {len(sub)} quadros (SIFT)")
    ax.set_xlabel("quadro")
    ax.set_ylabel("quadro")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conectividade.png"), dpi=160)
    plt.close(fig)
    return {"detectors": summary, "detail_pair": [names[i], names[j]], "detail_pair_metrics": g["metrics"]}
