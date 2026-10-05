import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config
from .features import detect_features_all, compare_detectors, draw_keypoints, scale_keypoints, to_uint8
from .matching import knn_match, match_features, draw_matches_before_after
from .ordering import build_connectivity_matrix, build_neighbor_graph, detect_intruders, infer_order
from .homography import (estimate_homography, make_camera, estimate_focal, initial_rotations,
                         bundle_adjust, wave_correct)
from .blending import compose_panorama, free_memory

def load_images(input_dir, shuffle=True, max_dim=None, seed=0):
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    filenames = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext))
    if shuffle:
        random.Random(seed).shuffle(filenames)

    def read(fname):
        img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img[..., :3]
        full_size = (img.shape[1], img.shape[0])
        scale = min(1.0, max_dim / max(full_size)) if max_dim else 1.0
        if scale < 1.0:
            size = (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img, full_size

    images, kept_names, sizes = [], [], []
    with ThreadPoolExecutor(8) as executor:
        for fname, (img, full_size) in zip(filenames, executor.map(read, filenames)):
            if img is None:
                print(f"Aviso: nao foi possivel ler {fname}, ignorando.")
                continue
            images.append(img)
            kept_names.append(fname)
            sizes.append(full_size)

    print("Ordem de leitura (embaralhada; usada so para referencia/depuracao do usuario):")
    for i, f in enumerate(kept_names):
        print(f"  indice {i}: {f}")

    return images, kept_names, sizes

def save_image(path, img_float, bits):
    dtype, scale = (np.uint16, 65535) if bits == 16 else (np.uint8, 255)
    out = np.empty(img_float.shape, dtype)
    for r in range(0, img_float.shape[0], 256):
        out[r:r + 256] = np.round(np.clip(img_float[r:r + 256], 0, 1) * scale)
    cv2.imwrite(str(path), out)

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def draw_connectivity(connectivity, order, intruders, names, path):
    idx = list(order) + list(intruders)
    n = len(idx)
    cell = max(10, 1400 // n)
    m = connectivity[np.ix_(idx, idx)].astype(np.float32)
    heat = cv2.applyColorMap((255 * np.sqrt(m / max(m.max(), 1))).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heat = cv2.resize(heat, (n * cell, n * cell), interpolation=cv2.INTER_NEAREST)
    margin = 9 * cell
    img = np.full((n * cell + margin, n * cell + margin, 3), 255, np.uint8)
    img[margin:, margin:] = heat
    font = 0.32 * cell / 10
    for k, i in enumerate(idx):
        color = (0, 0, 220) if i in intruders else (0, 0, 0)
        y = margin + k * cell + int(0.75 * cell)
        cv2.putText(img, names[i][:8], (4, y), cv2.FONT_HERSHEY_SIMPLEX, font, color, 1, cv2.LINE_AA)
    cols = np.full((margin, n * cell, 3), 255, np.uint8)
    cols = cv2.rotate(cols, cv2.ROTATE_90_CLOCKWISE)
    for k, i in enumerate(idx):
        color = (0, 0, 220) if i in intruders else (0, 0, 0)
        cv2.putText(cols, names[i][:8], (4, k * cell + int(0.75 * cell)), cv2.FONT_HERSHEY_SIMPLEX,
                    font, color, 1, cv2.LINE_AA)
    img[:margin, margin:] = cv2.rotate(cols, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if intruders:
        edge = margin + len(order) * cell
        cv2.line(img, (margin, edge), (img.shape[1], edge), (0, 0, 220), 2)
        cv2.line(img, (edge, margin), (edge, img.shape[0]), (0, 0, 220), 2)
    cv2.imwrite(str(path), img)

def run_pipeline(input_dir, output_dir, projection="cylindrical", beta=0.3, ba=None, align=None):
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    detector, ratio = config.DETECTOR, config.RATIO_THRESH

    def elapsed():
        return f"[{time.time() - t_start:.0f} s]"

    images, filenames, sizes = load_images(input_dir, max_dim=config.WORK_DIM)
    if len(images) < 6:
        print(f"Aviso: o enunciado pede no minimo 6 imagens; foram carregadas {len(images)}.")
    bits = 16 if images[0].dtype == np.uint16 else 8
    vis_scale = min(1.0, config.VIS_MAX_DIM / max(images[0].shape[:2]))
    vis_images = [cv2.resize(to_uint8(img), None, fx=vis_scale, fy=vis_scale, interpolation=cv2.INTER_AREA)
                  for img in images]

    print(f"\n[Etapa 2] Detectando caracteristicas ({detector}, com CLAHE)... {elapsed()}")
    kp_dir = os.path.join(output_dir, "2_keypoints")
    os.makedirs(kp_dir, exist_ok=True)
    all_kps, all_descs = detect_features_all(images, method=detector, equalize=True)
    for i, (img, kps) in enumerate(zip(vis_images, all_kps)):
        draw_keypoints(img, scale_keypoints(kps, vis_scale), os.path.join(kp_dir, f"{filenames[i][:-4]}_{detector}.jpg"))
    print(f"  Keypoints por imagem: min {min(map(len, all_kps))}, max {max(map(len, all_kps))}")

    print(f"\n[Etapa 4] Construindo matriz de conectividade (todos os pares)... {elapsed()}")
    connectivity, homographies, inlier_points = build_connectivity_matrix(
        all_kps, all_descs, detector, ratio, config.RANSAC_THRESH, config.MIN_MATCHES, beta)
    free_memory()
    graph = build_neighbor_graph(connectivity, min_inliers=config.MIN_INLIERS)
    intruders = detect_intruders(graph)
    if intruders:
        print("Imagens rejeitadas por falta de conectividade (intrusas): "
              f"{[filenames[i] for i in intruders]}")
    else:
        print("Nenhuma imagem foi rejeitada por conectividade insuficiente.")
    valid_nodes = [i for i in range(len(images)) if i not in intruders]
    if len({sizes[i] for i in valid_nodes}) != 1:
        raise ValueError("Todas as fotos do panorama precisam ter o mesmo tamanho (mesma camera/lente).")
    full_w, full_h = sizes[valid_nodes[0]]
    work_scale = images[valid_nodes[0]].shape[1] / full_w
    print(f"Fotos de {full_w}x{full_h} px, {bits} bits; etapas 2 a 5 na escala {work_scale:.3f}")

    order, closed = infer_order(graph, valid_nodes)
    print(f"Ordem inferida (arquivos): {[filenames[i] for i in order]}")
    if closed:
        print("O grafo de vizinhanca fecha uma volta (captura de 360 graus).")
    with open(os.path.join(output_dir, "4_matriz_conectividade.csv"), "w", newline="") as f:
        idx = order + intruders
        writer = csv.writer(f)
        writer.writerow([""] + [filenames[i] for i in idx])
        for i in idx:
            writer.writerow([filenames[i]] + [int(connectivity[i, j]) for j in idx])
    draw_connectivity(connectivity, order, intruders, filenames,
                      os.path.join(output_dir, "4_matriz_conectividade.png"))

    a, b = order[len(order) // 2], order[len(order) // 2 + 1]
    print(f"\n[Etapa 2.3] Comparando detectores no par ({filenames[a]}, {filenames[b]})... {elapsed()}")
    rows = compare_detectors(images[a], images[b], config.DETECTORS_COMPARED, kp_dir, True, vis_scale)
    write_csv(os.path.join(kp_dir, "comparacao_detectores.csv"), rows)
    for row in rows:
        print(f"  {row}")

    print(f"\n[Etapa 3] Emparelhamento dos pares consecutivos (ratio test {ratio})... {elapsed()}")
    match_dir = os.path.join(output_dir, "3_matches")
    os.makedirs(match_dir, exist_ok=True)
    pair_rows = []
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        all_matches = [m for m, _ in knn_match(all_descs[i], all_descs[j], detector)]
        good = match_features(all_descs[i], all_descs[j], detector, ratio)
        _, mask, metrics = estimate_homography(all_kps[i], all_kps[j], good, config.RANSAC_THRESH)
        draw_matches_before_after(vis_images[i], scale_keypoints(all_kps[i], vis_scale),
                                  vis_images[j], scale_keypoints(all_kps[j], vis_scale),
                                  all_matches, good, mask,
                                  os.path.join(match_dir, f"{k:02d}_{filenames[i][:-4]}_x_{filenames[j][:-4]}"))
        pair_rows.append({"par": k, "imagem_1": filenames[i], "imagem_2": filenames[j],
                          "matches_antes": len(all_matches), "matches_depois_ratio": len(good),
                          "inliers_ransac": metrics["n_inliers"],
                          "taxa_inliers": round(metrics["inlier_ratio"], 4),
                          "erro_reproj_homografia_px": round((metrics["mean_reproj_error"] or float("nan")) / work_scale, 3)})
        print(f"  {filenames[i]} <-> {filenames[j]}: {len(all_matches)} -> {len(good)} apos ratio test, "
              f"{metrics['n_inliers']} inliers ({metrics['inlier_ratio']:.0%})")

    print(f"\n[Etapa 5 / Extra X1] Bundle adjustment (rotacoes + focal + distorcao da lente)... {elapsed()}")
    neighbors = np.where(connectivity >= config.MIN_INLIERS, connectivity, 0)
    camera = make_camera(full_w, full_h, 1.0)
    focal = estimate_focal({p: H for p, H in homographies.items() if neighbors[p] > 0}, work_scale, camera)
    if focal is None:
        focal = 0.9 * max(full_w, full_h)
        print(f"  Aviso: focal nao estimavel pelas homografias; usando {focal:.0f} px como chute inicial")
    camera = make_camera(full_w, full_h, focal)
    print(f"  Focal inicial (a partir das homografias): {focal:.1f} px")

    ref = order[len(order) // 2]
    rotations = initial_rotations(homographies, neighbors, valid_nodes, ref, work_scale, camera)
    pairs = {p: pts for p, pts in inlier_points.items()
             if neighbors[p] > 0 and p[0] in rotations and p[1] in rotations}
    rotations, camera, ba_stats = bundle_adjust(pairs, rotations, camera, work_scale, ref, **(ba or {}))
    rotations = wave_correct(rotations)
    for row in pair_rows:
        i, j = filenames.index(row["imagem_1"]), filenames.index(row["imagem_2"])
        stats = ba_stats.get((min(i, j), max(i, j)), {})
        row["erro_BA_mediana_px"] = round(stats.get("mediana", float("nan")), 3)
        row["erro_BA_media_px"] = round(stats.get("media", float("nan")), 3)
    write_csv(os.path.join(output_dir, "5_metricas_alinhamento.csv"), pair_rows)
    print("  Metricas por par (Etapa 5.3) em 5_metricas_alinhamento.csv")
    skipped = [filenames[i] for i in order if i not in rotations]
    if skipped:
        print(f"Aviso: imagens fora do componente conexo da referencia foram descartadas: {skipped}")

    print(f"\n[Etapa 6] Compondo panorama final... {elapsed()}")
    del all_kps, all_descs, vis_images, homographies, inlier_points, pairs
    free_memory()

    def load_full(k):
        img = cv2.imread(os.path.join(input_dir, filenames[k]), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img[..., :3]

    out = compose_panorama({k: images[k] for k in rotations}, work_scale, load_full, rotations,
                           camera, projection, config.COMPOSE_SCALE, config.NUM_BANDS, align)

    save_image(os.path.join(output_dir, "6_panorama_final.png"), out["panorama"], bits)
    save_image(os.path.join(output_dir, "6_panorama_final.jpg"), out["panorama"], 8)
    save_image(os.path.join(output_dir, "6_panorama_ingenuo.png"), out["naive"], bits)
    save_image(os.path.join(output_dir, "6_costuras.png"), out["seams"], 8)
    cv2.imwrite(os.path.join(output_dir, "6_comparacao_fantasmas.png"), out["ghost"])
    prog_dir = os.path.join(output_dir, "5_mosaico_progressivo")
    os.makedirs(prog_dir, exist_ok=True)
    for n_added, snap in out["snapshots"]:
        cv2.imwrite(os.path.join(prog_dir, f"mosaico_{n_added:02d}_fotos.jpg"), snap)

    errors = [r["erro_BA_mediana_px"] for r in pair_rows if not np.isnan(r["erro_BA_mediana_px"])]
    metrics = {"fotos_no_panorama": len(rotations), "intrusas_rejeitadas": len(intruders),
               "largura_px": out["panorama"].shape[1], "altura_px": out["panorama"].shape[0],
               **out["metrics"],
               "erro_BA_mediana_pares_consecutivos_px": float(np.median(errors)) if errors else float("nan"),
               "focal_px": camera["f"], "tempo_total_s": round(time.time() - t_start)}
    with open(os.path.join(output_dir, "6_metricas_composicao.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metrica", "valor"])
        for key, value in metrics.items():
            writer.writerow([key, round(value, 3) if isinstance(value, float) else value])

    print(f"\nConcluido em {time.time() - t_start:.0f} s! Resultados salvos em: {output_dir}")
    for key, value in metrics.items():
        print(f"  {key}: {round(value, 3) if isinstance(value, float) else value}")
