"""Extras X1 a X3 - pipeline com bundle adjustment e projecao cilindrica
ou esferica, executado por 3_extras.py com os parametros de src/config.py. As etapas 2 a
5 rodam numa copia reduzida das fotos; so a composicao final le a resolucao total."""
import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config
from .etapa2_features import detect_features_all
from .etapa3_matching import knn_match, match_features
from .etapa4_ordering import build_connectivity_matrix, build_neighbor_graph, detect_intruders, infer_order_spectral
from .etapa5_homography import estimate_homography
from .etapa5_bundle_adjustment import make_camera, estimate_focal, initial_rotations, bundle_adjust, wave_correct
from .etapa6_blending_global import compose_panorama, free_memory
from .passo2_pipeline import write_csv

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

def run_pipeline(input_dir, output_dir, projection="cylindrical", beta=0.3, ba=None, align=None):
    align_dir, compose_dir = os.path.join(output_dir, "5_alinhamento"), os.path.join(output_dir, "6_composicao")
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(compose_dir, exist_ok=True)
    t_start = time.time()
    detector, ratio = config.DETECTOR, config.RATIO_THRESH

    def elapsed():
        return f"[{time.time() - t_start:.0f} s]"

    images, filenames, sizes = load_images(input_dir, max_dim=config.WORK_DIM, seed=config.SEED)
    if len(images) < 6:
        print(f"Aviso: o enunciado pede no minimo 6 imagens; foram carregadas {len(images)}.")
    bits = 16 if images[0].dtype == np.uint16 else 8

    print(f"\n[Etapa 2] Detectando caracteristicas ({detector}, com CLAHE)... {elapsed()}")
    all_kps, all_descs = detect_features_all(images, method=detector, equalize=True)
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

    order, closed = infer_order_spectral(graph, valid_nodes)
    print(f"Ordem inferida (arquivos): {[filenames[i] for i in order]}")
    if closed:
        print("O grafo de vizinhanca fecha uma volta (captura de 360 graus).")

    print(f"\n[Etapa 3] Emparelhamento dos pares consecutivos (ratio test {ratio})... {elapsed()}")
    pair_rows = []
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        all_matches = [m for m, _ in knn_match(all_descs[i], all_descs[j], detector)]
        good = match_features(all_descs[i], all_descs[j], detector, ratio)
        _, _, metrics = estimate_homography(all_kps[i], all_kps[j], good, config.RANSAC_THRESH)
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
    write_csv(os.path.join(align_dir, "metricas.csv"), pair_rows)
    skipped = [filenames[i] for i in order if i not in rotations]
    if skipped:
        print(f"Aviso: imagens fora do componente conexo da referencia foram descartadas: {skipped}")

    print(f"\n[Etapa 6] Compondo panorama final... {elapsed()}")
    del all_kps, all_descs, homographies, inlier_points, pairs
    free_memory()

    def load_full(k):
        img = cv2.imread(os.path.join(input_dir, filenames[k]), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img[..., :3]

    out = compose_panorama({k: images[k] for k in rotations}, work_scale, load_full, rotations,
                           camera, projection, config.COMPOSE_SCALE, config.NUM_BANDS, align)

    save_image(os.path.join(compose_dir, "panorama_final.png"), out["panorama"], bits)
    save_image(os.path.join(compose_dir, "panorama_final.jpg"), out["panorama"], 8)

    errors = [r["erro_BA_mediana_px"] for r in pair_rows if not np.isnan(r["erro_BA_mediana_px"])]
    metrics = {"fotos_no_panorama": len(rotations), "intrusas_rejeitadas": len(intruders),
               "largura_px": out["panorama"].shape[1], "altura_px": out["panorama"].shape[0],
               **out["metrics"],
               "erro_BA_mediana_pares_consecutivos_px": float(np.median(errors)) if errors else float("nan"),
               "focal_px": camera["f"], "tempo_total_s": round(time.time() - t_start)}
    with open(os.path.join(compose_dir, "metricas.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metrica", "valor"])
        for key, value in metrics.items():
            writer.writerow([key, round(value, 3) if isinstance(value, float) else value])

    print(f"\nConcluido em {time.time() - t_start:.0f} s! Resultados salvos em: {output_dir}")
    for key, value in metrics.items():
        print(f"  {key}: {round(value, 3) if isinstance(value, float) else value}")
