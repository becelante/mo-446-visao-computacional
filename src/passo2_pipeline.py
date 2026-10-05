"""
Pipeline principal - Trabalho 1 (Panoramas), Visao Computacional, 2o sem. 2026.

Executado por 2_panorama.py, com os parametros de src/config.py. Executa em
sequencia todas as etapas obrigatorias do enunciado (as saidas de cada etapa
ficam numa subpasta com o seu numero):
  1. Carrega as imagens (nomes embaralhados em memoria, sem uso de EXIF)
  2. Detecao e extracao de caracteristicas (+ comparacao de detectores)
  3. Emparelhamento com ratio test de Lowe (antes e depois da filtragem)
  4. Ordenacao automatica via matriz de conectividade / grafo de vizinhanca,
     rejeitando imagens sem conectividade suficiente (intrusas)
  5. Estimacao de homografia com RANSAC e composicao das transformacoes
     par-a-par ate um referencial comum
  6. Composicao do panorama com remocao de fantasmas (costura otima) e
     feathering, comparado com a composicao ingenua
"""
import csv
import os
import random

import cv2
import numpy as np

from . import config
from .etapa2_features import detect_features_all, compare_detectors
from .etapa3_matching import knn_match, match_features, draw_matches_before_after
from .etapa4_ordering import (build_connectivity_matrix, build_neighbor_graph, detect_intruders, infer_order,
                       draw_neighbor_graph)
from .etapa5_homography import estimate_homography, compose_pairwise_homographies
from .etapa6_blending import compose_panorama, ghost_comparison, overlap_metrics


def load_images(input_dir, shuffle=True, seed=0):
    """Carrega todas as imagens da pasta. A ordem de leitura e embaralhada em
    memoria (Etapa 4.1): o algoritmo de ordenacao (Etapa 4) nunca olha para
    o nome do arquivo, apenas para o conteudo visual."""
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    filenames = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext))
    if shuffle:
        random.Random(seed).shuffle(filenames)

    images, kept_names = [], []
    for fname in filenames:
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            print(f"Aviso: nao foi possivel ler {fname}, ignorando.")
            continue
        images.append(img)
        kept_names.append(fname)

    print("Ordem de leitura (embaralhada; usada so para referencia/depuracao do usuario):")
    for i, f in enumerate(kept_names):
        print(f"  indice {i}: {f}")

    return images, kept_names


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(input_dir, output_dir):
    detector, ratio = config.DETECTOR, config.RATIO_THRESH
    ransac_thresh, min_inliers = config.BASE_RANSAC_THRESH, config.BASE_MIN_INLIERS

    def out(step, name):
        os.makedirs(os.path.join(output_dir, step), exist_ok=True)
        return os.path.join(output_dir, step, name)

    # ---- Etapa 1: carregamento ----
    images, filenames = load_images(input_dir, seed=config.SEED)
    names = [os.path.splitext(f)[0] for f in filenames]
    if len(images) < 6:
        print(f"Aviso: o enunciado pede no minimo 6 imagens; foram carregadas {len(images)}.")

    # ---- Etapa 2: deteccao de caracteristicas ----
    print("\n[Etapa 2] Detectando caracteristicas...")
    all_kps, all_descs = detect_features_all(images, method=detector)
    for name, kps in zip(names, all_kps):
        print(f"  {name}: {len(kps)} keypoints")

    # ---- Etapa 4 (usa Etapa 3 internamente): matriz de conectividade + ordenacao ----
    print("\n[Etapa 4] Construindo matriz de conectividade (todos os pares)...")
    connectivity, homographies, _ = build_connectivity_matrix(
        all_kps, all_descs, detector, ratio, ransac_thresh, config.MIN_MATCHES, config.BETA)
    print("Matriz de conectividade (n. de inliers por par, na ordem de leitura):")
    print(connectivity)
    with open(out("4_ordenacao", "matriz_conectividade.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for name, row in zip(names, connectivity):
            writer.writerow([name] + [int(v) for v in row])

    graph = build_neighbor_graph(connectivity, min_inliers=min_inliers)
    intruders = detect_intruders(graph)
    if intruders:
        print(f"Imagens rejeitadas por falta de conectividade (intrusas): {[names[i] for i in intruders]}")
    else:
        print("Nenhuma imagem foi rejeitada por conectividade insuficiente.")

    valid_nodes = [i for i in range(len(images)) if i not in intruders]
    order = infer_order(graph, valid_nodes)
    print(f"Ordem inferida: {[names[i] for i in order]}")
    draw_neighbor_graph(connectivity, order, intruders, names, min_inliers, out("4_ordenacao", "grafo_vizinhanca.png"))

    # ---- Etapa 2.3: comparacao de detectores no par central da sequencia ----
    a, b = order[len(order) // 2 - 1], order[len(order) // 2]
    print(f"\n[Etapa 2.3] Comparando detectores no par ({names[a]}, {names[b]})...")
    rows = compare_detectors(images[a], images[b], config.DETECTORS_COMPARED, os.path.join(output_dir, "2_deteccao"),
                             ransac_thresh=ransac_thresh)
    write_csv(out("2_deteccao", "comparacao_detectores.csv"), rows)
    for row in rows:
        print(f"  {row}")

    # ---- Etapas 3 e 5.3: emparelhamento e metricas dos pares consecutivos ----
    print("\n[Etapas 3 e 5] Emparelhamento, homografias e metricas dos pares consecutivos...")
    pair_rows = []
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        all_matches = [m for m, _ in knn_match(all_descs[i], all_descs[j], detector)]
        good = match_features(all_descs[i], all_descs[j], detector, ratio)
        _, mask, metrics = estimate_homography(all_kps[i], all_kps[j], good, ransac_thresh)
        if (i, j) == (a, b):
            draw_matches_before_after(images[i], all_kps[i], images[j], all_kps[j], all_matches, good, mask,
                                      out("3_emparelhamento", "matches"))
        pair_rows.append({"imagem_1": names[i], "imagem_2": names[j], "matches_brutos": len(all_matches),
                          "matches_ratio_test": len(good), "inliers": metrics["n_inliers"],
                          "taxa_inliers": round(metrics["inlier_ratio"], 4),
                          "erro_reproj_medio_px": round(metrics["mean_reproj_error"] or float("nan"), 4)})
        print(f"  {pair_rows[-1]}")
    write_csv(out("5_alinhamento", "metricas.csv"), pair_rows)

    # ---- Etapa 5: homografias compostas para um referencial comum ----
    H_to_ref = compose_pairwise_homographies(order, homographies)
    kept = [i for i in order if i in H_to_ref]
    if len(kept) < len(order):
        print(f"Aviso: imagens fora da cadeia de homografias validas foram descartadas: "
              f"{[names[i] for i in order if i not in H_to_ref]}")
    ordered_images = [images[i] for i in kept]
    ordered_H = [H_to_ref[i] for i in kept]

    # ---- Etapa 6: composicao, blending e remocao de fantasmas ----
    print("\n[Etapa 6] Compondo panorama final...")
    snapshots = []
    panorama, naive = compose_panorama(ordered_images, ordered_H, feather_width=config.FEATHER_WIDTH,
                                       snapshots=snapshots)
    for n_added, snap in enumerate(snapshots, 1):
        cv2.imwrite(out("5_alinhamento", f"mosaico_{n_added:02d}.jpg"), snap)

    cv2.imwrite(out("6_composicao", "comparacao_fantasmas.jpg"), ghost_comparison(naive, panorama))
    write_csv(out("6_composicao", "metricas.csv"), overlap_metrics(ordered_images, ordered_H, [names[i] for i in kept]))

    ys, xs = np.nonzero(panorama.max(axis=2) > 0)
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    cv2.imwrite(out("6_composicao", "panorama_final.jpg"), panorama[crop], [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\nConcluido! Resultados salvos em: {output_dir}")
