"""
Pipeline principal - Trabalho 1 (Panoramas), Visao Computacional, 2o sem. 2026.

Uso:
    python -m src.pipeline --input_dir fotos_png --output_dir resultados \
        --detector SIFT --ratio 0.75 --min_inliers 20

Executa em sequencia todas as etapas obrigatorias do enunciado:
  1. Carrega as imagens (nomes embaralhados em memoria, sem uso de EXIF)
  2. Detecao e extracao de caracteristicas (+ comparacao de detectores)
  3. Emparelhamento com ratio test de Lowe
  4. Ordenacao automatica via matriz de conectividade / grafo de vizinhanca,
     rejeitando imagens sem conectividade suficiente (intrusas)
  5. Estimacao de homografia com RANSAC e composicao das transformacoes
     par-a-par ate um referencial comum
  6. Composicao do panorama com remocao de fantasmas (costura otima) e
     feathering, comparado com a composicao ingenua
"""
import argparse
import os
import random

import cv2
import numpy as np

from .features import detect_features_all, compare_detectors, draw_keypoints
from .matching import match_features, draw_matches
from .ordering import build_connectivity_matrix, build_neighbor_graph, detect_intruders, infer_order
from .homography import estimate_homography, compose_pairwise_homographies
from .blending import compose_panorama


def load_images(input_dir, shuffle=True):
    """Carrega todas as imagens da pasta. A ordem de leitura e embaralhada em
    memoria (Etapa 4.1): o algoritmo de ordenacao (Etapa 4) nunca olha para
    o nome do arquivo, apenas para o conteudo visual."""
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    filenames = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext))
    if shuffle:
        random.shuffle(filenames)

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


def run_pipeline(input_dir, output_dir, detector="SIFT", ratio_thresh=0.75,
                  ransac_thresh=4.0, min_matches=8, min_inliers=20,
                  feather_width=15):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Etapa 1: carregamento ----
    images, filenames = load_images(input_dir)
    if len(images) < 6:
        print(f"Aviso: o enunciado pede no minimo 6 imagens; foram carregadas {len(images)}.")

    # ---- Etapa 2: deteccao de caracteristicas ----
    print("\n[Etapa 2] Detectando caracteristicas...")
    kp_dir = os.path.join(output_dir, "etapa2_keypoints")
    summary = compare_detectors(images[0], out_dir=kp_dir)
    print(f"  Comparacao de detectores na primeira imagem (n. de keypoints): {summary}")

    all_kps_preview, _ = detect_features_all(images, method=detector)
    for i, (img, kps) in enumerate(zip(images, all_kps_preview)):
        draw_keypoints(img, kps, os.path.join(kp_dir, f"img{i}_{detector}.png"))

    # ---- Etapa 4 (usa Etapa 3 internamente): matriz de conectividade + ordenacao ----
    print("\n[Etapa 4] Construindo matriz de conectividade (todos os pares)...")
    connectivity, all_kps, all_descs, homographies = build_connectivity_matrix(
        images, method=detector, ratio_thresh=ratio_thresh,
        ransac_thresh=ransac_thresh, min_matches=min_matches,
    )
    print("Matriz de conectividade (n. de inliers por par):")
    print(connectivity)
    np.savetxt(os.path.join(output_dir, "matriz_conectividade.csv"), connectivity, fmt="%d", delimiter=",")

    graph = build_neighbor_graph(connectivity, min_inliers=min_inliers)
    intruders = detect_intruders(graph)
    if intruders:
        print("Imagens rejeitadas por falta de conectividade (candidatas a intrusas): "
              f"{[filenames[i] for i in intruders]}")
    else:
        print("Nenhuma imagem foi rejeitada por conectividade insuficiente "
              "(ajuste --min_inliers se uma intrusa nao foi detectada).")

    valid_nodes = [i for i in range(len(images)) if i not in intruders]
    order = infer_order(graph, valid_nodes)
    print(f"Ordem inferida (indices): {order}")
    print(f"Ordem inferida (arquivos): {[filenames[i] for i in order]}")

    # ---- Etapa 3: visualizacao do emparelhamento para os pares consecutivos da ordem ----
    print("\n[Etapa 3] Salvando visualizacoes de emparelhamento para pares consecutivos...")
    match_dir = os.path.join(output_dir, "etapa3_matches")
    os.makedirs(match_dir, exist_ok=True)
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        good = match_features(all_descs[i], all_descs[j], detector, ratio_thresh)
        draw_matches(images[i], all_kps[i], images[j], all_kps[j], good,
                     os.path.join(match_dir, f"match_{k}_{filenames[i]}_x_{filenames[j]}.png"))
        print(f"  {filenames[i]} <-> {filenames[j]}: {len(good)} correspondencias apos ratio test")

    # ---- Etapa 5: homografias compostas para um referencial comum ----
    print("\n[Etapa 5] Estimando homografias e metricas de alinhamento...")
    H_to_ref = compose_pairwise_homographies(order, homographies)
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        good = match_features(all_descs[i], all_descs[j], detector, ratio_thresh)
        if len(good) >= min_matches:
            _, _, metrics = estimate_homography(all_kps[i], all_kps[j], good, ransac_thresh)
            print(f"  Par ({filenames[i]}, {filenames[j]}): {metrics}")

    ordered_images = [images[i] for i in order if i in H_to_ref]
    ordered_H = [H_to_ref[i] for i in order if i in H_to_ref]
    if len(ordered_images) < len(order):
        skipped = [filenames[i] for i in order if i not in H_to_ref]
        print(f"Aviso: imagens fora da cadeia de homografias validas foram descartadas: {skipped}")

    # ---- Etapa 6: composicao, blending e remocao de fantasmas ----
    print("\n[Etapa 6] Compondo panorama final...")
    panorama, naive = compose_panorama(ordered_images, ordered_H, feather_width=feather_width)

    cv2.imwrite(os.path.join(output_dir, "panorama_ingenuo.png"), naive)
    cv2.imwrite(os.path.join(output_dir, "panorama_final.png"), panorama)

    print(f"\nConcluido! Resultados salvos em: {output_dir}")
    print("  - panorama_ingenuo.png   (composicao simples, evidencia fantasmas)")
    print("  - panorama_final.png     (com remocao de fantasmas via costura + feathering)")
    print("  - matriz_conectividade.csv")
    print("  - etapa2_keypoints/, etapa3_matches/   (visualizacoes para o relatorio)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de construcao de panoramas (Trabalho 1)")
    parser.add_argument("--input_dir", required=True, help="Pasta com as imagens (fora de ordem)")
    parser.add_argument("--output_dir", default="resultados")
    parser.add_argument("--detector", default="SIFT", choices=["SIFT", "ORB", "AKAZE"])
    parser.add_argument("--ratio", type=float, default=0.75, help="Limiar do ratio test de Lowe")
    parser.add_argument("--ransac_thresh", type=float, default=4.0)
    parser.add_argument("--min_matches", type=int, default=8)
    parser.add_argument("--min_inliers", type=int, default=20,
                         help="N. minimo de inliers para considerar duas imagens vizinhas")
    parser.add_argument("--feather_width", type=int, default=15)
    args = parser.parse_args()

    run_pipeline(args.input_dir, args.output_dir, args.detector, args.ratio,
                 args.ransac_thresh, args.min_matches, args.min_inliers, args.feather_width)
