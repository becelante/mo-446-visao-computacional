"""
Etapa 4 - Ordenacao Automatica das Imagens (obrigatoria, sem uso de EXIF).

Constroi a matriz de conectividade a partir do numero de correspondencias
inliers entre cada par de imagens (Figura 4), monta o grafo de vizinhanca e
infere a sequencia de captura. Imagens sem conectividade suficiente (ex.:
a imagem intrusa da Etapa 4.4) sao rejeitadas automaticamente.
"""
import numpy as np
import cv2

from .features import detect_features
from .matching import match_features


def count_inliers(kps1, descs1, kps2, descs2, method="SIFT", ratio_thresh=0.75,
                   ransac_thresh=4.0, min_matches=8):
    """
    Conta o numero de correspondencias inliers entre duas imagens apos ajuste
    de homografia com RANSAC. Retorna (n_inliers, matches_bons, H).
    H mapeia pontos da imagem 1 para o referencial da imagem 2.
    """
    good = match_features(descs1, descs2, method, ratio_thresh)
    if len(good) < min_matches:
        return 0, good, None

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None:
        return 0, good, None

    n_inliers = int(mask.sum())
    return n_inliers, good, H


def build_connectivity_matrix(images, method="SIFT", ratio_thresh=0.75,
                               ransac_thresh=4.0, min_matches=8):
    """
    Constroi a matriz N x N com o numero de inliers entre cada par de imagens.
    Tambem retorna keypoints/descritores ja calculados (reaproveitados nas
    etapas seguintes) e as homografias estimadas para cada par conectado
    (armazenadas apenas como homographies[(i, j)] com i < j, mapeando i -> j).
    """
    n = len(images)
    all_kps, all_descs = [], []
    for img in images:
        kps, descs = detect_features(img, method)
        all_kps.append(kps)
        all_descs.append(descs)

    connectivity = np.zeros((n, n), dtype=int)
    homographies = {}
    for i in range(n):
        for j in range(i + 1, n):
            n_inliers, _, H = count_inliers(
                all_kps[i], all_descs[i], all_kps[j], all_descs[j],
                method, ratio_thresh, ransac_thresh, min_matches,
            )
            connectivity[i, j] = n_inliers
            connectivity[j, i] = n_inliers
            if H is not None:
                homographies[(i, j)] = H  # mapeia i -> j

    return connectivity, all_kps, all_descs, homographies


def build_neighbor_graph(connectivity, min_inliers=20):
    """
    Converte a matriz de conectividade em um grafo de adjacencia (dict de
    listas de (vizinho, peso)), mantendo apenas pares cuja contagem de
    inliers ultrapassa min_inliers.
    """
    n = connectivity.shape[0]
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and connectivity[i, j] >= min_inliers:
                graph[i].append((j, connectivity[i, j]))
    return graph


def detect_intruders(graph):
    """Imagens sem nenhuma conexao valida sao consideradas intrusas (Etapa 4.4)."""
    return [node for node, neighbors in graph.items() if len(neighbors) == 0]


def infer_order(graph, valid_nodes):
    """
    Infere a sequencia de captura a partir do grafo de vizinhanca (Etapa 4.3).

    Heuristica: parte do no de menor grau (tende a ser uma extremidade do
    panorama, ja que imagens do meio se conectam a duas vizinhas e as das
    pontas a apenas uma) e caminha sempre para o vizinho nao visitado de
    maior peso (mais inliers = conexao mais confiavel). Assume topologia
    aproximadamente linear, como a captura em faixa da Figura 1.
    """
    if not valid_nodes:
        return []

    sub_graph = {n: [(nb, w) for nb, w in graph[n] if nb in valid_nodes] for n in valid_nodes}

    start = min(valid_nodes, key=lambda n: len(sub_graph[n]))

    order = [start]
    visited = {start}
    current = start
    while len(order) < len(valid_nodes):
        candidates = [(nb, w) for nb, w in sub_graph[current] if nb not in visited]
        if not candidates:
            remaining = [n for n in valid_nodes if n not in visited]
            if not remaining:
                break
            # grafo desconectado: reinicia a partir de outro componente
            current = remaining[0]
            order.append(current)
            visited.add(current)
            continue
        next_node = max(candidates, key=lambda x: x[1])[0]
        order.append(next_node)
        visited.add(next_node)
        current = next_node

    return order
