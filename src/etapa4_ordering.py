"""
Etapa 4 - Ordenacao Automatica das Imagens (obrigatoria, sem uso de EXIF).

Constroi a matriz de conectividade a partir do numero de correspondencias
inliers entre cada par de imagens (Figura 4), monta o grafo de vizinhanca e
infere a sequencia de captura. Imagens sem conectividade suficiente (ex.:
a imagem intrusa da Etapa 4.4) sao rejeitadas automaticamente.
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from .etapa3_matching import match_features


def count_inliers(kps1, descs1, kps2, descs2, method="SIFT", ratio_thresh=0.75,
                   ransac_thresh=4.0, min_matches=8, beta=None):
    """
    Conta o numero de correspondencias inliers entre duas imagens apos ajuste
    de homografia com RANSAC. Retorna (n_inliers, matches_bons, H, mascara_inliers).
    H mapeia pontos da imagem 1 para o referencial da imagem 2.

    Com `beta`, o par so e aceito se n_inliers > 8 + beta * n_matches (Brown &
    Lowe, 2007) e se a homografia nao e degenerada (escala local entre 0.2 e 5):
    texturas repetitivas ou densas geram "inliers" falsos entre imagens que nao
    se sobrepoem.
    """
    good = match_features(descs1, descs2, method, ratio_thresh)
    if len(good) < min_matches:
        return 0, good, None, None

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None:
        return 0, good, None, None

    mask = mask.ravel().astype(bool)
    n_inliers = int(mask.sum())
    if beta is not None:
        if n_inliers <= 8 + beta * len(good):
            return 0, good, None, None
        cx, cy = pts1[mask].reshape(-1, 2).mean(axis=0)
        local_scale = np.linalg.det(H) / (H[2, 0] * cx + H[2, 1] * cy + H[2, 2]) ** 3
        if not 0.2 < local_scale < 5:
            return 0, good, None, None
    return n_inliers, good, H, mask


def build_connectivity_matrix(all_kps, all_descs, method="SIFT", ratio_thresh=0.75,
                               ransac_thresh=4.0, min_matches=8, beta=None):
    """
    Constroi a matriz N x N com o numero de inliers entre cada par de imagens
    (pares processados em paralelo). Tambem retorna as homografias de cada par
    conectado (homographies[(i, j)] com i < j, mapeando i -> j) e os pontos
    inliers de cada par (inlier_points[(i, j)] = (pts_i, pts_j)).
    """
    n = len(all_kps)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def process_pair(pair):
        i, j = pair
        return count_inliers(all_kps[i], all_descs[i], all_kps[j], all_descs[j],
                             method, ratio_thresh, ransac_thresh, min_matches, beta)

    connectivity = np.zeros((n, n), dtype=int)
    homographies, inlier_points = {}, {}
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        for (i, j), (n_inliers, good, H, mask) in zip(pairs, executor.map(process_pair, pairs)):
            connectivity[i, j] = connectivity[j, i] = n_inliers
            if H is not None:
                homographies[(i, j)] = H  # mapeia i -> j
                inliers = [m for m, ok in zip(good, mask) if ok]
                inlier_points[(i, j)] = (np.float32([all_kps[i][m.queryIdx].pt for m in inliers]),
                                         np.float32([all_kps[j][m.trainIdx].pt for m in inliers]))

    return connectivity, homographies, inlier_points


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


def draw_neighbor_graph(connectivity, order, intruders, names, min_inliers, out_path, size=1000):
    """Etapa 4.5 - grafo de vizinhanca: imagens em circulo na ordem inferida, arestas
    entre pares com pelo menos `min_inliers` (as da sequencia em azul, com o numero de
    inliers) e as intrusas em vermelho, no mesmo circulo e sem arestas."""
    img = np.full((size, size, 3), 255, np.uint8)
    blue, gray, red, black = (180, 110, 30), (205, 205, 205), (80, 40, 210), (40, 40, 40)
    font, c, radius = cv2.FONT_HERSHEY_SIMPLEX, size / 2, 0.36 * size
    nodes = list(order) + list(intruders)
    pos = {}
    for k, node in enumerate(nodes):
        ang = 2 * math.pi * k / len(nodes) - math.pi / 2
        pos[node] = (int(c + radius * math.cos(ang)), int(c + radius * math.sin(ang)))
    for i in order:
        for j in order:
            if i < j and connectivity[i, j] >= min_inliers:
                cv2.line(img, pos[i], pos[j], gray, 1, cv2.LINE_AA)
    top = max(connectivity[a, b] for a, b in zip(order, order[1:])) if len(order) > 1 else 1
    for a, b in zip(order, order[1:]):
        w = int(connectivity[a, b])
        cv2.line(img, pos[a], pos[b], blue, 2 + int(6 * w / top), cv2.LINE_AA)
        mx, my = (pos[a][0] + pos[b][0]) // 2, (pos[a][1] + pos[b][1]) // 2
        dx, dy = mx - c, my - c
        norm = max(math.hypot(dx, dy), 1)
        cv2.putText(img, str(w), (int(mx + 30 * dx / norm) - 20, int(my + 30 * dy / norm) + 5),
                    font, 0.55, blue, 2, cv2.LINE_AA)
    for k, node in enumerate(nodes):
        x, y = pos[node]
        color = red if node in intruders else blue
        label = "X" if node in intruders else str(k + 1)
        cv2.circle(img, (x, y), 24, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 24, color, 3, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.putText(img, label, (x - tw // 2, y + th // 2), font, 0.7, color, 2, cv2.LINE_AA)
        dx, dy = x - c, y - c
        norm = max(math.hypot(dx, dy), 1)
        name = os.path.splitext(names[node])[0] + (" (intrusa)" if node in intruders else "")
        (tw, _), _ = cv2.getTextSize(name, font, 0.5, 1)
        ux, uy = dx / norm, dy / norm
        d = 32 + tw / 2 * abs(ux) + 8 * abs(uy)
        cv2.putText(img, name, (int(x + d * ux - tw / 2), int(y + d * uy + 5)), font, 0.5,
                    red if node in intruders else black, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def _connected_components(sub_graph, nodes):
    components, seen = [], set()
    for start in nodes:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb, _ in sub_graph[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(sorted(comp))
    return sorted(components, key=len, reverse=True)


def _refine_path(seq, W, ring):
    """Trocas de vizinhos adjacentes enquanto aumentarem a soma de inliers entre consecutivas."""
    seq, n = list(seq), len(seq)

    def w(a, b):
        return W[seq[a % n], seq[b % n]] if ring or (0 <= a < n and 0 <= b < n) else 0

    improved = True
    while improved:
        improved = False
        for k in range(n if ring else n - 1):
            if w(k - 1, k + 1) + w(k, k + 2) > w(k - 1, k) + w(k + 1, k + 2):
                seq[k % n], seq[(k + 1) % n] = seq[(k + 1) % n], seq[k % n]
                improved = True
    return seq


def infer_order_spectral(graph, valid_nodes):
    """
    Alternativa a `infer_order` para muitas imagens (usada nos extras): ordena as
    imagens pelo vetor de Fiedler do Laplaciano do grafo (W = n. de inliers), que
    usa todos os pares de uma vez, e corrige inversoes locais com trocas entre
    vizinhos. Se o grafo e um anel (360 graus), a ordem sai do angulo no plano
    dos dois primeiros autovetores e o anel e cortado no elo mais fraco.
    Retorna (ordem, fecha_volta).
    """
    if not valid_nodes:
        return [], False
    valid = set(valid_nodes)
    sub_graph = {n: [(nb, w) for nb, w in graph[n] if nb in valid] for n in valid_nodes}

    order, closed = [], False
    for c, comp in enumerate(_connected_components(sub_graph, valid_nodes)):
        if len(comp) <= 2:
            order.extend(comp)
            continue
        index = {node: k for k, node in enumerate(comp)}
        W = np.zeros((len(comp), len(comp)))
        for node in comp:
            for nb, w in sub_graph[node]:
                W[index[node], index[nb]] = w
        _, eigvecs = np.linalg.eigh(np.diag(W.sum(axis=1)) - W)
        seq = list(np.argsort(np.arctan2(eigvecs[:, 2], eigvecs[:, 1])))
        links = [W[seq[k], seq[(k + 1) % len(seq)]] for k in range(len(seq))]
        is_ring = len(comp) >= 4 and min(links) > 0
        if is_ring:
            seq = _refine_path(seq, W, ring=True)
            links = [W[seq[k], seq[(k + 1) % len(seq)]] for k in range(len(seq))]
            cut = int(np.argmin(links))
            seq = seq[cut + 1:] + seq[:cut + 1]
        else:
            seq = _refine_path(list(np.argsort(eigvecs[:, 1])), W, ring=False)
            if len(sub_graph[comp[seq[-1]]]) < len(sub_graph[comp[seq[0]]]):
                seq.reverse()
        order.extend(comp[k] for k in seq)
        if c == 0:
            closed = is_ring
    return order, closed
