import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from .matching import match_features

def count_inliers(kps1, descs1, kps2, descs2, method="SIFT", ratio_thresh=0.75,
                   ransac_thresh=4.0, min_matches=8, beta=0.3):
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
    # Brown & Lowe (2007) + homografia nao degenerada: rejeitam pares falsos de texturas repetitivas
    if n_inliers <= 8 + beta * len(good):
        return 0, good, None, None
    cx, cy = pts1[mask].reshape(-1, 2).mean(axis=0)
    local_scale = np.linalg.det(H) / (H[2, 0] * cx + H[2, 1] * cy + H[2, 2]) ** 3
    if not 0.2 < local_scale < 5:
        return 0, good, None, None
    return n_inliers, good, H, mask

def build_connectivity_matrix(all_kps, all_descs, method="SIFT", ratio_thresh=0.75,
                               ransac_thresh=4.0, min_matches=8, beta=0.3):
    n = len(all_kps)

    def process_pair(pair):
        i, j = pair
        return count_inliers(
            all_kps[i], all_descs[i], all_kps[j], all_descs[j],
            method, ratio_thresh, ransac_thresh, min_matches, beta,
        )

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    connectivity = np.zeros((n, n), dtype=int)
    homographies = {}
    inlier_points = {}
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        for (i, j), (n_inliers, good, H, mask) in zip(pairs, executor.map(process_pair, pairs)):
            connectivity[i, j] = n_inliers
            connectivity[j, i] = n_inliers
            if H is not None:
                homographies[(i, j)] = H
                inliers = [m for m, ok in zip(good, mask) if ok]
                inlier_points[(i, j)] = (
                    np.float32([all_kps[i][m.queryIdx].pt for m in inliers]),
                    np.float32([all_kps[j][m.trainIdx].pt for m in inliers]),
                )

    return connectivity, homographies, inlier_points

def build_neighbor_graph(connectivity, min_inliers=20):
    n = connectivity.shape[0]
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and connectivity[i, j] >= min_inliers:
                graph[i].append((j, connectivity[i, j]))
    return graph

def detect_intruders(graph):
    return [node for node, neighbors in graph.items() if len(neighbors) == 0]

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

def infer_order(graph, valid_nodes):
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
