"""
Etapa 3 - Emparelhamento de Caracteristicas com ratio test de Lowe.
"""
import numpy as np
import cv2


def create_matcher(method="SIFT"):
    """FLANN para descritores float (SIFT); BFMatcher com Hamming para binarios (ORB/AKAZE)."""
    method = method.upper()
    if method == "SIFT":
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    else:  # ORB, AKAZE -> descritores binarios
        return cv2.BFMatcher(cv2.NORM_HAMMING)


def match_features(descs1, descs2, method="SIFT", ratio_thresh=0.75):
    """
    Emparelha descritores entre duas imagens usando k-NN (k=2) e aplica o
    ratio test de Lowe (Etapa 3.2). Retorna a lista de cv2.DMatch aprovados,
    ordenada por distancia (melhores primeiro).
    """
    if descs1 is None or descs2 is None or len(descs1) < 2 or len(descs2) < 2:
        return []

    matcher = create_matcher(method)
    if method.upper() != "SIFT":
        descs1 = descs1.astype(np.uint8)
        descs2 = descs2.astype(np.uint8)
    else:
        descs1 = descs1.astype(np.float32)
        descs2 = descs2.astype(np.float32)

    knn_matches = matcher.knnMatch(descs1, descs2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    good.sort(key=lambda m: m.distance)
    return good


def draw_matches(img1, kps1, img2, kps2, matches, out_path=None, max_draw=100):
    """Etapa 3.3 - visualiza linhas conectando os keypoints casados."""
    vis = cv2.drawMatches(
        img1, kps1, img2, kps2, matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if out_path:
        cv2.imwrite(out_path, vis)
    return vis
