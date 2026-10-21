"""
Etapa 3 - Emparelhamento de Caracteristicas com ratio test de Lowe.
"""
import numpy as np
import cv2


def create_matcher(method="SIFT"):
    """FLANN para descritores float (SIFT); BFMatcher com Hamming para binarios (ORB)."""
    method = method.upper()
    if method == "SIFT":
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    else:  # ORB -> descritores binarios
        return cv2.BFMatcher(cv2.NORM_HAMMING)


def knn_match(descs1, descs2, method="SIFT"):
    """Os 2 vizinhos mais proximos (matches brutos) de cada descritor de descs1 em descs2."""
    if descs1 is None or descs2 is None or len(descs1) < 2 or len(descs2) < 2:
        return []
    dtype = np.float32 if method.upper() == "SIFT" else np.uint8
    # o FLANN e aleatorio; a semente fixa (por thread) deixa o resultado reprodutivel
    cv2.setRNGSeed(0)
    pairs = create_matcher(method).knnMatch(descs1.astype(dtype), descs2.astype(dtype), k=2)
    return [p for p in pairs if len(p) == 2]


def match_features(descs1, descs2, method="SIFT", ratio_thresh=0.75):
    """
    Emparelha descritores entre duas imagens usando k-NN (k=2) e aplica o
    ratio test de Lowe (Etapa 3.2). Retorna a lista de cv2.DMatch aprovados,
    ordenada por distancia (melhores primeiro).
    """
    good = [m for m, n in knn_match(descs1, descs2, method) if m.distance < ratio_thresh * n.distance]
    good.sort(key=lambda m: m.distance)
    return good


def draw_matches_before_after(img1, kps1, img2, kps2, all_matches, good, inlier_mask,
                              out_prefix, max_draw=150, seed=0):
    """Etapa 3.3 - "antes": amostra de todos os matches brutos; "depois": amostra
    dos aprovados no ratio test, em verde os inliers do RANSAC e em vermelho os outliers."""
    rng = np.random.default_rng(seed)
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    idx = rng.choice(len(all_matches), min(max_draw, len(all_matches)), replace=False) if all_matches else []
    before = cv2.drawMatches(img1, kps1, img2, kps2, [all_matches[i] for i in idx], None,
                             matchColor=(0, 255, 255), flags=flags)
    cv2.imwrite(f"{out_prefix}_antes.jpg", before)

    mask = inlier_mask if inlier_mask is not None else np.ones(len(good), bool)
    idx = rng.choice(len(good), min(max_draw, len(good)), replace=False) if good else []
    after = cv2.drawMatches(img1, kps1, img2, kps2, [good[i] for i in idx if mask[i]], None,
                            matchColor=(0, 200, 0), flags=flags)
    after = cv2.drawMatches(img1, kps1, img2, kps2, [good[i] for i in idx if not mask[i]], after,
                            matchColor=(0, 0, 255), flags=flags | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    cv2.imwrite(f"{out_prefix}_depois.jpg", after)
