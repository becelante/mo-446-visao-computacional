"""
Etapa 2 - Deteccao e Extracao de Caracteristicas.

Implementa deteccao de keypoints com SIFT ou ORB e visualizacao com
escala/orientacao (Figura 2 do enunciado).
"""
import os

import numpy as np
import cv2


def create_detector(method="SIFT"):
    method = method.upper()
    if method == "SIFT":
        return cv2.SIFT_create()
    elif method == "ORB":
        return cv2.ORB_create(nfeatures=4000)
    else:
        raise ValueError(f"Detector desconhecido: {method}")


def to_uint8(image):
    """Converte imagens de 16 bits (ex.: PNGs vindos do RAW) para 8 bits."""
    if image.dtype == np.uint8:
        return image
    return cv2.convertScaleAbs(image, alpha=255.0 / np.iinfo(image.dtype).max)


def to_gray8(image, equalize=False):
    """Cinza de 8 bits; `equalize` aplica CLAHE, que multiplica os keypoints em cenas escuras."""
    gray = to_uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image)
    if equalize:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def detect_features(image, method="SIFT", equalize=False):
    """Retorna (keypoints, descriptors) para uma imagem BGR ou em tons de cinza (8 ou 16 bits)."""
    gray = to_gray8(image, equalize)
    detector = create_detector(method)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def detect_features_all(images, method="SIFT", equalize=False):
    """Aplica deteccao em uma lista de imagens. Retorna listas paralelas de kps e descritores."""
    all_kps, all_descs = [], []
    for img in images:
        kps, descs = detect_features(img, method, equalize)
        all_kps.append(kps)
        all_descs.append(descs)
    return all_kps, all_descs


def draw_keypoints(image, keypoints, out_path=None):
    """Desenha keypoints com circulo (escala) e raio (orientacao)."""
    vis = cv2.drawKeypoints(
        to_uint8(image), keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    if out_path:
        cv2.imwrite(out_path, vis)
    return vis


def compare_detectors(img_a, img_b, methods=("SIFT", "ORB"), out_dir=None, ransac_thresh=4.0):
    """Etapa 2.3 - roda cada detector num par de imagens vizinhas e mede keypoints,
    matches apos o ratio test, inliers do RANSAC e erro de reprojecao. Salva os
    keypoints de cada detector sobre img_a. Retorna uma linha por detector."""
    from .etapa3_matching import knn_match, match_features
    from .etapa5_homography import estimate_homography

    rows = []
    for m in methods:
        kps_a, descs_a = detect_features(img_a, m)
        kps_b, descs_b = detect_features(img_b, m)
        good = match_features(descs_a, descs_b, m)
        _, _, metrics = estimate_homography(kps_a, kps_b, good, ransac_thresh)
        rows.append({"detector": m, "keypoints": len(kps_a),
                     "matches_brutos": len(knn_match(descs_a, descs_b, m)), "matches_ratio_test": len(good),
                     "inliers": metrics["n_inliers"], "taxa_inliers": round(metrics["inlier_ratio"], 3),
                     "erro_reproj_px": round(metrics["mean_reproj_error"] or float("nan"), 3)})
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            draw_keypoints(img_a, kps_a, os.path.join(out_dir, f"keypoints_{m}.jpg"))
    return rows
