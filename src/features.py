"""
Detecção e Extração de Características.

Implementa detecção de keypoints com SIFT e ORB, e visualização com
escala/orientação
"""
import os
import cv2


def create_detector(method="SIFT"):
    method = method.upper()
    if method == "SIFT":
        return cv2.SIFT_create()
    elif method == "ORB":
        return cv2.ORB_create(nfeatures=4000)
    else:
        raise ValueError(f"Detector desconhecido: {method}")


def detect_features(image, method="SIFT"):
    """Retorna (keypoints, descriptors) para uma imagem BGR ou em tons de cinza."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    detector = create_detector(method)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def detect_features_all(images, method="SIFT"):
    """Aplica detecção em uma lista de imagens. Retorna listas paralelas de kps e descritores."""
    all_kps, all_descs = [], []
    for img in images:
        kps, descs = detect_features(img, method)
        all_kps.append(kps)
        all_descs.append(descs)
    return all_kps, all_descs


def draw_keypoints(image, keypoints, out_path=None):
    """Desenha keypoints com círculo (escala) e raio (orientação)."""
    vis = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    if out_path:
        cv2.imwrite(out_path, vis)
    return vis


def compare_detectors(image, methods=("SIFT", "ORB"), out_dir=None):
    """Compara detectores (para somente uma imagem) e retorna um resumo (nome -> número de keypoints)."""
    summary = {}
    for m in methods:
        kps, _ = detect_features(image, m)
        summary[m] = len(kps)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            draw_keypoints(image, kps, os.path.join(out_dir, f"keypoints_{m}.png"))
    return summary
