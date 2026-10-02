"""
Etapa 2 - Deteccao e Extracao de Caracteristicas.

Implementa deteccao de keypoints com SIFT, ORB ou AKAZE, e visualizacao com
escala/orientacao (Figura 2 do enunciado).
"""
import os
import cv2


def create_detector(method="SIFT"):
    method = method.upper()
    if method == "SIFT":
        return cv2.SIFT_create()
    elif method == "ORB":
        return cv2.ORB_create(nfeatures=4000)
    elif method == "AKAZE":
        return cv2.AKAZE_create()
    else:
        raise ValueError(f"Detector desconhecido: {method}")


def detect_features(image, method="SIFT"):
    """Retorna (keypoints, descriptors) para uma imagem BGR ou em tons de cinza."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = create_detector(method)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def detect_features_all(images, method="SIFT"):
    """Aplica deteccao em uma lista de imagens. Retorna listas paralelas de kps e descritores."""
    all_kps, all_descs = [], []
    for img in images:
        kps, descs = detect_features(img, method)
        all_kps.append(kps)
        all_descs.append(descs)
    return all_kps, all_descs


def draw_keypoints(image, keypoints, out_path=None):
    """Desenha keypoints com circulo (escala) e raio (orientacao)."""
    vis = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    if out_path:
        cv2.imwrite(out_path, vis)
    return vis


def compare_detectors(image, methods=("SIFT", "ORB", "AKAZE"), out_dir=None):
    """Etapa 2.3 - compara detectores e retorna um resumo (nome -> numero de keypoints)."""
    summary = {}
    for m in methods:
        kps, _ = detect_features(image, m)
        summary[m] = len(kps)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            draw_keypoints(image, kps, os.path.join(out_dir, f"keypoints_{m}.png"))
    return summary
