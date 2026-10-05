import os
import time

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
    if image.dtype == np.uint8:
        return image
    return cv2.convertScaleAbs(image, alpha=255.0 / np.iinfo(image.dtype).max)

def to_gray8(image, equalize=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = to_uint8(gray)
    if equalize:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray

def detect_features(image, method="SIFT", equalize=False):
    gray = to_gray8(image, equalize)
    detector = create_detector(method)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

def detect_features_all(images, method="SIFT", equalize=False):
    all_kps, all_descs = [], []
    for img in images:
        kps, descs = detect_features(img, method, equalize)
        all_kps.append(kps)
        all_descs.append(descs)
    return all_kps, all_descs

def scale_keypoints(keypoints, factor):
    return [cv2.KeyPoint(kp.pt[0] * factor, kp.pt[1] * factor, kp.size * factor,
                         kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

def draw_keypoints(image, keypoints, out_path=None):
    vis = cv2.drawKeypoints(
        to_uint8(image), keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    if out_path:
        cv2.imwrite(out_path, vis)
    return vis

def compare_detectors(img_a, img_b, methods=("SIFT", "ORB"), out_dir=None,
                      equalize=False, vis_scale=1.0):
    from .matching import match_features
    from .homography import estimate_homography

    rows = []
    for m in methods:
        t0 = time.perf_counter()
        kps_a, descs_a = detect_features(img_a, m, equalize)
        kps_b, descs_b = detect_features(img_b, m, equalize)
        t_detect = (time.perf_counter() - t0) / 2
        good = match_features(descs_a, descs_b, m)
        _, _, metrics = estimate_homography(kps_a, kps_b, good)
        rows.append({"detector": m, "keypoints": (len(kps_a) + len(kps_b)) // 2,
                     "tempo_deteccao_s": round(t_detect, 3), "matches_ratio_test": len(good),
                     "inliers": metrics["n_inliers"], "taxa_inliers": round(metrics["inlier_ratio"], 3),
                     "erro_reproj_px": round(metrics["mean_reproj_error"] or float("nan"), 3)})
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            vis = cv2.resize(to_uint8(img_a), None, fx=vis_scale, fy=vis_scale, interpolation=cv2.INTER_AREA)
            draw_keypoints(vis, scale_keypoints(kps_a, vis_scale), os.path.join(out_dir, f"keypoints_{m}.jpg"))
    return rows
