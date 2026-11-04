"""
Detecta keypoints e extrai descritores com SIFT (limiar de contraste padrao do OpenCV,
0.04, ou baixo, 0.006, o equivalente ao `peak_threshold` 0.002 do COLMAP) ou ORB, e
desenha os keypoints com escala/orientacao.
"""
import cv2

def create_detector(method="SIFT"):
    method = method.upper()
    if method == "SIFT":
        return cv2.SIFT_create()
    elif method == "SIFT_BAIXO":
        return cv2.SIFT_create(contrastThreshold=0.006)
    elif method == "ORB":
        return cv2.ORB_create(nfeatures=4000)
    else:
        raise ValueError(f"Detector desconhecido: {method}")

def detect_features(image, method="SIFT"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return create_detector(method).detectAndCompute(gray, None)

def detect_features_all(images, method="SIFT"):
    all_kps, all_descs = [], []
    for img in images:
        kps, descs = detect_features(img, method)
        all_kps.append(kps)
        all_descs.append(descs)
    return all_kps, all_descs

def draw_keypoints(image, keypoints, out_path=None):
    vis = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if out_path:
        cv2.imwrite(out_path, vis)
    return vis
