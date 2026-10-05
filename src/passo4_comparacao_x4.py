"""Extra X4 - SuperPoint + LightGlue (GPU) comparado ao SIFT com ratio test."""
import csv
import time

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from . import config
from .etapa2_features import detect_features, to_gray8
from .etapa3_matching import match_features

def load(path, max_dim):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., :3]
    s = min(1.0, max_dim / max(img.shape[:2]))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s

def match_sift(img_a, img_b):
    t0 = time.perf_counter()
    kps_a, descs_a = detect_features(img_a, "SIFT", equalize=True)
    kps_b, descs_b = detect_features(img_b, "SIFT", equalize=True)
    good = match_features(descs_a, descs_b, "SIFT", config.RATIO_THRESH)
    elapsed = time.perf_counter() - t0
    pts_a = np.float32([kps_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kps_b[m.trainIdx].pt for m in good])
    return (len(kps_a) + len(kps_b)) / 2, pts_a, pts_b, elapsed

def match_lightglue(img_a, img_b, extractor, matcher, device):
    def to_tensor(img):
        return torch.from_numpy(to_gray8(img, equalize=True)).float()[None].to(device) / 255

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        feats_a = extractor.extract(to_tensor(img_a), resize=None)
        feats_b = extractor.extract(to_tensor(img_b), resize=None)
        matches = matcher({"image0": feats_a, "image1": feats_b})
    feats_a, feats_b, matches = rbd(feats_a), rbd(feats_b), rbd(matches)
    idx = matches["matches"].cpu().numpy()
    pts_a = feats_a["keypoints"].cpu().numpy()[idx[:, 0]].astype(np.float32)
    pts_b = feats_b["keypoints"].cpu().numpy()[idx[:, 1]].astype(np.float32)
    torch.cuda.synchronize()
    n_kps = (len(feats_a["keypoints"]) + len(feats_b["keypoints"])) / 2
    return n_kps, pts_a, pts_b, time.perf_counter() - t0

def ransac_metrics(pts_a, pts_b, scale):
    if len(pts_a) < 4:
        return 0, 0.0, float("nan")
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, config.RANSAC_THRESH)
    if H is None:
        return 0, 0.0, float("nan")
    mask = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(pts_a[mask].reshape(-1, 1, 2), H).reshape(-1, 2)
    error = float(np.linalg.norm(proj - pts_b[mask], axis=1).mean()) / scale
    return int(mask.sum()), float(mask.mean()), error

def compare_matchers(png_dir, results_dir):
    with open(results_dir / "5_alinhamento" / "metricas.csv") as f:
        pairs = [(row["imagem_1"], row["imagem_2"]) for row in csv.DictReader(f)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo do LightGlue: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    extractor = SuperPoint(max_num_keypoints=8192).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    cache = {}

    def image(name):
        if name not in cache:
            cache[name] = load(png_dir / name, config.WORK_DIM)
        return cache[name]

    match_lightglue(*[image(n)[0] for n in pairs[0]], extractor, matcher, device)
    rows = []
    for k, (a, b) in enumerate(pairs):
        (img_a, scale), (img_b, _) = image(a), image(b)
        for name, run in (("SIFT + ratio test (CPU)", lambda: match_sift(img_a, img_b)),
                          ("SuperPoint + LightGlue (GPU)", lambda: match_lightglue(img_a, img_b, extractor, matcher, device))):
            n_kps, pts_a, pts_b, elapsed = run()
            inliers, rate, error = ransac_metrics(pts_a, pts_b, scale)
            rows.append({"par": k, "metodo": name, "keypoints": round(n_kps), "matches": len(pts_a),
                         "inliers": inliers, "taxa_inliers": round(rate, 4),
                         "erro_reproj_px": round(error, 3), "tempo_s": round(elapsed, 3)})
        print(f"  par {k}: SIFT {rows[-2]['inliers']} inliers, LightGlue {rows[-1]['inliers']} inliers")

    summary = []
    for name in dict.fromkeys(r["metodo"] for r in rows):
        sel = [r for r in rows if r["metodo"] == name]
        summary.append({"metodo": name, **{key: round(float(np.nanmean([r[key] for r in sel])), 3)
                                         for key in ("keypoints", "matches", "inliers", "taxa_inliers",
                                                     "erro_reproj_px", "tempo_s")}})
    return summary
