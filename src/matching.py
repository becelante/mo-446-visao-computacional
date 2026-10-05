import numpy as np
import cv2

def create_matcher(method="SIFT"):
    method = method.upper()
    if method == "SIFT":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    else:
        return cv2.BFMatcher(cv2.NORM_HAMMING)

def knn_match(descs1, descs2, method="SIFT"):
    if descs1 is None or descs2 is None or len(descs1) < 2 or len(descs2) < 2:
        return []
    dtype = np.float32 if method.upper() == "SIFT" else np.uint8
    # o FLANN e aleatorio; a semente fixa (por thread) deixa o resultado reprodutivel
    cv2.setRNGSeed(0)
    pairs = create_matcher(method).knnMatch(descs1.astype(dtype), descs2.astype(dtype), k=2)
    return [p for p in pairs if len(p) == 2]

def match_features(descs1, descs2, method="SIFT", ratio_thresh=0.75):
    good = [m for m, n in knn_match(descs1, descs2, method) if m.distance < ratio_thresh * n.distance]
    good.sort(key=lambda m: m.distance)
    return good

def draw_matches(img1, kps1, img2, kps2, matches, out_path=None, max_draw=100):
    vis = cv2.drawMatches(
        img1, kps1, img2, kps2, matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if out_path:
        cv2.imwrite(out_path, vis)
    return vis

def draw_matches_before_after(img1, kps1, img2, kps2, all_matches, good, inlier_mask,
                              out_prefix, max_draw=150, seed=0):
    rng = np.random.default_rng(seed)

    def sample(matches):
        idx = rng.choice(len(matches), min(max_draw, len(matches)), replace=False) if matches else []
        return [matches[i] for i in idx]

    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    before = cv2.drawMatches(img1, kps1, img2, kps2, sample(all_matches), None,
                             matchColor=(0, 255, 255), flags=flags)
    cv2.imwrite(f"{out_prefix}_antes.jpg", before)

    mask = inlier_mask if inlier_mask is not None else np.ones(len(good), bool)
    idx = rng.choice(len(good), min(max_draw, len(good)), replace=False) if good else []
    inliers = [good[i] for i in idx if mask[i]]
    outliers = [good[i] for i in idx if not mask[i]]
    after = cv2.drawMatches(img1, kps1, img2, kps2, inliers, None, matchColor=(0, 200, 0), flags=flags)
    after = cv2.drawMatches(img1, kps1, img2, kps2, outliers, after, matchColor=(0, 0, 255),
                            flags=flags | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    cv2.imwrite(f"{out_prefix}_depois.jpg", after)
