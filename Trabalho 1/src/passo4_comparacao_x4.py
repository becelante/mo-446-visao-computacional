"""Extra X4 - SuperPoint + LightGlue (GPU), executado por 4_comparacao_x4.py.

compare_matchers: SIFT com ratio test x SuperPoint + LightGlue nos pares consecutivos de um conjunto
(inliers, erro de reprojecao e tempo por par).

compare_stitcher: cv2.Stitcher (ORB) x SuperPoint + LightGlue com a MESMA composicao, a do cv2.Stitcher
no modo PANORAMA (homografias -> bundle adjustment de focal e rotacao -> correcao de onda -> projecao
esferica -> ganho por blocos -> costura por graph-cut -> blending multibanda), refeita com cv2.detail
porque o binding Python do Stitcher nao deixa trocar o emparelhador; com o ORB ela reproduz o
cv2.Stitcher. O emparelhador do Stitcher e aleatorio, entao cada metodo roda com varias sementes e as
metricas sao a mediana entre elas. O desalinhamento e medido com as correspondencias do pipeline
principal (SIFT, ratio test e RANSAC de 1 px), que nao entram em nenhum dos dois metodos.
"""
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
from .etapa5_homography import estimate_homography
from .passo1_raw_convert import original_names

REGISTRATION_MP, SEAM_MP, CONF_THRESH = 0.6, 0.1, 1.0  # defaults do cv2.Stitcher

def load_lightglue():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo do LightGlue: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    extractor = SuperPoint(max_num_keypoints=8192).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher, device

def superpoint(img, extractor, device):
    gray = torch.from_numpy(to_gray8(img, equalize=True)).float()[None].to(device) / 255
    return extractor.extract(gray, resize=None)

# ---------------------------------------------------------------- SIFT x LightGlue por par
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
    return pts_a, pts_b, elapsed

def match_lightglue(img_a, img_b, extractor, matcher, device):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        feats_a, feats_b = superpoint(img_a, extractor, device), superpoint(img_b, extractor, device)
        matches = matcher({"image0": feats_a, "image1": feats_b})
    feats_a, feats_b, matches = rbd(feats_a), rbd(feats_b), rbd(matches)
    idx = matches["matches"].cpu().numpy()
    pts_a = feats_a["keypoints"].cpu().numpy()[idx[:, 0]].astype(np.float32)
    pts_b = feats_b["keypoints"].cpu().numpy()[idx[:, 1]].astype(np.float32)
    torch.cuda.synchronize()
    return pts_a, pts_b, time.perf_counter() - t0

def ransac_metrics(pts_a, pts_b, scale):
    """Inliers do RANSAC e erro de reprojecao medio (px da foto original)."""
    if len(pts_a) < 4:
        return 0, float("nan")
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, config.RANSAC_THRESH)
    if H is None:
        return 0, float("nan")
    mask = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(pts_a[mask].reshape(-1, 1, 2), H).reshape(-1, 2)
    return int(mask.sum()), float(np.linalg.norm(proj - pts_b[mask], axis=1).mean()) / scale

def compare_matchers(png_dir, results_dir, extractor, matcher, device):
    with open(results_dir / "5_alinhamento" / "metricas.csv") as f:
        pairs = [(row["imagem_1"], row["imagem_2"]) for row in csv.DictReader(f)]
    cache = {}

    def image(name):
        if name not in cache:
            cache[name] = load(png_dir / name, config.WORK_DIM)
        return cache[name]

    methods = {"SIFT + ratio test (CPU)": match_sift,
               "SuperPoint + LightGlue (GPU)": lambda a, b: match_lightglue(a, b, extractor, matcher, device)}
    methods["SuperPoint + LightGlue (GPU)"](*[image(n)[0] for n in pairs[0]])  # aquece a GPU
    rows = {name: [] for name in methods}
    for k, (a, b) in enumerate(pairs):
        (img_a, scale), (img_b, _) = image(a), image(b)
        for name, run in methods.items():
            pts_a, pts_b, elapsed = run(img_a, img_b)
            rows[name].append((*ransac_metrics(pts_a, pts_b, scale), elapsed))
        print(f"  par {k}: " + ", ".join(f"{name.split()[0]} {r[-1][0]} inliers" for name, r in rows.items()))
    return [{"metodo": name, **{key: round(float(v), 3) for key, v in
                                zip(("inliers", "erro_reproj_px", "tempo_s"), np.nanmean(r, axis=0))}}
            for name, r in rows.items()]

# ---------------------------------------------------------------- cv2.Stitcher x LightGlue
def match_orb(imgs):
    """Emparelhamento padrao do cv2.Stitcher: ORB + BestOf2NearestMatcher."""
    finder, features = cv2.ORB_create(), []
    for i, img in enumerate(imgs):
        f = cv2.detail.computeImageFeatures2(finder, img)
        f.img_idx = i
        features.append(f)
    return features, list(cv2.detail_BestOf2NearestMatcher(False, 0.3).apply2(features))

def _matches_info(i, j, kp_i, kp_j, m, size_i, size_j):
    """MatchesInfo do par (i, j) como o BestOf2NearestMatcher monta: homografia por RANSAC em
    coordenadas centradas, confianca n_i / (8 + 0,3 n_m) e reestimacao so com os inliers."""
    mi = cv2.detail.MatchesInfo()
    mi.src_img_idx, mi.dst_img_idx = i, j
    mi.matches = tuple(cv2.DMatch(int(a), int(b), 0.0) for a, b in m)
    if len(m) < 6:
        return mi
    src = kp_i[m[:, 0]] - np.float32(size_i) / 2
    dst = kp_j[m[:, 1]] - np.float32(size_j) / 2
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None or abs(np.linalg.det(H)) < np.finfo(float).eps:
        return mi
    mi.inliers_mask = mask.ravel().astype(np.uint8)
    mi.num_inliers = int(mask.sum())
    conf = mi.num_inliers / (8 + 0.3 * len(m))
    mi.confidence = 0.0 if conf > 3 else conf  # > 3: fotos quase iguais, como no OpenCV
    if mi.num_inliers >= 6:
        inl = mask.ravel().astype(bool)
        H = cv2.findHomography(src[inl], dst[inl], cv2.RANSAC)[0]
    mi.H = H
    return mi

def _flip(mi, i, j):
    """Par (j, i) a partir de (i, j)."""
    out = cv2.detail.MatchesInfo()
    out.src_img_idx, out.dst_img_idx = j, i
    out.matches = tuple(cv2.DMatch(d.trainIdx, d.queryIdx, d.distance) for d in mi.matches)
    out.num_inliers, out.confidence = mi.num_inliers, mi.confidence
    if mi.num_inliers:
        out.inliers_mask, out.H = mi.inliers_mask, np.linalg.inv(mi.H)
    return out

def match_lightglue_all(imgs, extractor, matcher, device):
    """SuperPoint + LightGlue em todos os pares, no formato do cv2.detail."""
    n = len(imgs)
    with torch.inference_mode():
        feats = [superpoint(img, extractor, device) for img in imgs]
        pair_matches = {(i, j): rbd(matcher({"image0": feats[i], "image1": feats[j]}))["matches"].cpu().numpy()
                        for i in range(n) for j in range(i + 1, n)}
    kps = [rbd(f)["keypoints"].cpu().numpy().astype(np.float32) for f in feats]
    sizes = [(img.shape[1], img.shape[0]) for img in imgs]
    features = []
    for i, kp in enumerate(kps):
        f = cv2.detail.ImageFeatures()
        f.img_idx, f.img_size = i, sizes[i]
        f.keypoints = tuple(cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kp)
        features.append(f)
    pairwise = [cv2.detail.MatchesInfo() for _ in range(n * n)]
    for (i, j), m in pair_matches.items():
        pairwise[i * n + j] = _matches_info(i, j, kps[i], kps[j], m, sizes[i], sizes[j])
        pairwise[j * n + i] = _flip(pairwise[i * n + j], i, j)
    return features, pairwise

def _subset(features, pairwise, idx):
    """Reindexa features e matches para as fotos mantidas (no C++ o leaveBiggestComponent faz isso no
    lugar; no Python as listas sao copias)."""
    n = len(features)
    feats = []
    for new, old in enumerate(idx):
        f = cv2.detail.ImageFeatures()
        f.img_idx, f.img_size, f.keypoints = new, features[old].img_size, features[old].keypoints
        feats.append(f)
    pw = []
    for a, oa in enumerate(idx):
        for b, ob in enumerate(idx):
            s, m = pairwise[oa * n + ob], cv2.detail.MatchesInfo()
            if s.src_img_idx >= 0:
                m.src_img_idx, m.dst_img_idx = a, b
            m.matches, m.num_inliers, m.confidence = s.matches, s.num_inliers, s.confidence
            if s.num_inliers:
                m.inliers_mask, m.H = s.inliers_mask, s.H
            pw.append(m)
    return feats, pw

def sift_reference(images, labels):
    """Correspondencias do pipeline principal entre fotos consecutivas na ordem de captura."""
    order = sorted(range(len(labels)), key=labels.__getitem__)
    feats = {i: detect_features(images[i], "SIFT") for i in order}
    ref = {}
    for a, b in zip(order, order[1:]):
        (kps_a, descs_a), (kps_b, descs_b) = feats[a], feats[b]
        good = match_features(descs_a, descs_b, "SIFT", config.RATIO_THRESH)
        _, mask, _ = estimate_homography(kps_a, kps_b, good, config.BASE_RANSAC_THRESH)
        if mask is not None:
            ref[a, b] = (np.float32([kps_a[m.queryIdx].pt for m in good])[mask],
                         np.float32([kps_b[m.trainIdx].pt for m in good])[mask])
    return ref

def stitch(full_imgs, matcher_fn, ref):
    """Composicao do cv2.Stitcher (modo PANORAMA) com o emparelhamento de matcher_fn. Alem do panorama,
    devolve as fotos usadas, a projecao foto -> panorama, o desalinhamento de cada correspondencia de
    ref e a media de inliers do emparelhador nesses mesmos pares."""
    h, w = full_imgs[0].shape[:2]
    work_scale = min(1.0, np.sqrt(REGISTRATION_MP * 1e6 / (w * h)))
    seam_scale = min(1.0, np.sqrt(SEAM_MP * 1e6 / (w * h)))
    seam_aspect, compose_aspect = seam_scale / work_scale, 1.0 / work_scale
    features, pairwise = matcher_fn([cv2.resize(im, None, fx=work_scale, fy=work_scale,
                                                interpolation=cv2.INTER_LINEAR_EXACT) for im in full_imgs])

    idx = [int(i) for i in np.ravel(cv2.detail.leaveBiggestComponent(features, pairwise, CONF_THRESH))]
    features, pairwise = _subset(features, pairwise, idx)
    full_imgs = [full_imgs[i] for i in idx]
    ok, cameras = cv2.detail_HomographyBasedEstimator().apply(features, pairwise, None)
    if ok:
        for c in cameras:
            c.R = c.R.astype(np.float32)
        adjuster = cv2.detail_BundleAdjusterRay()
        adjuster.setConfThresh(CONF_THRESH)
        ok, cameras = adjuster.apply(features, pairwise, cameras)
    if not ok:
        raise RuntimeError("falha na estimacao das cameras")
    scale = float(np.median([c.focal for c in cameras]))
    for c, R in zip(cameras, cv2.detail.waveCorrect([np.copy(c.R) for c in cameras], cv2.detail.WAVE_CORRECT_HORIZ)):
        c.R = R

    def warp(warper, img, K, R):
        corner, img_w = warper.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        _, mask_w = warper.warp(np.full(img.shape[:2], 255, np.uint8), K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        return corner, img_w, mask_w

    # ganho e costura na resolucao de costura (o Stitcher compensa o ganho antes da costura)
    warper = cv2.PyRotationWarper("spherical", scale * seam_aspect)
    corners, imgs_w, masks_w = [], [], []
    for img, cam in zip(full_imgs, cameras):
        K = cam.K().astype(np.float32)
        K[:2] *= seam_aspect
        img = cv2.resize(img, None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        corner, img_w, mask_w = warp(warper, img, K, cam.R)
        corners.append(corner); imgs_w.append(img_w); masks_w.append(mask_w)
    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners=corners, images=imgs_w, masks=masks_w)
    imgs_w = [compensator.apply(k, corners[k], im, masks_w[k]) for k, im in enumerate(imgs_w)]
    seams = cv2.detail_GraphCutSeamFinder("COST_COLOR").find([im.astype(np.float32) for im in imgs_w],
                                                             corners, masks_w)

    # composicao na resolucao original, com blending multibanda
    warper = cv2.PyRotationWarper("spherical", scale * compose_aspect)
    for cam in cameras:
        cam.focal *= compose_aspect
        cam.ppx *= compose_aspect
        cam.ppy *= compose_aspect
    rois = [warper.warpRoi((img.shape[1], img.shape[0]), cam.K().astype(np.float32), cam.R)
            for img, cam in zip(full_imgs, cameras)]
    dst_roi = cv2.detail.resultRoi(corners=[r[:2] for r in rois], sizes=[r[2:] for r in rois])
    blender = cv2.detail_MultiBandBlender()
    blender.prepare(dst_roi)
    for k, (img, cam) in enumerate(zip(full_imgs, cameras)):
        corner, img_w, mask_w = warp(warper, img, cam.K().astype(np.float32), cam.R)
        img_w = compensator.apply(k, corner, img_w, mask_w)
        seam = cv2.resize(cv2.dilate(seams[k], None), (mask_w.shape[1], mask_w.shape[0]), 0, 0,
                          cv2.INTER_LINEAR_EXACT)
        blender.feed(cv2.UMat(img_w.astype(np.int16)), cv2.bitwise_and(seam, mask_w), corner)
    pano = np.clip(blender.blend(None, None)[0], 0, 255).astype(np.uint8)

    pos, origin = {old: new for new, old in enumerate(idx)}, np.float64(dst_roi[:2])

    def project(i, pts):
        cam = cameras[pos[i]]
        K = cam.K().astype(np.float32)
        return np.array([warper.warpPoint((float(x), float(y)), K, cam.R) for x, y in pts]) - origin

    used = [(a, b) for a, b in ref if a in pos and b in pos]
    return {"panorama": pano, "fotos": idx, "projetar": project,
            "erros": np.concatenate([np.linalg.norm(project(a, ref[a, b][0]) - project(b, ref[a, b][1]), axis=1)
                                     for a, b in used]),
            "inliers": np.mean([pairwise[pos[a] * len(idx) + pos[b]].num_inliers for a, b in used])}

def crop_around(img, center, size=(450, 400)):
    w, h = size
    x0 = int(np.clip(center[0] - w / 2, 0, img.shape[1] - w))
    y0 = int(np.clip(center[1] - h / 2, 0, img.shape[0] - h))
    return img[y0:y0 + h, x0:x0 + w]

def compare_stitcher(png_dir, out_dir, extractor, matcher, device):
    # o resultado do cv2.Stitcher depende da ordem de entrada; usamos a do nomes_originais.csv
    names = original_names(png_dir)
    images = [cv2.imread(str(png_dir / f)) for f in names]
    labels = list(names.values())
    ref = sift_reference(images, labels)
    methods = {"cv2.Stitcher (ORB)": ("stitcher_orb", match_orb),
               "SuperPoint + LightGlue (GPU)": ("superpoint_lightglue",
                                                lambda imgs: match_lightglue_all(imgs, extractor, matcher, device))}
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)  # o RNG do OpenCV e por thread: com varias, nem o cv2.Stitcher e reprodutivel
    anchor_img, anchor_pt = config.X4_DETAIL_ANCHOR
    summary = []
    try:
        for name, (key, matcher_fn) in methods.items():
            runs = []
            for seed in range(config.X4_SEEDS):
                cv2.setRNGSeed(seed)
                runs.append({"semente": seed, **stitch(images, matcher_fn, ref)})
            run = runs[config.X4_FIGURE_SEED]
            cv2.imwrite(str(out_dir / f"panorama_{key}.png"), run["panorama"])
            center = run["projetar"](labels.index(anchor_img), [anchor_pt])[0]
            cv2.imwrite(str(out_dir / f"detalhe_{key}.png"), crop_around(run["panorama"], center))
            if key == "stitcher_orb":
                cv2.setRNGSeed(run["semente"])
                status, real = cv2.Stitcher_create(cv2.Stitcher_PANORAMA).stitch(images)
                if status == cv2.Stitcher_OK and real.shape == run["panorama"].shape:
                    diff = f"difere em {np.abs(real.astype(int) - run['panorama'].astype(int)).mean():.3f} niveis de cinza"
                else:
                    diff = f"NAO bate (status {status}, {None if real is None else real.shape})"
                print(f"  validacao: o cv2.Stitcher de verdade, com a mesma semente, {diff}")
            medians = [np.median(r["erros"]) for r in runs]
            inliers = np.median([r["inliers"] for r in runs])
            print(f"  {name}: {len(run['fotos'])} fotos, {inliers:.0f} inliers por par, desalinhamento mediano "
                  f"{np.median(medians):.2f} px ({min(medians):.2f} a {max(medians):.2f} px em {len(runs)} sementes; "
                  f"{medians[config.X4_FIGURE_SEED]:.2f} px na figura)")
            summary.append({"metodo": name, "inliers": round(float(inliers), 1),
                            "erro_alinhamento_px": round(float(np.median(medians)), 3),
                            "erro_min_px": round(float(min(medians)), 3), "erro_max_px": round(float(max(medians)), 3)})
    finally:
        cv2.setNumThreads(threads)
    return summary
