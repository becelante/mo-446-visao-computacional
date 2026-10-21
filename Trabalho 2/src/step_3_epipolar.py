"""
Geometria epipolar entre pares de imagens.

Estima a matriz fundamental F e a matriz essencial E com RANSAC, recupera a
pose relativa (R, t) da segunda camera em relacao a primeira e triangula os
pontos 3D iniciais.

  - F (3x3, posto 2) relaciona pixels: x2^T F x1 = 0. Nao precisa da calibracao.
  - E (3x3) relaciona coordenadas normalizadas: E = K^T F K = [t]x R. Precisa
    da matriz intrinseca K e, decomposta, da a rotacao R e a direcao de t.
"""
import importlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

# modulos que comecam com digito nao podem ser importados com "from .2_matching import"
match_features = importlib.import_module(".2_matching", __package__).match_features


def build_intrinsics(width, height, focal_px=None):
    """
    Matriz intrinseca K com o ponto principal no centro da imagem. Sem
    `focal_px` (p.ex., do EXIF: focal_mm / largura_sensor_mm * width), usa
    1.2 * max(width, height), a mesma estimativa inicial do COLMAP.
    """
    f = focal_px if focal_px is not None else 1.2 * max(width, height)
    return np.array([[f, 0, width / 2],
                     [0, f, height / 2],
                     [0, 0, 1]], dtype=np.float64)


def matched_points(kps1, kps2, matches):
    """Coordenadas (N x 2) dos keypoints emparelhados em cada imagem."""
    pts1 = np.float64([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def estimate_fundamental(pts1, pts2, ransac_thresh=1.0, confidence=0.999):
    """
    Estima F com RANSAC (algoritmo de 8 pontos). Retorna (F, mascara_inliers);
    inlier e o par cuja distancia a linha epipolar fica abaixo de `ransac_thresh` px.
    """
    if len(pts1) < 8:
        return None, None
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_thresh, confidence)
    if F is None or F.shape != (3, 3):  # pode devolver varias solucoes empilhadas
        return None, None
    return F, mask.ravel().astype(bool)


def estimate_essential(pts1, pts2, K, ransac_thresh=1.0, confidence=0.999):
    """Estima E com RANSAC (algoritmo de 5 pontos de Nister). Retorna (E, mascara_inliers)."""
    if len(pts1) < 5:
        return None, None
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, confidence, ransac_thresh)
    if E is None or E.shape != (3, 3):
        return None, None
    return E, mask.ravel().astype(bool)


def fundamental_from_essential(E, K):
    """F = K^-T E K^-1, util para desenhar as linhas epipolares coerentes com a pose."""
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    return F / F[2, 2]


def recover_pose(E, pts1, pts2, K, mask=None):
    """
    Decompoe E nas 4 solucoes (R, t) possiveis e escolhe a que deixa os pontos na
    frente das duas cameras (teste de quiralidade). Retorna (R, t, mascara), em que
    a mascara marca os inliers que passaram no teste. t tem norma 1: a escala so e
    recuperada com a referencia metrica (Etapa 4).
    """
    mask_in = None if mask is None else mask.astype(np.uint8).reshape(-1, 1) * 255
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask_in)
    return R, t, pose_mask.ravel() > 0


def projection_matrices(K, R, t):
    """P1 = K [I | 0] (camera 1 na origem) e P2 = K [R | t]."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    return P1, P2


def reprojection_errors(points3d, pts, K, R, t):
    """Distancia em pixels entre cada ponto 3D projetado na camera (R, t) e sua observacao."""
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(points3d, rvec, t.astype(np.float64), K, None)
    return np.linalg.norm(projected.reshape(-1, 2) - pts, axis=1)


def triangulate(pts1, pts2, K, R, t, max_reproj_error=4.0):
    """
    Triangula (DLT) os pares de pontos com P1 = K[I|0] e P2 = K[R|t]. Mantem so os
    pontos com profundidade positiva nas duas cameras e erro de reprojecao abaixo
    de `max_reproj_error` px nas duas imagens. Retorna (pontos3d N x 3, mascara).
    """
    if len(pts1) == 0:
        return np.empty((0, 3)), np.zeros(0, bool)
    P1, P2 = projection_matrices(K, R, t)
    X = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points3d = (X[:3] / X[3]).T

    depth1 = points3d[:, 2]
    depth2 = (points3d @ R.T + t.reshape(1, 3))[:, 2]
    err1 = reprojection_errors(points3d, pts1, K, np.eye(3), np.zeros(3))
    err2 = reprojection_errors(points3d, pts2, K, R, t)
    valid = (depth1 > 0) & (depth2 > 0) & (err1 < max_reproj_error) & (err2 < max_reproj_error)
    return points3d[valid], valid


def two_view_geometry(kps1, descs1, kps2, descs2, K, method="SIFT", ratio_thresh=0.75,
                      ransac_thresh=1.0, min_matches=15):
    """
    Pipeline de duas vistas: matching -> F e E com RANSAC -> (R, t) -> triangulacao.
    Retorna um dict com os matches aprovados no ratio test (`good`), F, E, R, t, as
    mascaras alinhadas com `good` (`mask_F`, `mask_E`, `mask_pose`), os pontos 3D e
    metricas; R, t e os pontos ficam None se o par nao tiver geometria suficiente.
    """
    good = match_features(descs1, descs2, method, ratio_thresh)
    result = {"good": good, "F": None, "E": None, "R": None, "t": None,
              "mask_F": None, "mask_E": None, "mask_pose": None, "points3d": None,
              "metrics": {"n_matches": len(good), "inliers_F": 0, "inliers_E": 0,
                          "inliers_pose": 0, "n_points3d": 0,
                          "reproj_mean": float("nan"), "reproj_median": float("nan")}}
    if len(good) < min_matches:
        return result

    pts1, pts2 = matched_points(kps1, kps2, good)
    F, mask_F = estimate_fundamental(pts1, pts2, ransac_thresh)
    E, mask_E = estimate_essential(pts1, pts2, K, ransac_thresh)
    result.update(F=F, E=E, mask_F=mask_F, mask_E=mask_E)
    result["metrics"]["inliers_F"] = int(mask_F.sum()) if F is not None else 0
    if E is None:
        return result
    result["metrics"]["inliers_E"] = int(mask_E.sum())

    R, t, mask_pose = recover_pose(E, pts1, pts2, K, mask_E)
    points3d, valid = triangulate(pts1[mask_pose], pts2[mask_pose], K, R, t)
    errors = np.concatenate([reprojection_errors(points3d, pts1[mask_pose][valid], K, np.eye(3), np.zeros(3)),
                             reprojection_errors(points3d, pts2[mask_pose][valid], K, R, t)])
    result.update(R=R, t=t, mask_pose=mask_pose, points3d=points3d)
    result["metrics"].update(inliers_pose=int(mask_pose.sum()), n_points3d=len(points3d),
                             reproj_mean=float(errors.mean()) if len(errors) else float("nan"),
                             reproj_median=float(np.median(errors)) if len(errors) else float("nan"))
    return result


def build_connectivity_matrix(all_kps, all_descs, K, method="SIFT", ratio_thresh=0.75,
                              ransac_thresh=1.0, min_matches=15):
    """
    Constroi a matriz N x N com o numero de inliers de E entre cada par de imagens
    (pares processados em paralelo). Tambem retorna o resultado de
    `two_view_geometry` de cada par (geometries[(i, j)] com i < j); o par com mais
    pontos triangulados e um bom candidato a par inicial do SfM.
    """
    n = len(all_kps)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def process_pair(pair):
        i, j = pair
        return two_view_geometry(all_kps[i], all_descs[i], all_kps[j], all_descs[j], K,
                                 method, ratio_thresh, ransac_thresh, min_matches)

    connectivity = np.zeros((n, n), dtype=int)
    geometries = {}
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        for (i, j), result in zip(pairs, executor.map(process_pair, pairs)):
            connectivity[i, j] = connectivity[j, i] = result["metrics"]["inliers_E"]
            geometries[(i, j)] = result

    return connectivity, geometries


def draw_epipolar_lines(img1, img2, pts1, pts2, F, out_path=None, max_draw=20, seed=0):
    """
    Desenha, lado a lado, uma amostra de pontos e suas linhas epipolares: a linha
    em img2 de cada ponto de img1 (l2 = F x1) e vice-versa (l1 = F^T x2). Com F
    correta, cada ponto cai sobre a linha da mesma cor na outra imagem.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts1), min(max_draw, len(pts1)), replace=False)
    colors = [tuple(int(c) for c in rng.integers(0, 255, 3)) for _ in idx]
    vis1, vis2 = img1.copy(), img2.copy()
    thickness = max(1, img1.shape[1] // 800)

    def draw(img, lines, pts, color_list):
        w = img.shape[1]
        for (a, b, c), (x, y), color in zip(lines, pts, color_list):
            if abs(b) > 1e-12:
                cv2.line(img, (0, int(-c / b)), (w, int(-(c + a * w) / b)), color, thickness, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 4 * thickness, color, -1, cv2.LINE_AA)

    lines2 = cv2.computeCorrespondEpilines(pts1[idx].reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines1 = cv2.computeCorrespondEpilines(pts2[idx].reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    draw(vis1, lines1, pts1[idx], colors)
    draw(vis2, lines2, pts2[idx], colors)

    h = max(vis1.shape[0], vis2.shape[0])
    vis = np.hstack([cv2.copyMakeBorder(v, 0, h - v.shape[0], 0, 0, cv2.BORDER_CONSTANT) for v in (vis1, vis2)])
    if out_path:
        cv2.imwrite(out_path, vis)
    return vis


def save_point_cloud(points3d, out_path, colors=None):
    """Salva a nuvem em PLY (ASCII), que abre no Open3D e no MeshLab. `colors` em BGR (0-255)."""
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points3d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for k, (x, y, z) in enumerate(points3d):
            line = f"{x} {y} {z}"
            if colors is not None:
                b, g, r = colors[k]
                line += f" {int(r)} {int(g)} {int(b)}"
            f.write(line + "\n")
