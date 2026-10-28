"""
Etapa 4d (recomendada na Variante A): densificacao (MVS).

Dois metodos:
  - patchmatch: PatchMatch Stereo + fusao do COLMAP (o padrao do COLMAP; exige CUDA).
  - sgbm: alternativa em CPU. Para cada imagem de referencia escolhe um vizinho com
    base lateral adequada, retifica o par com as poses do SfM (cv2.stereoRectify),
    calcula a disparidade com Semi-Global Block Matching e reprojeta em 3D. A fusao
    mantem so os voxels confirmados por pelo menos dois pares diferentes (consistencia
    entre vistas), o que elimina boa parte do ruido do SGBM.
Os dois rodam sobre o modelo ja metrico e alinhado, entao a nuvem densa sai no
mesmo referencial (metros, z para cima).
"""
import os
import shutil

import cv2
import numpy as np
import pycolmap

from .step_4_sfm import image_pose, registered_images, run_colmap
from .viz_utils import read_ply


def undistort(sparse_dir, image_dir, work_dir, max_image_size=1200):
    """Remove a distorcao (camera PINHOLE) e reduz as imagens para o MVS."""
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = max_image_size
    pycolmap.undistort_images(work_dir, sparse_dir, image_dir, output_type="COLMAP", undistort_options=opts)
    return pycolmap.Reconstruction(os.path.join(work_dir, "sparse"))


def dense_patchmatch(sparse_dir, image_dir, work_dir, max_image_size=1200, colmap_exe=None, log=print):
    log(f"  PatchMatch Stereo (COLMAP, CUDA{' via colmap.exe' if colmap_exe else ''})...")
    undistort(sparse_dir, image_dir, work_dir, max_image_size)
    fused = os.path.join(work_dir, "fused.ply")
    if colmap_exe:
        log_path = os.path.join(work_dir, "colmap.log")
        run_colmap(colmap_exe, "patch_match_stereo", {
            "workspace_path": work_dir, "PatchMatchStereo.max_image_size": max_image_size,
            "PatchMatchStereo.geom_consistency": True}, log_path)
        log("  fusao dos mapas de profundidade...")
        run_colmap(colmap_exe, "stereo_fusion", {
            "workspace_path": work_dir, "input_type": "geometric", "output_type": "PLY",
            "output_path": fused}, log_path)
        return read_ply(fused)
    pm = pycolmap.PatchMatchOptions()
    pm.max_image_size = max_image_size
    pm.geom_consistency = True
    pycolmap.patch_match_stereo(work_dir, options=pm)
    pycolmap.stereo_fusion(fused, work_dir, input_type="geometric", output_type="PLY")
    return read_ply(fused)


# ----------------------------------------------------------------------------------------
# Alternativa em CPU: SGBM em pares retificados
# ----------------------------------------------------------------------------------------

def _choose_pairs(rec, num_refs, min_shared=40):
    imgs = registered_images(rec)
    pose = {im.image_id: image_pose(im) for im in imgs}
    pts_of = {im.image_id: {p.point3D_id for p in im.points2D if p.has_point3D()} for im in imgs}
    step = max(1, len(imgs) // num_refs)
    pairs = []
    for i, ref in enumerate(imgs[::step]):
        Ri, ti = pose[ref.image_id]
        Ci, ai = -Ri.T @ ti, Ri[2]
        ids_i = pts_of[ref.image_id]
        if len(ids_i) < min_shared:
            continue
        best, best_score = None, -1
        for cand in imgs:
            if cand.image_id == ref.image_id:
                continue
            shared = ids_i & pts_of[cand.image_id]
            if len(shared) < min_shared:
                continue
            Rj, tj = pose[cand.image_id]
            b = (-Rj.T @ tj) - Ci
            X = np.array([rec.point3D(k).xyz for k in list(shared)[:200]])
            depth = np.median((X @ Ri.T + ti)[:, 2])
            ratio = np.linalg.norm(b) / max(depth, 1e-9)
            lateral = abs(b @ ai) / max(np.linalg.norm(b), 1e-9)  # componente ao longo do eixo optico
            angle = np.degrees(np.arccos(np.clip(ai @ Rj[2], -1, 1)))
            if not (0.04 < ratio < 0.35 and lateral < 0.6 and angle < 25):
                continue
            score = len(shared) * (1 - abs(ratio - 0.12))
            if score > best_score:
                best, best_score = cand, score
        if best is not None:
            pairs.append((ref, best))
    return pairs


def _stereo_pair(rec, img_dir, ref, nb, step_px=2):
    K = np.asarray(rec.camera(ref.camera_id).calibration_matrix())
    cam = rec.camera(ref.camera_id)
    size = (cam.width, cam.height)
    Ri, ti = image_pose(ref)
    Rj, tj = image_pose(nb)
    R_rel = Rj @ Ri.T
    T_rel = (tj - R_rel @ ti).reshape(3, 1)  # OpenCV 5 exige T como coluna (3x1)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, None, K, None, size, R_rel, T_rel,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if abs(P2[1, 3]) > abs(P2[0, 3]):  # par vertical: SGBM so trata disparidade horizontal
        return None
    if P2[0, 3] > 0:  # vizinho a esquerda: inverte os papeis
        return _stereo_pair(rec, img_dir, nb, ref, step_px)
    m1 = cv2.initUndistortRectifyMap(K, None, R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K, None, R2, P2, size, cv2.CV_32FC1)
    I1 = cv2.imread(os.path.join(img_dir, ref.name))
    I2 = cv2.imread(os.path.join(img_dir, nb.name))
    L = cv2.remap(I1, *m1, cv2.INTER_LINEAR)
    Rr = cv2.remap(I2, *m2, cv2.INTER_LINEAR)
    valid_l = cv2.remap(np.full(I1.shape[:2], 255, np.uint8), *m1, cv2.INTER_NEAREST) > 0

    # faixa de disparidade a partir dos pontos esparsos vistos pela referencia
    f, B = P1[0, 0], -P2[0, 3] / P2[0, 0]
    ids = [p.point3D_id for p in ref.points2D if p.has_point3D()]
    X = np.array([rec.point3D(k).xyz for k in ids])
    z_rect = ((X @ Ri.T + ti) @ R1.T)[:, 2]
    z_rect = z_rect[z_rect > 0]
    if len(z_rect) < 20:
        return None
    zlo, zhi = np.percentile(z_rect, [2, 98])
    dmin, dmax = f * B / (zhi * 1.4), f * B / (zlo * 0.7)
    min_disp = int(np.floor(dmin))
    num_disp = int(np.ceil((dmax - min_disp) / 16.0)) * 16
    num_disp = int(np.clip(num_disp, 16, 320))
    if min_disp + num_disp >= 0.75 * size[0]:  # retificacao degenerada (sobra pouca imagem para casar)
        return None
    bs = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=bs,
                                 P1=8 * 3 * bs * bs, P2=32 * 3 * bs * bs, disp12MaxDiff=1,
                                 uniquenessRatio=12, speckleWindowSize=120, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(L, Rr).astype(np.float32) / 16.0
    good = (disp > min_disp + 0.5) & valid_l
    good[:, :num_disp] = False  # faixa sem correspondencia possivel
    pts = cv2.reprojectImageTo3D(disp, Q)
    sel = np.zeros_like(good)
    sel[::step_px, ::step_px] = True
    good &= sel
    Xr = pts[good].astype(np.float64)
    z = Xr[:, 2]
    keep = (z > zlo * 0.7) & (z < zhi * 1.4)
    Xr = Xr[keep]
    col = L[good][keep][:, ::-1]  # BGR -> RGB
    Xc1 = Xr @ R1  # (R1^T x) em linhas
    Xw = (Xc1 - ti) @ Ri  # R^T (x - t)
    return Xw, col


def dense_sgbm(sparse_dir, image_dir, work_dir, max_image_size=1200, num_refs=60, voxel=0.015,
               min_pairs=2, log=print):
    log("  undistorcao das imagens...")
    rec = undistort(sparse_dir, image_dir, work_dir, max_image_size)
    img_dir = os.path.join(work_dir, "images")
    pairs = _choose_pairs(rec, num_refs)
    log(f"  SGBM em {len(pairs)} pares retificados...")
    all_x, all_c, all_pair = [], [], []
    for k, (ref, nb) in enumerate(pairs):
        r = _stereo_pair(rec, img_dir, ref, nb)
        if r is None or len(r[0]) == 0:
            continue
        all_x.append(r[0].astype(np.float32))
        all_c.append(r[1])
        all_pair.append(np.full(len(r[0]), k, np.int32))
    if not all_x:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    X = np.concatenate(all_x)
    C = np.concatenate(all_c)
    P = np.concatenate(all_pair)
    # fusao: voxel confirmado por >= min_pairs pares distintos
    keys = np.floor(X / voxel).astype(np.int64)
    keys -= keys.min(0)
    dims = keys.max(0) + 1
    lin = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    uniq_lp = np.unique(np.stack([lin, P.astype(np.int64)], 1), axis=0)
    vox, n_pairs = np.unique(uniq_lp[:, 0], return_counts=True)
    ok_vox = vox[n_pairs >= min_pairs]
    keep = np.isin(lin, ok_vox)
    X, C, lin = X[keep], C[keep], lin[keep]
    u, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    xs = np.zeros((len(u), 3))
    cs = np.zeros((len(u), 3))
    np.add.at(xs, inv, X)
    np.add.at(cs, inv, C.astype(float))
    log(f"  fusao: {len(u)} voxels de {voxel * 100:.1f} cm confirmados por >= {min_pairs} pares")
    return xs / cnt[:, None], (cs / cnt[:, None]).astype(np.uint8)


def densify(sparse_dir, image_dir, work_dir, method="auto", max_image_size=1200, num_refs=60, log=print,
            colmap_exe=None):
    """Retorna (xyz, rgb, metodo_usado). `colmap_exe`: PatchMatch pelo executavel com CUDA."""
    if method == "none":
        return None, None, "none"
    has_cuda = pycolmap.has_cuda or colmap_exe is not None
    if method == "auto":
        method = "patchmatch" if has_cuda else "sgbm"
    if method == "patchmatch":
        if not has_cuda:
            log("  AVISO: pycolmap sem CUDA e sem colmap.exe; usando SGBM em CPU")
            method = "sgbm"
        else:
            xyz, rgb = dense_patchmatch(sparse_dir, image_dir, work_dir, max_image_size,
                                        None if pycolmap.has_cuda else colmap_exe, log)
            return xyz, rgb, method
    xyz, rgb = dense_sgbm(sparse_dir, image_dir, work_dir, max_image_size, num_refs, log=log)
    return xyz, rgb, "sgbm"
