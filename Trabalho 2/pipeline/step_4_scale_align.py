"""
Etapa 4b: escala metrica e alinhamento da sala.

Escala: o SfM so recupera a cena a menos de uma similaridade (a escala e
arbitraria). Detectamos o tabuleiro de xadrez em todos os quadros, triangulamos os
seus cantos em 3D com as poses do SfM (DLT em multiplas vistas) e comparamos o
tamanho reconstruido com o real: s = lado_real / mediana(lados_reconstruidos).

Alinhamento: estima o "para cima" pelas cameras (o eixo y da imagem aponta para
baixo e a camera fica de pe), encontra o plano do chao com RANSAC e gira a sala
para z = vertical, chao em z = 0 e paredes alinhadas com os eixos x e y
(hipotese de Manhattan). Assim largura/comprimento/altura viram extensoes nos eixos.
"""
import os

import cv2
import numpy as np

from .step_4_sfm import apply_sim3, image_pose, registered_images

def detect_checkerboard(gray, inner):
    s = 1600 / max(gray.shape)
    small = cv2.resize(gray, None, fx=s, fy=s) if s < 1 else gray
    # flags nomeado: o 3o argumento posicional do binding Python e a saida `corners`
    ok, c = cv2.findChessboardCornersSB(small, tuple(inner),
                                        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    c = c.reshape(-1, 2) / min(s, 1.0)
    # a ordem pode vir invertida em 180 graus; fixa o 1o canto como o mais "superior-esquerdo"
    if c[0].sum() > c[-1].sum():
        c = c[::-1].copy()
    return c

def collect_checkerboard_observations(rec, image_dir, inner, log=print):
    boards = []
    for img in registered_images(rec):
        corners = detect_checkerboard(cv2.imread(os.path.join(image_dir, img.name), cv2.IMREAD_GRAYSCALE), inner)
        if corners is not None:
            boards.append((img, corners))
    log(f"  tabuleiro encontrado em {len(boards)} imagens")
    obs = {}
    if boards:
        # o tabuleiro muda de lugar durante a gravacao: cada posicao vira um grupo de cantos
        groups = group_checkerboards(boards, rec)
        log(f"  {len(groups)} posicao(oes) do tabuleiro: " + ", ".join(str(len(g)) + " vistas" for g in groups))
        for b, group in enumerate(groups):
            for img, corners in group:
                for k, c in enumerate(corners):
                    obs.setdefault((b, k), []).append((img, c))
    return obs

def _project(rec, img, X):
    R, t = image_pose(img)
    Xc = R @ X + t
    if Xc[2] <= 0:
        return None
    return np.asarray(rec.camera(img.camera_id).img_from_cam(Xc.reshape(1, 3), False)).reshape(2)

def group_checkerboards(boards, rec, rel_tol=0.15, min_views=2):
    centers = [c.mean(0) for _, c in boards]
    diags = [np.linalg.norm(c.max(0) - c.min(0)) for _, c in boards]
    remaining = list(range(len(boards)))
    groups = []
    while len(remaining) >= min_views:
        best = []
        for a_i, a in enumerate(remaining):
            for b in remaining[a_i + 1:]:
                r = triangulate_multiview([(boards[a][0], centers[a]), (boards[b][0], centers[b])], rec,
                                          max_error_px=0.05 * min(diags[a], diags[b]), min_views=2)
                if r is None:
                    continue
                inl = []
                for k in remaining:
                    uv = _project(rec, boards[k][0], r[0])
                    if uv is not None and np.linalg.norm(uv - centers[k]) < rel_tol * diags[k]:
                        inl.append(k)
                if len(inl) > len(best):
                    best = inl
        if len(best) < min_views:
            break
        groups.append([boards[k] for k in best])
        remaining = [k for k in remaining if k not in best]
    return groups

def triangulate_multiview(observations, rec, max_error_px=3.0, min_views=3):
    data = []
    for img, uv in observations:
        cam = rec.camera(img.camera_id)
        xn = np.asarray(cam.cam_from_img(np.asarray(uv, float).reshape(1, 2))).reshape(2)
        R, t = image_pose(img)
        data.append((xn, np.hstack([R, t[:, None]]), cam, uv))
    for _ in range(3):
        if len(data) < min_views:
            return None
        A = []
        for xn, P, _, _ in data:
            A.append(xn[0] * P[2] - P[0])
            A.append(xn[1] * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.array(A))
        X = Vt[-1, :3] / Vt[-1, 3]
        errs = []
        for xn, P, cam, uv in data:
            Xc = P[:, :3] @ X + P[:, 3]
            if Xc[2] <= 0:
                errs.append(np.inf)
                continue
            proj = np.asarray(cam.img_from_cam(Xc.reshape(1, 3), False)).reshape(2)
            errs.append(np.linalg.norm(proj - uv))
        errs = np.array(errs)
        if np.all(errs < max_error_px):
            return X, float(errs.mean()), len(data)
        keep = errs < max(max_error_px, np.median(errs[np.isfinite(errs)]) if np.isfinite(errs).any() else 0)
        if keep.all():
            keep = errs < errs.max()
        data = [d for d, k in zip(data, keep) if k]
    return None

def estimate_scale(rec, image_dir, cfg, log=print):
    cols, rows = inner = tuple(cfg["checkerboard_inner"])
    sq = cfg["square_size_m"]
    pts = {}
    for key, o in collect_checkerboard_observations(rec, image_dir, inner, log).items():
        r = triangulate_multiview(o, rec, 3.0, cfg["min_views"])
        if r is not None:
            pts[key] = r
    ratios, edges = [], []
    for b in sorted({k[0] for k in pts}):
        for k in range(cols * rows):
            r, c = divmod(k, cols)
            for nb in ([k + 1] if c + 1 < cols else []) + ([k + cols] if r + 1 < rows else []):
                if (b, k) in pts and (b, nb) in pts:
                    d = np.linalg.norm(pts[(b, k)][0] - pts[(b, nb)][0])
                    if not d > 1e-9:  # cantos triangulados no mesmo ponto (geometria degenerada)
                        continue
                    ratios.append(sq / d)
                    edges.append({"board": b, "corner": k, "neighbor": nb, "model_len": float(d), "real_m": sq})
    info = {"n_corners_3d": len(pts), "n_edges": len(ratios),
            "corner_reproj_px": float(np.mean([v[1] for v in pts.values()])) if pts else None,
            "corners_3d": {str(k): v[0].tolist() for k, v in pts.items()}}
    if not ratios:
        log("  AVISO: referencia de escala nao triangulada; o modelo fica sem escala metrica")
        return None, info
    ratios = np.array(ratios)
    med = float(np.median(ratios))
    good = np.abs(ratios / med - 1) < 0.1  # descarta arestas incoerentes (>10% da mediana)
    if good.sum() < 0.5 * len(ratios):
        # tabuleiros discordam (p.ex. deriva sem fechamento de laco): a mediana cai entre os grupos;
        # usa o maior grupo de arestas coerentes entre si
        support = (np.abs(ratios[:, None] / ratios[None, :] - 1) < 0.1).sum(1)
        good = np.abs(ratios / ratios[np.argmax(support)] - 1) < 0.1
        log(f"  AVISO: arestas em grupos de escala diferentes; usando o maior grupo ({good.sum()} de {len(ratios)})")
        info["scale_consensus_fallback"] = True
    scale = float(np.median(ratios[good]))
    info.update(scale=scale, scale_rel_std=float(ratios[good].std() / scale), n_edges_used=int(good.sum()),
                edges=edges)
    log(f"  escala = {scale:.5f} m/unidade ({good.sum()} arestas, desvio relativo {100 * info['scale_rel_std']:.2f}%)")
    return scale, info

def ransac_plane(points, thresh, iters=2000, normal_ok=None, rng=None, min_inliers=50):
    rng = rng or np.random.default_rng(0)
    n_pts = len(points)
    if n_pts < max(3, min_inliers):
        return None
    best, best_count = None, min_inliers - 1
    batch = 256
    for _ in range(iters // batch + 1):
        idx = rng.integers(0, n_pts, (batch, 3))
        p0, p1, p2 = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm, axis=1)
        valid = norm > 1e-12
        nrm[valid] /= norm[valid, None]
        if normal_ok is not None:
            valid &= normal_ok(nrm)
        if not valid.any():
            continue
        nrm, p0 = nrm[valid], p0[valid]
        d = -np.einsum("ij,ij->i", nrm, p0)
      
        sub = points if n_pts <= 20000 else points[rng.choice(n_pts, 20000, replace=False)]
        counts = (np.abs(sub @ nrm.T + d) < thresh).sum(0)
        k = int(np.argmax(counts))
        c = counts[k] * (n_pts / len(sub))
        if c > best_count:
            best_count, best = c, (nrm[k], d[k])
    if best is None:
        return None
    n, d = best
    mask = np.abs(points @ n + d) < thresh
    for _ in range(2):
        if mask.sum() < 3:
            return None
        c = points[mask].mean(0)
        _, _, Vt = np.linalg.svd(points[mask] - c, full_matrices=False)
        n_new = Vt[-1] * np.sign(Vt[-1] @ n)
        if normal_ok is not None and not normal_ok(n_new[None])[0]:
            break
        n, d = n_new, -n_new @ c
        mask = np.abs(points @ n + d) < thresh
    return n, d, mask

def detect_planes(points, thresh, max_planes=10, min_inliers=100, normal_ok=None, seed=0):
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points))
    planes = []
    for _ in range(max_planes):
        r = ransac_plane(points[remaining], thresh, normal_ok=normal_ok, rng=rng, min_inliers=min_inliers)
        if r is None:
            break
        n, d, mask = r
        planes.append({"n": n, "d": float(d), "idx": remaining[mask], "n_inliers": int(mask.sum())})
        remaining = remaining[~mask]
    return planes

def statistical_outlier_mask(points, k=16, std_ratio=2.0):
    from scipy.spatial import cKDTree
    if len(points) <= k:
        return np.ones(len(points), bool)
    dist, _ = cKDTree(points).query(points, k + 1)
    mean_d = dist[:, 1:].mean(1)
    return mean_d < mean_d.mean() + std_ratio * mean_d.std()

def camera_up(Rs):
    up = -Rs[:, 1, :].mean(0)
    return up / np.linalg.norm(up)

def _rot_from_z(z):
    z = z / np.linalg.norm(z)
    x = np.cross([0, 1.0, 0], z) if abs(z[1]) < 0.9 else np.cross([1.0, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])

def align_room(rec, metric=True, plane_thresh=0.03, log=print):
    from .step_4_sfm import scene_arrays
    sc = scene_arrays(rec)
    unit = 1.0 if metric else np.percentile(np.linalg.norm(sc["xyz"] - np.median(sc["xyz"], 0), axis=1), 90) / 2.5
    thr = plane_thresh * unit
    keep = (sc["track_len"] >= 3) & (sc["error"] < 2.0)
    pts = sc["xyz"][keep]
    pts = pts[statistical_outlier_mask(pts)]
    up = camera_up(sc["R"])
    cam_h = sc["centers"] @ up

    # 1) chao: plano quase perpendicular ao "para cima", abaixo das cameras
    horiz = detect_planes(pts, thr, max_planes=6, min_inliers=max(50, len(pts) // 200),
                          normal_ok=lambda n: np.abs(n @ up) > np.cos(np.deg2rad(20)))
    for p in horiz:
        if p["n"] @ up < 0:
            p["n"], p["d"] = -p["n"], -p["d"]
        p["height"] = -p["d"]
    below = [p for p in horiz if p["height"] < np.median(cam_h) - 0.3 * unit]
    above = [p for p in horiz if p["height"] > np.median(cam_h) + 0.3 * unit]
    floor = min(below, key=lambda p: p["height"]) if below else None
    if floor is not None:
        z_axis = floor["n"]
        floor_h = floor["height"]
        log(f"  chao: plano RANSAC com {floor['n_inliers']} pontos")
    else:
        z_axis = up
        floor_h = np.percentile(pts @ up, 1)
        log("  AVISO: plano do chao nao encontrado; usando o 1o percentil da altura")
    R1 = _rot_from_z(z_axis)
    p1 = pts @ R1.T
    p1[:, 2] -= floor_h
    ceiling = max(above, key=lambda p: p["height"]) if above else None
    height = (ceiling["height"] - floor_h) if ceiling is not None else None

    # 2) paredes: planos verticais na faixa de altura entre chao e teto
    top = height if height else np.percentile(p1[:, 2], 98)
    band = (p1[:, 2] > 0.15 * top) & (p1[:, 2] < 0.85 * top)
    walls = detect_planes(p1[band], thr, max_planes=12, min_inliers=max(50, band.sum() // 100),
                          normal_ok=lambda n: np.abs(n[:, 2]) < np.sin(np.deg2rad(10)), seed=1)
    if walls:
        # direcao de Manhattan: media circular de 4*theta ponderada pelos inliers
        th = np.array([np.arctan2(w["n"][1], w["n"][0]) for w in walls])
        wt = np.array([w["n_inliers"] for w in walls], float)
        yaw = np.arctan2((wt * np.sin(4 * th)).sum(), (wt * np.cos(4 * th)).sum()) / 4
    else:
        yaw = 0.0
        log("  AVISO: nenhuma parede detectada; orientacao horizontal arbitraria")
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rz @ R1
    t = np.array([0, 0, -floor_h])
    # centraliza a sala (na faixa das paredes) na origem
    p2 = pts @ R.T + t
    b2 = p2[(p2[:, 2] > 0.15 * top) & (p2[:, 2] < 0.85 * top)]
    if len(b2) == 0:
        b2 = p2
    center = (np.percentile(b2[:, :2], 2, 0) + np.percentile(b2[:, :2], 98, 0)) / 2
    t = t + np.array([-center[0], -center[1], 0])
    apply_sim3(rec, 1.0, R, t)
    log(f"  sala alinhada (yaw das paredes = {np.degrees(yaw):.1f} graus)")
    return {"R": R.tolist(), "t": t.tolist(), "floor_found": floor is not None,
            "ceiling_found": ceiling is not None, "plane_height": height, "unit": float(unit)}
