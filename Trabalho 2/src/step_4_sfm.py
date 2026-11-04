"""
Structure from Motion (SfM) incremental.

Parte do melhor par de duas vistas da Etapa 3 e, enquanto houver imagens:
  1. registra a imagem com mais correspondencias 2D-3D (PnP com RANSAC);
  2. triangula os pontos novos entre ela e as imagens ja registradas;
  3. refina cameras e pontos juntos com bundle adjustment e remove outliers.

Convencao: a pose (R, t) de cada camera leva coordenadas do mundo para a camera
(x_cam = R X + t); o mundo e o referencial da primeira camera do par inicial.
A reconstrucao sai em escala arbitraria (|t| = 1 no par inicial) ate ser
ajustada com a referencia metrica (`checkerboard_scale`).
"""
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import cv2
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from .step_1_keypoints_detector import to_gray8, to_uint8
from .step_3_epipolar import reprojection_errors, save_point_cloud, triangulate


@dataclass
class Reconstruction:
    """
    Estado do SfM. `points[pid]` e o ponto 3D e `tracks[pid]` o dict {imagem: keypoint}
    com as suas observacoes (o "track"); `point_of[(imagem, keypoint)]` faz o caminho
    inverso. `fixed` e a camera que fica parada no bundle adjustment. `pids_of(imagem)` e o
    mesmo mapa inverso como array (pid de cada keypoint, -1 se nao triangulado), para as
    buscas vetorizadas.
    """
    K: np.ndarray
    n_kps: list  # numero de keypoints de cada imagem
    poses: dict = field(default_factory=dict)  # imagem -> (R, t)
    points: dict = field(default_factory=dict)  # pid -> X (3,)
    tracks: dict = field(default_factory=dict)  # pid -> {imagem: keypoint}
    point_of: dict = field(default_factory=dict)  # (imagem, keypoint) -> pid
    kp_pid: dict = field(default_factory=dict)  # imagem -> array (n_kps,)
    fixed: int = None
    next_id: int = 0

    def pids_of(self, img):
        if img not in self.kp_pid:
            self.kp_pid[img] = np.full(self.n_kps[img], -1, dtype=np.int64)
        return self.kp_pid[img]

    def add_point(self, X, obs):
        pid = self.next_id
        self.next_id += 1
        self.points[pid] = X
        self.tracks[pid] = {}
        for img, kp in obs.items():
            self.add_observation(pid, img, kp)
        return pid

    def add_observation(self, pid, img, kp):
        """Liga o keypoint ao ponto, se nenhum dos dois ja tiver essa ligacao com a imagem."""
        if img in self.tracks[pid] or (img, kp) in self.point_of:
            return False
        self.tracks[pid][img] = kp
        self.point_of[(img, kp)] = pid
        self.pids_of(img)[kp] = pid
        return True

    def remove_observation(self, pid, img):
        kp = self.tracks[pid].pop(img)
        del self.point_of[(img, kp)]
        self.kp_pid[img][kp] = -1

    def remove_point(self, pid):
        for img, kp in self.tracks.pop(pid).items():
            del self.point_of[(img, kp)]
            self.kp_pid[img][kp] = -1
        del self.points[pid]


def camera_center(R, t):
    """Posicao da camera no mundo: C = -R^T t (o ponto que vai para a origem da camera)."""
    return -R.T @ np.reshape(t, 3)


_coords_cache = {}


def keypoint_coords(kps):
    """Coordenadas N x 2 dos keypoints (guardadas por lista, para nao refazer a cada busca)."""
    if hasattr(kps, "xy"):  # Keypoints (Etapa 2): as coordenadas ja estao num array
        return kps.xy
    key = id(kps)
    if key not in _coords_cache or _coords_cache[key][0] is not kps:
        _coords_cache[key] = (kps, np.float64([k.pt for k in kps]).reshape(-1, 2))
    return _coords_cache[key][1]


def triangulation_angles(points3d, C1, C2):
    """Angulo (graus) entre os dois raios que saem dos centros C1 e C2 e chegam em cada ponto."""
    r1, r2 = points3d - C1, points3d - C2
    cos = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def verified_matches(geometries):
    """
    Matches que sobreviveram ao RANSAC de E, como arrays M x 2 de indices de keypoints.
    Guarda as duas direcoes: matches[(i, j)] e matches[(j, i)]. Nao usa a mascara do
    teste de quiralidade: ela depende da pose (R, t) do par, que e instavel quando a
    camera mais gira do que anda entre as fotos, e descartaria matches corretos. As
    poses do SfM vem do PnP, e matches errados sao barrados na triangulacao.
    """
    matches = {}
    for (i, j), g in geometries.items():
        if g["mask_E"] is None:
            continue
        good = g["good"]  # array N x 3 do cache ou lista de cv2.DMatch
        pairs = good[:, :2] if isinstance(good, np.ndarray) else [(m.queryIdx, m.trainIdx) for m in good]
        idx = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)[g["mask_E"]]
        matches[(i, j)] = idx
        matches[(j, i)] = idx[:, ::-1]
    return matches


def choose_initial_pair(geometries, min_angle=3.0):
    """
    Par com mais pontos triangulados entre os que tem mediana do angulo de
    triangulacao >= `min_angle` graus. Angulo pequeno significa baseline curta em
    relacao a profundidade da cena: a profundidade fica mal determinada e o erro se
    propaga para toda a reconstrucao. Sem nenhum par acima do limiar, usa o de mais pontos.
    """
    scored = []
    for pair, g in geometries.items():
        if g["points3d"] is None or len(g["points3d"]) == 0:
            continue
        angle = np.median(triangulation_angles(g["points3d"], np.zeros(3), camera_center(g["R"], g["t"])))
        scored.append((angle >= min_angle, len(g["points3d"]), pair))
    if not scored:
        raise ValueError("Nenhum par de imagens tem geometria suficiente para iniciar o SfM")
    return max(scored)[2]


def triangulate_views(pts1, pts2, K, pose1, pose2, max_reproj_error=4.0, min_angle=1.0):
    """
    Triangula entre duas cameras com poses quaisquer reaproveitando `triangulate` da
    Etapa 3, que supoe a camera 1 na origem: triangula no referencial da camera 1, com a
    pose relativa R = R2 R1^T e t = t2 - R t1, e volta para o mundo com X = R1^T (X1 - t1).
    Alem dos filtros da Etapa 3, descarta pontos com angulo de triangulacao < `min_angle`.
    Retorna (pontos N x 3 no mundo, mascara sobre as entradas).
    """
    (R1, t1), (R2, t2) = pose1, pose2
    R = R2 @ R1.T
    t = t2 - R @ t1
    X_cam1, valid = triangulate(pts1, pts2, K, R, t, max_reproj_error)
    X = (X_cam1 - t1) @ R1
    keep = triangulation_angles(X, camera_center(R1, t1), camera_center(R2, t2)) >= min_angle
    valid[np.flatnonzero(valid)[~keep]] = False
    return X[keep], valid


def triangulate_new_points(recon, img, matches, all_kps, max_reproj_error=4.0, min_angle=1.0):
    """
    Cria pontos 3D a partir dos matches entre `img` e cada imagem ja registrada em que
    nenhum dos dois keypoints tem ponto 3D ainda. Retorna quantos pontos criou.
    """
    created = 0
    for other in list(recon.poses):
        if other == img or (img, other) not in matches:
            continue
        m = matches[(img, other)]
        pairs = m[(recon.pids_of(img)[m[:, 0]] < 0) & (recon.pids_of(other)[m[:, 1]] < 0)]
        if not len(pairs):
            continue
        pts1 = keypoint_coords(all_kps[img])[pairs[:, 0]]
        pts2 = keypoint_coords(all_kps[other])[pairs[:, 1]]
        X, valid = triangulate_views(pts1, pts2, recon.K, recon.poses[img], recon.poses[other],
                                     max_reproj_error, min_angle)
        for Xk, (a, b) in zip(X, pairs[valid].tolist()):
            # um mesmo keypoint pode ter casado com dois do outro lado
            if (other, b) not in recon.point_of and (img, a) not in recon.point_of:
                recon.add_point(Xk, {img: a, other: b})
                created += 1
    return created


def initialize(geometries, matches, all_kps, K, pair, **triangulation_kw):
    """Reconstrucao inicial: camera i na origem, camera j com o (R, t) da Etapa 3, e os pontos do par."""
    i, j = pair
    g = geometries[pair]
    recon = Reconstruction(K, [len(k) for k in all_kps], fixed=i)
    recon.poses[i] = (np.eye(3), np.zeros(3))
    recon.poses[j] = (g["R"], g["t"].reshape(3))
    triangulate_new_points(recon, j, matches, all_kps, **triangulation_kw)
    return recon


def correspondences_2d3d(recon, img, matches):
    """
    Lista de (keypoint de `img`, pid): keypoints de `img` que casam com um keypoint ja
    triangulado em alguma imagem registrada, e por isso "enxergam" um ponto 3D conhecido.
    """
    kps, pids = [], []
    for other in recon.poses:
        m = matches.get((img, other))
        if m is None:
            continue
        p = recon.pids_of(other)[m[:, 1]]
        kps.append(m[p >= 0, 0])
        pids.append(p[p >= 0])
    if not kps:
        return []
    kps, pids = np.concatenate(kps), np.concatenate(pids)
    first = np.sort(np.unique(kps, return_index=True)[1])  # a 1a imagem que enxerga o keypoint vence
    return list(zip(kps[first].tolist(), pids[first].tolist()))


def register_image(recon, img, corr, all_kps, ransac_thresh=4.0, min_inliers=30):
    """
    Estima a pose de `img` a partir das correspondencias 2D-3D (Perspective-n-Point):
    EPnP dentro do RANSAC para descartar correspondencias erradas e refinamento de
    Levenberg-Marquardt nos inliers. Os inliers viram observacoes dos pontos (os tracks
    crescem). Retorna o numero de inliers, ou 0 se a imagem nao puder ser registrada.
    """
    if len(corr) < max(min_inliers, 6):
        return 0
    kps = [a for a, _ in corr]
    pids = [p for _, p in corr]
    pts2d = np.float64([all_kps[img][a].pt for a in kps])
    pts3d = np.float64([recon.points[p] for p in pids])
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, recon.K, None, iterationsCount=1000,
                                                 reprojectionError=ransac_thresh, confidence=0.999,
                                                 flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < min_inliers:
        return 0
    inliers = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(pts3d[inliers], pts2d[inliers], recon.K, None, rvec, tvec)
    recon.poses[img] = (cv2.Rodrigues(rvec)[0], tvec.ravel())
    for k in inliers:
        recon.add_observation(pids[k], img, kps[k])
    return len(inliers)


def rotate(rvec, X):
    """
    Gira X pelo vetor de Rodrigues rvec (torch, diferenciavel):
    R X = X + A (r x X) + B (r (r . X) - theta^2 X), A = sin(theta) / theta,
    B = (1 - cos(theta)) / theta^2, com Taylor perto de theta = 0 (onde a forma fechada da 0/0).
    """
    th2 = (rvec * rvec).sum(-1, keepdim=True)
    small = th2 < 1e-8
    th2_safe = torch.where(small, torch.ones_like(th2), th2)
    th = th2_safe.sqrt()
    A = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    B = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2_safe)
    dot = (rvec * X).sum(-1, keepdim=True)
    return X + A * torch.linalg.cross(rvec, X, dim=-1) + B * (rvec * dot - th2 * X)


def _project(cam, X, f, c):
    """Pixel de X na camera cam = (rvec, t), focal f e ponto principal c."""
    X_cam = rotate(cam[:3], X) + cam[3:]
    return f * X_cam[:2] / X_cam[2] + c


_project_all = torch.func.vmap(_project, in_dims=(0, 0, None, None))
# jacobiano 2 x 6 (camera), 2 x 3 (ponto) e 2 x 1 (focal) de cada observacao
_jacobians = torch.func.vmap(torch.func.jacfwd(_project, argnums=(0, 1, 2)), in_dims=(0, 0, None, None))


def huber(r, f_scale):
    """Custo de Huber por componente do residuo e o peso do IRLS (1 dentro de f_scale, f_scale/|r| fora)."""
    a = r.abs()
    inside = a <= f_scale
    return torch.where(inside, r * r, 2 * f_scale * a - f_scale ** 2).sum(), torch.where(inside, 1.0, f_scale / a)


def bundle_adjustment(recon, all_kps, f_scale=2.0, max_iters=50, refine_focal=False, device="cuda",
                      pair_chunk=300_000, obs_chunk=200_000):
    """
    Minimiza o erro de reprojecao de todas as observacoes ajustando juntos as poses
    (rvec e t: 6 parametros por camera), os pontos (3 por ponto) e, com `refine_focal`, a
    distancia focal (compartilhada: o video vem de uma camera so, sem zoom). A camera
    `recon.fixed` nao se move: sem isso a cena inteira poderia girar e transladar sem mudar
    o erro (ambiguidade de gauge). A perda de Huber reduz o peso de residuos acima de
    `f_scale` px (outliers).

    Levenberg-Marquardt na GPU, no esquema classico de BA: os jacobianos por observacao saem
    do autodiff do torch (vmap + jacfwd); os blocos 3 x 3 dos pontos sao invertidos em lote
    e eliminados pelo complemento de Schur, sobrando um sistema denso so das cameras
    (6 por camera + focal), resolvido por Cholesky. Cada par de observacoes de um mesmo
    ponto contribui com um bloco do complemento de Schur. Retorna o RMS (px) antes e depois.
    """
    if not torch.cuda.is_available():
        device = "cpu"
    dt = torch.float64
    cams = [c for c in recon.poses if c != recon.fixed] + [recon.fixed]  # fixa por ultimo
    cam_index = {c: k for k, c in enumerate(cams)}
    n_cams = len(cams)
    pids = list(recon.points)
    obs_cam, obs_pt, obs_xy = [], [], []
    for n, pid in enumerate(pids):  # observacoes agrupadas por ponto (contiguas)
        for img, kp in recon.tracks[pid].items():
            obs_cam.append(cam_index[img])
            obs_pt.append(n)
            obs_xy.append(all_kps[img][kp].pt)
    obs_cam = torch.tensor(obs_cam, device=device)
    obs_pt = torch.tensor(obs_pt, device=device)
    obs_xy = torch.tensor(np.float64(obs_xy), dtype=dt, device=device)
    n_obs, n_pts = len(obs_pt), len(pids)

    cam_x = torch.tensor(np.float64([np.hstack([cv2.Rodrigues(recon.poses[c][0])[0].ravel(), recon.poses[c][1]])
                                     for c in cams]), dtype=dt, device=device)
    pts_x = torch.tensor(np.float64([recon.points[p] for p in pids]), dtype=dt, device=device)
    f_x = torch.tensor([recon.K[0, 0]], dtype=dt, device=device)
    c = torch.tensor([recon.K[0, 2], recon.K[1, 2]], dtype=dt, device=device)

    # incognitas das cameras: 6 por camera + a focal no fim; a camera fixa e a focal (sem
    # `refine_focal`) viram identidade no sistema, o que zera a atualizacao delas
    n_c = 6 * n_cams + 1
    idx7 = torch.cat([6 * obs_cam[:, None] + torch.arange(6, device=device), torch.full((n_obs, 1), n_c - 1, device=device)], 1)
    frozen = list(range(6 * (n_cams - 1), 6 * n_cams)) + ([] if refine_focal else [n_c - 1])
    frozen = torch.tensor(frozen, device=device)

    # pares (o1, o2) de observacoes do mesmo ponto, incluindo o1 = o2
    counts = torch.bincount(obs_pt, minlength=n_pts)
    start = torch.cumsum(counts, 0) - counts
    lens = counts[obs_pt]
    o1 = torch.repeat_interleave(torch.arange(n_obs, device=device), lens)
    o2 = start[obs_pt[o1]] + torch.arange(len(o1), device=device) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)

    # o sistema reduzido e esparso: so ha bloco 7 x 7 entre duas cameras que veem um mesmo ponto.
    # Os blocos sao acumulados na GPU e o sistema e resolvido como matriz esparsa na CPU: a versao
    # densa ((6 cameras)^2) nao cabe na GPU com milhares de cameras
    block_key, pair_block = torch.unique(obs_cam[o1] * n_cams + obs_cam[o2], return_inverse=True)
    n_blocks = len(block_key)
    obs_block = torch.searchsorted(block_key, obs_cam * n_cams + obs_cam)  # bloco (c, c) de cada observacao
    idx7_cam = torch.cat([6 * torch.arange(n_cams, device=device)[:, None] + torch.arange(6, device=device),
                          torch.full((n_cams, 1), n_c - 1, device=device)], 1)
    ca, cb = block_key // n_cams, block_key % n_cams
    sp_rows = idx7_cam[ca][:, :, None].expand(-1, 7, 7).reshape(-1).cpu().numpy()
    sp_cols = idx7_cam[cb][:, None, :].expand(-1, 7, 7).reshape(-1).cpu().numpy()
    keep = np.setdiff1d(np.arange(n_c), frozen.cpu().numpy())

    def sparse(blocks):
        return coo_matrix((blocks.reshape(-1).cpu().numpy().astype(np.float64), (sp_rows, sp_cols)),
                          shape=(n_c, n_c)).tocsc()

    # observacoes em blocos: os jacobianos (vmap + jacfwd) e os produtos por observacao de todas
    # de uma vez nao cabem nos 6 GB da GPU com milhoes de observacoes
    chunks = [slice(s0, min(s0 + obs_chunk, n_obs)) for s0 in range(0, n_obs, obs_chunk)]

    def residuals(cam_x, pts_x, f_x):
        return torch.cat([_project_all(cam_x[obs_cam[b]], pts_x[obs_pt[b]], f_x, c) - obs_xy[b] for b in chunks])

    r = residuals(cam_x, pts_x, f_x)
    rms_before = float(r.pow(2).mean().sqrt())
    cost, weights = huber(r, f_scale)
    lam = 1e-3
    eye3 = torch.eye(3, dtype=dt, device=device)
    W = torch.empty(n_obs, 7, 3, dtype=dt, device=device)  # 7 x 3 por observacao
    for _ in range(max_iters):
        U = torch.zeros(n_blocks, 7, 7, dtype=dt, device=device)
        V = torch.zeros(n_pts, 3, 3, dtype=dt, device=device)
        g_c = torch.zeros(n_c, dtype=dt, device=device)
        g_p = torch.zeros(n_pts, 3, dtype=dt, device=device)
        for b in chunks:
            Jc, Jp, Jf = _jacobians(cam_x[obs_cam[b]], pts_x[obs_pt[b]], f_x, c)
            Jcam = torch.cat([Jc, Jf], 2)  # 2 x 7
            wJcam, wJp = Jcam * weights[b, :, None], Jp * weights[b, :, None]
            U.index_add_(0, obs_block[b], torch.einsum("nki,nkj->nij", wJcam, Jcam))
            V.index_add_(0, obs_pt[b], torch.einsum("nki,nkj->nij", wJp, Jp))
            W[b] = torch.einsum("nki,nkj->nij", wJcam, Jp)
            g_c.index_add_(0, idx7[b].reshape(-1), torch.einsum("nki,nk->ni", wJcam, r[b]).reshape(-1))
            g_p.index_add_(0, obs_pt[b], torch.einsum("nki,nk->ni", wJp, r[b]))
        U = sparse(U)
        U_diag = diags(U.diagonal())

        improved = False
        W32 = W.float()
        while lam < 1e8:
            # blocos dos pontos e complemento de Schur em float32: placas de consumo (como a RTX
            # 3050) fazem float64 a 1/64 da velocidade, e essa montagem domina o tempo do BA. O
            # sistema reduzido e resolvido em float64, e o passo so e aceito se o custo, calculado
            # em float64, cair: um passo aproximado custa no maximo mais uma iteracao
            V_inv = torch.linalg.inv((V + lam * torch.diag_embed(torch.diagonal(V, dim1=1, dim2=2))).float()
                                     + 1e-6 * eye3.float())
            Y = W32 @ V_inv[obs_pt]  # 7 x 3 por observacao
            blocks = torch.zeros(n_blocks, 7, 7, dtype=torch.float32, device=device)
            for s0 in range(0, len(o1), pair_chunk):
                a, b = o1[s0:s0 + pair_chunk], o2[s0:s0 + pair_chunk]
                blocks.index_add_(0, pair_block[s0:s0 + pair_chunk], -(Y[a] @ W32[b].transpose(1, 2)))
            S = U + lam * U_diag + sparse(blocks)
            S = ((S + S.T) / 2)[keep][:, keep]  # sem as incognitas congeladas (camera fixa, focal)
            rhs = -g_c.clone().index_add_(0, idx7.reshape(-1), -(Y.double() @ g_p[obs_pt][..., None]).reshape(-1))
            try:
                sol = splu(S.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(rhs[keep].cpu().numpy())
            except RuntimeError:  # matriz singular: mais amortecimento
                sol = None
            if sol is None or not np.all(np.isfinite(sol)):
                lam *= 10
                continue
            d_c = torch.zeros(n_c, dtype=dt, device=device)
            d_c[torch.from_numpy(keep).to(device)] = torch.from_numpy(sol).to(device)
            back = torch.zeros(n_pts, 3, dtype=dt, device=device).index_add_(
                0, obs_pt, (W.transpose(1, 2) @ d_c[idx7][..., None]).squeeze(-1))
            d_p = -(V_inv.double() @ (g_p + back)[..., None]).squeeze(-1)
            new = (cam_x + d_c[:-1].view(n_cams, 6), pts_x + d_p, f_x + d_c[-1:])
            r_new = residuals(*new)
            cost_new, weights_new = huber(r_new, f_scale)
            if torch.isfinite(cost_new) and cost_new < cost:
                converged = (cost - cost_new) < 1e-6 * cost
                cam_x, pts_x, f_x = new
                r, cost, weights = r_new, cost_new, weights_new
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 4
        if not improved or converged:
            break

    cam_x, pts_x = cam_x.cpu().numpy(), pts_x.cpu().numpy()
    for k, cam in enumerate(cams[:-1]):
        recon.poses[cam] = (cv2.Rodrigues(cam_x[k, :3])[0], cam_x[k, 3:].copy())
    for pid, X in zip(pids, pts_x):
        recon.points[pid] = X
    recon.K = recon.K.copy()
    recon.K[0, 0] = recon.K[1, 1] = float(f_x)
    return rms_before, float(r.pow(2).mean().sqrt())


def observation_errors(recon, all_kps):
    """Para cada camera: lista de (pid, erro de reprojecao em px, profundidade) das suas observacoes."""
    per_camera = {img: [] for img in recon.poses}
    for (img, kp), pid in recon.point_of.items():
        per_camera[img].append((pid, kp))
    errors = {}
    for img, obs in per_camera.items():
        if not obs:
            errors[img] = []
            continue
        R, t = recon.poses[img]
        X = np.float64([recon.points[pid] for pid, _ in obs])
        pts = np.float64([all_kps[img][kp].pt for _, kp in obs])
        err = reprojection_errors(X, pts, recon.K, R, t)
        depth = (X @ R.T + t)[:, 2]
        errors[img] = [(pid, e, d) for (pid, _), e, d in zip(obs, err, depth)]
    return errors


def filter_outliers(recon, all_kps, max_reproj_error=4.0):
    """
    Remove as observacoes com erro de reprojecao acima de `max_reproj_error` px ou com o
    ponto atras da camera; pontos que ficam com menos de 2 observacoes deixam de estar
    triangulados e saem tambem. Retorna o numero de pontos removidos.
    """
    for img, obs in observation_errors(recon, all_kps).items():
        for pid, err, depth in obs:
            if err > max_reproj_error or depth <= 0:
                recon.remove_observation(pid, img)
    weak = [pid for pid, track in recon.tracks.items() if len(track) < 2]
    for pid in weak:
        recon.remove_point(pid)
    return len(weak)


def reconstruction_stats(recon, all_kps):
    """Metricas para o relatorio: cameras, pontos, tamanho medio dos tracks e erro de reprojecao."""
    errors = np.array([e for obs in observation_errors(recon, all_kps).values() for _, e, _ in obs])
    return {"n_cameras": len(recon.poses), "n_points": len(recon.points),
            "mean_track_length": float(np.mean([len(t) for t in recon.tracks.values()])) if recon.tracks else 0.0,
            "reproj_mean": float(errors.mean()) if len(errors) else float("nan"),
            "reproj_median": float(np.median(errors)) if len(errors) else float("nan")}


def project(recon, X, img):
    """Projecao (pixel) do ponto X na camera `img`, e a profundidade dele nessa camera."""
    R, t = recon.poses[img]
    x = recon.K @ (R @ X + t)
    return x[:2] / x[2], x[2]


def fits(recon, X, obs, all_kps, max_error):
    """True se X fica na frente de todas as cameras de `obs` e reprojeta a menos de `max_error` px."""
    for img, kp in obs.items():
        uv, depth = project(recon, X, img)
        if depth <= 0 or np.linalg.norm(uv - all_kps[img][kp].pt) > max_error:
            return False
    return True


def merge_tracks(recon, matches, all_kps, max_error=4.0):
    """
    Junta observacoes que o registro incremental deixou separadas, usando os matches
    verificados entre imagens ja registradas:
      - so um dos keypoints tem ponto 3D: o outro vira mais uma observacao dele (o track
        cresce e o ponto passa a amarrar mais uma camera no bundle adjustment);
      - os dois tem pontos diferentes: sao o mesmo ponto fisico triangulado duas vezes, e
        os tracks sao fundidos, mantendo a posicao de um dos dois.
    So aceita se o ponto reprojetar a menos de `max_error` px nas observacoes novas (como
    no COLMAP), para nao amarrar no BA observacoes incompativeis. Tracks mais longos
    deixam o modelo mais rigido e reduzem a deriva, mas uma deriva que ja passou desse
    limite (p.ex., ao fechar um laco grande) nao e corrigida aqui.
    Retorna (observacoes adicionadas, pontos fundidos).
    """
    extended = merged = 0
    for (i, j), idx in matches.items():
        if i > j or i not in recon.poses or j not in recon.poses:
            continue
        # filtro vetorizado: so os matches em que algum dos dois keypoints tem ponto 3D e os pontos
        # diferem podem estender ou fundir tracks; os outros (a grande maioria) nao precisam do laco
        p_all, q_all = recon.pids_of(i)[idx[:, 0]], recon.pids_of(j)[idx[:, 1]]
        candidates = idx[(p_all != q_all) & ((p_all >= 0) | (q_all >= 0))]
        for a, b in candidates.tolist():
            p, q = recon.point_of.get((i, a)), recon.point_of.get((j, b))
            if p is None and q is None or p == q:
                continue
            if p is None or q is None:
                pid, img, kp = (q, i, a) if p is None else (p, j, b)
                if img not in recon.tracks[pid] and fits(recon, recon.points[pid], {img: kp}, all_kps, max_error):
                    extended += recon.add_observation(pid, img, kp)
                continue
            if set(recon.tracks[p]) & set(recon.tracks[q]):
                continue  # os dois tracks tem observacoes da mesma imagem
            for keep, drop in ((p, q), (q, p)):
                if fits(recon, recon.points[keep], recon.tracks[drop], all_kps, max_error):
                    obs = recon.tracks[drop]
                    recon.remove_point(drop)
                    for img, kp in obs.items():
                        recon.add_observation(keep, img, kp)
                    merged += 1
                    break
    return extended, merged


def refine(recon, all_kps, matches, max_reproj_error=4.0, log=print, final=False, refine_focal=True,
           min_cameras_focal=10, device="cuda"):
    """
    Roda o bundle adjustment e remove outliers. Na rodada final, antes de cada BA funde
    os tracks (`merge_tracks`, que percorre todos os matches e e lento demais para cada
    rodada) e repete enquanto houver o que fundir (ate 3 vezes): cada BA move os pontos
    e pode deixar novas observacoes dentro do limite de reprojecao. A focal so entra no
    BA com pelo menos `min_cameras_focal` cameras: com poucas vistas ela se confunde com a
    profundidade da cena.
    """
    focal = refine_focal and len(recon.poses) >= min_cameras_focal
    for _ in range(3 if final else 1):
        extended, merged = merge_tracks(recon, matches, all_kps, max_reproj_error) if final else (0, 0)
        rms = bundle_adjustment(recon, all_kps, max_iters=100 if final else 30, refine_focal=focal, device=device)
        removed = filter_outliers(recon, all_kps, max_reproj_error)
        log(f"  {'BA final' if final else 'BA'}: {extended} observacoes e {merged} pontos fundidos, "
            f"RMS {rms[0]:.2f} -> {rms[1]:.2f} px, f = {recon.K[0, 0]:.0f} px, {removed} pontos removidos")
        if not (extended or merged):
            break


def run_sfm(all_kps, geometries, K, init_pair=None, min_inliers_pnp=30, pnp_thresh=4.0,
            max_reproj_error=4.0, min_angle=1.0, ba_growth=0.1, refine_focal=True, device="cuda", verbose=True,
            resume=None, checkpoint=None):
    """
    SfM incremental completo a partir das saidas da Etapa 3 (`geometries` de
    `build_connectivity_matrix`). A cada passo registra a imagem que mais enxerga
    pontos ja reconstruidos e triangula pontos novos. O bundle adjustment (seguido da
    remocao de outliers) roda quando o numero de cameras cresce `ba_growth` (10%) desde o
    ultimo, como no COLMAP: no inicio, quando o modelo e pequeno e mais fragil, roda a
    cada imagem; depois fica espacado, porque cada BA custa mais. Uma imagem que falha
    volta a ser tentada depois que outra for registrada (ai ha mais pontos 3D para casar).
    No fim, `merge_tracks` junta os tracks que ficaram separados.
    """
    log = print if verbose else (lambda *a, **k: None)
    ref_kw = dict(refine_focal=refine_focal, device=device)
    tri_kw = dict(max_reproj_error=max_reproj_error, min_angle=min_angle)
    matches = verified_matches(geometries)
    # `resume`: reconstrucao salva ao fim da fase incremental (`checkpoint`), para refazer so o
    # refinamento final, que e a parte mais pesada em memoria
    if resume is not None:
        recon, failed = resume, set()
    else:
        pair = init_pair if init_pair is not None else choose_initial_pair(geometries)
        recon = initialize(geometries, matches, all_kps, K, pair, **tri_kw)
        log(f"Par inicial {pair}: {len(recon.points)} pontos")
        refine(recon, all_kps, matches, max_reproj_error, log, **ref_kw)

        # so as imagens da fronteira (pendentes com match verificado com alguma ja registrada) podem
        # ter correspondencias 2D-3D: avaliar todas as pendentes a cada passo nao escala para milhares
        partners = {}
        for i, j in matches:
            partners.setdefault(i, set()).add(j)
        frontier = {o for img in recon.poses for o in partners.get(img, ())} - set(recon.poses)
        failed = set()
        cameras_at_ba = len(recon.poses)
        while True:
            pending = frontier - failed
            if not pending:
                break
            candidates = [(img, correspondences_2d3d(recon, img, matches)) for img in sorted(pending)]
            img, corr = max(candidates, key=lambda c: len(c[1]))
            n_inliers = register_image(recon, img, corr, all_kps, pnp_thresh, min_inliers_pnp)
            if not n_inliers:
                log(f"Imagem {img}: nao registrada ({len(corr)} correspondencias 2D-3D)")
                failed.add(img)
                continue
            failed.clear()
            frontier.discard(img)
            frontier |= partners.get(img, set()) - set(recon.poses)
            created = triangulate_new_points(recon, img, matches, all_kps, **tri_kw)
            log(f"Imagem {img}: {n_inliers}/{len(corr)} inliers no PnP, {created} pontos novos")
            if len(recon.poses) >= cameras_at_ba * (1 + ba_growth):
                refine(recon, all_kps, matches, max_reproj_error, log, **ref_kw)
                cameras_at_ba = len(recon.poses)
        if checkpoint is not None:
            checkpoint(recon)

    refine(recon, all_kps, matches, max_reproj_error, log, final=True, **ref_kw)
    if failed:
        log(f"Imagens nao registradas: {sorted(failed)}")
    return recon


def scale_reconstruction(recon, s):
    """
    Multiplica pontos e translacoes por `s`. As rotacoes nao mudam e as projecoes ficam
    identicas (x_cam = R (sX) + s t = s (R X + t), e a divisao pela profundidade cancela s).
    """
    for pid in recon.points:
        recon.points[pid] = recon.points[pid] * s
    for img, (R, t) in recon.poses.items():
        recon.poses[img] = (R, t * s)


def transform_reconstruction(recon, T):
    """
    Aplica a transformacao rigida T (4 x 4, X' = Rw X + tw) a pontos e cameras, p.ex. a de
    `align_to_floor`. Como x_cam = R X + t = R Rw^T (X' - tw) + t, a nova pose e
    R' = R Rw^T e t' = t - R' tw; as projecoes (e o erro de reprojecao) nao mudam.
    """
    Rw, tw = T[:3, :3], T[:3, 3]
    for pid in recon.points:
        recon.points[pid] = Rw @ recon.points[pid] + tw
    for img, (R, t) in recon.poses.items():
        R_new = R @ Rw.T
        recon.poses[img] = (R_new, t - R_new @ tw)


def detect_checkerboard(image, pattern=(8, 6), max_side=2000):
    """
    Cantos internos do tabuleiro (`pattern` = cantos por linha e por coluna; o tabuleiro
    A4 de 9 x 7 quadrados tem 8 x 6). A deteccao roda numa copia reduzida para
    `max_side` px, que e muito mais rapida nas fotos de alta resolucao, e os cantos sao
    refinados com precisao subpixel na imagem original. Retorna N x 2 ou None.
    """
    gray = to_gray8(image)
    f = min(1.0, max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    ok, corners = cv2.findChessboardCornersSB(small, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        return None
    corners = corners.reshape(-1, 2) / f
    if f < 1:
        # janela menor que meio quadrado, para nao "pular" para o canto vizinho
        grid = corners.reshape(pattern[1], pattern[0], 2)
        spacing = np.linalg.norm(np.diff(grid, axis=1), axis=2).min()
        half = int(np.clip(0.4 * spacing, 2, 11))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(gray, np.float32(corners).reshape(-1, 1, 2), (half, half), (-1, -1),
                                   criteria).reshape(-1, 2)
    return np.float64(corners)


def detect_checkerboards(images, indices, pattern=(8, 6)):
    """`detect_checkerboard` nas imagens `indices`, em paralelo. Retorna {imagem: cantos N x 2}."""
    with ThreadPoolExecutor(min(4, os.cpu_count())) as executor:  # limita a memoria (quadros 1080p)
        found = executor.map(lambda i: detect_checkerboard(images[i], pattern), indices)
        return {i: c for i, c in zip(indices, found) if c is not None}


def triangulate_multiview(observations, K):
    """
    Triangulacao linear (DLT) de N pontos vistos em varias cameras. `observations` e uma
    lista de ((R, t), pts N x 2): cada vista da 2 equacoes por ponto, x P3 - P1 = 0 e
    y P3 - P2 = 0, e a solucao e o vetor singular de menor valor singular (SVD em lote).
    """
    P = np.stack([K @ np.hstack([R, np.reshape(t, (3, 1))]) for (R, t), _ in observations])  # V x 3 x 4
    pts = np.stack([p for _, p in observations])  # V x N x 2
    A = np.concatenate([pts[..., :1] * P[:, None, 2] - P[:, None, 0],
                        pts[..., 1:] * P[:, None, 2] - P[:, None, 1]])  # 2V x N x 4
    Xh = np.linalg.svd(A.transpose(1, 0, 2))[2][:, -1]
    return Xh[:, :3] / Xh[:, 3:]


def board_reprojection(recon, views):
    """Triangula os cantos com as vistas {imagem: cantos} e devolve (X, mediana do erro por vista)."""
    X = triangulate_multiview([(recon.poses[i], c) for i, c in views.items()], recon.K)
    return X, {i: float(np.median(reprojection_errors(X, c, recon.K, *recon.poses[i]))) for i, c in views.items()}


def board_normal(recon, img, corners, grid):
    """Normal do tabuleiro no mundo, pela pose do tabuleiro (PnP com a grade metrica) na camera `img`."""
    _, rvec, _ = cv2.solvePnP(grid, corners, recon.K, None)
    return recon.poses[img][0].T @ cv2.Rodrigues(rvec)[0][:, 2]


def group_boards(recon, detections, max_reproj_error, grid, min_angle=2.0, max_normal_angle=15.0):
    """
    Agrupa as deteccoes por posicao do tabuleiro (ele foi mudado de lugar durante a gravacao;
    tambem serviria para varias copias). Duas deteccoes sao da mesma posicao se:
      - triangulando so as duas, os cantos reprojetam a menos de `max_reproj_error` px nas
        duas, com angulo de triangulacao >= `min_angle` graus no centro do tabuleiro. Sem o
        angulo minimo, quadros vizinhos (baseline quase nula) "triangulam" qualquer par de
        deteccoes com erro baixo, jogando os cantos para longe: o tabuleiro sai gigante;
      - as normais do tabuleiro no mundo (PnP em cada vista, independe da escala) diferem em
        ate `max_normal_angle` graus, o que separa posicoes em paredes diferentes.
    Com numero par de cantos nas duas direcoes o tabuleiro e simetrico por 180 graus e o
    OpenCV pode devolver a lista invertida: cada aresta guarda se a ordem precisa ser
    invertida, e a ordem de cada grupo e propagada a partir da raiz (a inversao gira o
    tabuleiro no proprio plano, entao nao muda a normal). Retorna a lista de grupos
    {imagem: cantos ja na ordem do grupo}.
    """
    imgs = list(detections)
    normals = {i: board_normal(recon, i, detections[i], grid) for i in imgs}
    centers = {i: camera_center(*recon.poses[i]) for i in imgs}
    edges = {i: [] for i in imgs}
    for a in range(len(imgs)):
        for b in range(a + 1, len(imgs)):
            i, j = imgs[a], imgs[b]
            if np.degrees(np.arccos(np.clip(normals[i] @ normals[j], -1, 1))) > max_normal_angle:
                continue
            best = None
            for flip in (False, True):
                cj = detections[j][::-1] if flip else detections[j]
                X, errors = board_reprojection(recon, {i: detections[i], j: cj})
                err = max(errors.values())
                if best is None or err < best[0]:
                    best = (err, flip, X.mean(axis=0))
            angle = triangulation_angles(best[2][None], centers[i], centers[j])[0]
            if best[0] <= max_reproj_error and angle >= min_angle:
                edges[i].append((j, best[1]))
                edges[j].append((i, best[1]))

    groups, seen = [], set()
    for root in sorted(imgs, key=lambda i: -len(edges[i])):
        if root in seen:
            continue
        flips, stack = {root: False}, [root]
        seen.add(root)
        while stack:
            i = stack.pop()
            for j, flip in edges[i]:
                if j not in seen:
                    seen.add(j)
                    flips[j] = flips[i] != flip
                    stack.append(j)
        groups.append({i: detections[i][::-1] if f else detections[i] for i, f in flips.items()})
    return groups


def checkerboard_scale(recon, images, pattern=(8, 6), square_size=0.025, max_reproj_error=5.0,
                       detections=None):
    """
    Fator de escala metrica a partir do(s) tabuleiro(s) de xadrez (`square_size` em metros):
      1. detecta os cantos em todas as imagens registradas (ou usa `detections`);
      2. agrupa as deteccoes por posicao do tabuleiro (`group_boards`), ja com a ordem dos cantos
         alinhada entre as vistas;
      3. triangula cada canto com todas as vistas do grupo (DLT) e descarta vistas com erro alto;
      4. compara as distancias entre todos os pares de cantos com as distancias reais da
         grade: o fator s minimiza sum (s d_modelo - d_real)^2, s = sum(d_m d_r) / sum(d_m^2),
         somando sobre todos os tabuleiros com pelo menos 2 vistas.
    Usar todos os pares (e nao so cantos vizinhos) aproveita distancias longas, menos
    sensiveis ao ruido; a escala de cada tabuleiro em separado mostra a consistencia
    (deriva de escala ao longo do trajeto). Retorna um dict com o fator, os tabuleiros (vistas,
    escala, cantos 3D ja na escala metrica, erros) e diagnosticos para o relatorio; aplique
    com `scale_reconstruction`.
    """
    if detections is None:
        detections = detect_checkerboards(images, list(recon.poses), pattern)
    detections = {i: c for i, c in detections.items() if i in recon.poses}

    cols, rows = pattern
    grid = np.float64([(c * square_size, r * square_size) for r in range(rows) for c in range(cols)])
    iu = np.triu_indices(len(grid), k=1)
    d_real = np.linalg.norm(grid[:, None] - grid[None], axis=2)[iu]

    boards = []
    grid3d = np.hstack([grid, np.zeros((len(grid), 1))])
    for views in group_boards(recon, detections, max_reproj_error, grid3d):
        if len(views) < 2:
            continue
        X, errors = board_reprojection(recon, views)
        good = {i: c for i, c in views.items() if errors[i] <= max_reproj_error}
        if 2 <= len(good) < len(views):
            views = good
            X, errors = board_reprojection(recon, views)
        d_model = np.linalg.norm(X[:, None] - X[None], axis=2)[iu]
        boards.append({"views": sorted(views), "X": X, "d_model": d_model, "reproj_errors": errors,
                       "scale": float(np.sum(d_model * d_real) / np.sum(d_model ** 2))})
    if not boards:
        raise ValueError(f"Tabuleiro em {len(detections)} imagem(ns) registrada(s), mas nenhum visto em 2 delas")

    d_model = np.concatenate([b["d_model"] for b in boards])
    d_all = np.tile(d_real, len(boards))
    s = float(np.sum(d_model * d_all) / np.sum(d_model ** 2))
    rel = np.abs(s * d_model - d_all) / d_all

    for b in boards:
        corners3d = b.pop("X") * s
        centered = corners3d - corners3d.mean(axis=0)
        b["corners3d"] = corners3d
        b["planarity_rms"] = float(np.linalg.svd(centered, compute_uv=False)[-1] / np.sqrt(len(centered)))
        b["pair_rel_error_mean"] = float(np.mean(np.abs(s * b.pop("d_model") - d_real) / d_real))
    scales = np.array([b["scale"] for b in boards])
    return {"scale": s, "boards": boards, "corners3d": np.vstack([b["corners3d"] for b in boards]),
            "views": sorted(v for b in boards for v in b["views"]),
            "reproj_errors": {i: e for b in boards for i, e in b["reproj_errors"].items()},
            "pair_rel_error_mean": float(rel.mean()), "pair_rel_error_max": float(rel.max()),
            "board_scale_spread": float(scales.std() / scales.mean()) if len(scales) > 1 else 0.0,
            "planarity_rms": float(np.mean([b["planarity_rms"] for b in boards])),
            "detected_in": sorted(detections)}


def point_colors(recon, all_kps, images):
    """Cor (BGR, 0-255) de cada ponto, na ordem de `recon.points`, tirada do pixel da sua primeira observacao."""
    pids = list(recon.points)
    colors = np.zeros((len(pids), 3), dtype=np.uint8)
    by_image = {}
    for n, pid in enumerate(pids):
        img, kp = next(iter(recon.tracks[pid].items()))
        by_image.setdefault(img, []).append((n, kp))
    for img, obs in by_image.items():  # um quadro por vez na memoria
        image = to_uint8(images[img])
        if image.ndim == 2:
            image = image[..., None].repeat(3, axis=2)
        xy = np.round(keypoint_coords(all_kps[img])[[kp for _, kp in obs]]).astype(int)
        x = np.clip(xy[:, 0], 0, image.shape[1] - 1)
        y = np.clip(xy[:, 1], 0, image.shape[0] - 1)
        colors[[n for n, _ in obs]] = image[y, x]
    return colors


def export_ply(recon, all_kps, images, out_path, camera_color=(0, 0, 255)):
    """Salva a nuvem colorida em PLY, com os centros das cameras como pontos de cor `camera_color` (BGR)."""
    centers = [camera_center(R, t) for R, t in recon.poses.values()]
    points = np.vstack([np.float64(list(recon.points.values())).reshape(-1, 3), centers])
    colors = np.vstack([point_colors(recon, all_kps, images), [camera_color] * len(centers)])
    save_point_cloud(points, out_path, colors)
