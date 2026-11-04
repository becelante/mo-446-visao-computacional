"""
Densificacao (MVS) por plane-sweep na GPU, depois do SfM (recomendado na Variante A).

  1. vizinhas: para cada imagem de referencia, as `n_sources` imagens que mais compartilham
     pontos esparsos com ela, com angulo de triangulacao razoavel (baseline util);
  2. plane-sweep: para cada profundidade d (amostrada em 1/d entre os percentis dos pontos
     esparsos vistos pela referencia), cada vizinha e reprojetada na referencia supondo a
     cena no plano frontal a distancia d, e a semelhanca e a NCC numa janela; o custo de d
     e a media das 2 melhores vizinhas (tolera oclusao em parte delas). A profundidade do
     pixel e o minimo do custo, refinado por uma parabola;
  3. filtros: NCC minima, textura minima (parede lisa casa com qualquer profundidade) e
     consistencia geometrica: projetado na vizinha e trazido de volta pela profundidade
     dela, o pixel precisa cair a menos de 1 px de onde saiu, com profundidade ate 1%
     diferente, em pelo menos `min_consistent` vizinhas;
  4. fusao: os pixels aprovados viram pontos 3D coloridos, reduzidos por voxel.

Supoe a reconstrucao ja na escala metrica e sem distorcao de lente (a mesma K do SfM).
"""
from collections import Counter

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d

from .step_1_keypoints_detector import to_gray8
from .step_4_sfm import camera_center


def select_sources(recon, ref, candidates, n_sources=4, min_angle=3.0, max_angle=40.0, sample=200):
    """As `n_sources` imagens de `candidates` com mais pontos em comum com `ref` e angulo mediano no intervalo."""
    shared = Counter()
    pids = observed_points(recon, ref)
    for pid in pids:
        shared.update(i for i in recon.tracks[pid] if i != ref and i in candidates)
    C_ref = camera_center(*recon.poses[ref])
    chosen = []
    for img, count in shared.most_common():
        common = [pid for pid in pids if img in recon.tracks[pid]][:sample]
        X = np.float64([recon.points[p] for p in common])
        r1, r2 = X - C_ref, X - camera_center(*recon.poses[img])
        cos = np.sum(r1 * r2, 1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))
        angle = np.degrees(np.median(np.arccos(np.clip(cos, -1, 1))))
        if min_angle <= angle <= max_angle:
            chosen.append(img)
        if len(chosen) == n_sources:
            break
    return chosen


def observed_points(recon, img):
    """pids dos pontos esparsos observados por `img`."""
    pids = recon.pids_of(img)
    return pids[pids >= 0].tolist()


def depth_range(recon, img, low=2, high=98, margin=1.25):
    """Faixa de profundidades da cena vista por `img`, pelos pontos esparsos que ela observa."""
    R, t = recon.poses[img]
    X = np.float64([recon.points[pid] for pid in observed_points(recon, img)])
    z = (X @ R.T + t)[:, 2]
    z = z[z > 0]
    return float(np.percentile(z, low) / margin), float(np.percentile(z, high) * margin)


def box_mean(x, win):
    return F.avg_pool2d(x, win, stride=1, padding=win // 2, count_include_pad=False)


def ncc(ref, src, win):
    """NCC por pixel entre ref e src (B x 1 x H x W), numa janela win x win."""
    mr, ms = box_mean(ref, win), box_mean(src, win)
    vr = box_mean(ref * ref, win) - mr * mr
    vs = box_mean(src * src, win) - ms * ms
    cov = box_mean(ref * src, win) - mr * ms
    return cov / torch.sqrt((vr * vs).clamp_min(1e-10))


def plane_sweep(ref_img, src_imgs, K, pose_ref, poses_src, dmin, dmax, n_depths=128, win=7, device="cuda"):
    """
    Mapa de profundidade da referencia (H x W) e a NCC do minimo. As imagens sao tensores
    1 x 1 x H x W em [0, 1] na GPU, ja na resolucao de K.
    """
    _, _, H, W = ref_img.shape
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                          torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    pix = torch.stack([u.ravel(), v.ravel(), torch.ones_like(u).ravel()])  # 3 x HW
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    rays = torch.linalg.inv(Kt) @ pix
    depths = 1 / torch.linspace(1 / dmin, 1 / dmax, n_depths, device=device)
    R_ref, t_ref = (torch.tensor(np.float32(x), device=device) for x in pose_ref)

    warp = []  # x_src(d) = A d + b para cada vizinha
    for R_src, t_src in poses_src:
        R_src, t_src = torch.tensor(np.float32(R_src), device=device), torch.tensor(np.float32(t_src), device=device)
        R_rel = R_src @ R_ref.T
        t_rel = t_src - R_rel @ t_ref
        warp.append((Kt @ R_rel @ rays, (Kt @ t_rel)[:, None]))

    cost = torch.empty(n_depths, H, W, device=device)
    for k, d in enumerate(depths):
        scores = []
        for (A, b), src in zip(warp, src_imgs):
            x = A * d + b
            z = x[2].clamp_min(1e-6)
            grid = torch.stack([x[0] / z / (W - 1) * 2 - 1, x[1] / z / (H - 1) * 2 - 1], -1).view(1, H, W, 2)
            sampled = F.grid_sample(src, grid, align_corners=True, padding_mode="zeros")
            score = ncc(ref_img, sampled, win)[0, 0]
            outside = (grid.abs() > 1).any(-1)[0] | (x[2] <= 0).view(H, W)
            scores.append(torch.where(outside, torch.full_like(score, -1.0), score))
        scores = torch.stack(scores)
        best2 = scores.topk(min(2, len(scores)), dim=0).values.mean(0)
        cost[k] = 1 - best2

    idx = cost.argmin(0)
    c0 = cost.gather(0, (idx - 1).clamp(0, n_depths - 1)[None])[0]
    c1 = cost.gather(0, idx[None])[0]
    c2 = cost.gather(0, (idx + 1).clamp(0, n_depths - 1)[None])[0]
    denom = c0 - 2 * c1 + c2
    offset = torch.where(denom > 1e-6, 0.5 * (c0 - c2) / denom, torch.zeros_like(denom)).clamp(-0.5, 0.5)
    inv = 1 / depths
    inv_step = inv[1] - inv[0] if n_depths > 1 else torch.tensor(0.0, device=device)
    depth = 1 / (inv[idx] + offset * inv_step)
    return depth, 1 - c1


def geometric_consistency(depth_ref, K, pose_ref, others, max_px=1.0, max_rel=0.01):
    """
    Quantas vizinhas confirmam cada pixel da referencia. `others` e uma lista de
    (mapa de profundidade, pose) das vizinhas, todas na mesma resolucao e K.
    """
    device = depth_ref.device
    H, W = depth_ref.shape
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                          torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    K_inv = torch.linalg.inv(Kt)
    R_ref, t_ref = (torch.tensor(np.float32(x), device=device) for x in pose_ref)
    pix = torch.stack([u.ravel(), v.ravel(), torch.ones(H * W, device=device)])
    X = R_ref.T @ (K_inv @ pix * depth_ref.ravel() - t_ref[:, None])  # mundo
    count = torch.zeros(H * W, device=device)
    for depth_src, (R_src, t_src) in others:
        R_src, t_src = torch.tensor(np.float32(R_src), device=device), torch.tensor(np.float32(t_src), device=device)
        x = Kt @ (R_src @ X + t_src[:, None])
        z = x[2].clamp_min(1e-6)
        grid = torch.stack([x[0] / z / (W - 1) * 2 - 1, x[1] / z / (H - 1) * 2 - 1], -1).view(1, 1, -1, 2)
        d_src = F.grid_sample(depth_src[None, None], grid, mode="nearest", align_corners=True)[0, 0, 0]
        X_back = R_src.T @ (K_inv @ torch.stack([x[0] / z, x[1] / z, torch.ones_like(z)]) * d_src - t_src[:, None])
        y = Kt @ (R_ref @ X_back + t_ref[:, None])
        zy = y[2].clamp_min(1e-6)
        err = torch.sqrt((y[0] / zy - pix[0]) ** 2 + (y[1] / zy - pix[1]) ** 2)
        ok = (x[2] > 0) & (d_src > 0) & (err < max_px) & ((zy - depth_ref.ravel()).abs() < max_rel * depth_ref.ravel())
        count += ok.float()
    return count.view(H, W)


def dense_reconstruction(recon, images, max_side=640, n_sources=4, n_depths=128, ref_step=2, min_ncc=0.5,
                         min_texture=0.02, min_consistent=2, voxel_size=0.01, device="cuda", log=print):
    """
    Nuvem densa (Open3D) com os pixels aprovados de cada referencia (uma a cada `ref_step`
    cameras registradas). As imagens sao reduzidas para `max_side` px no lado maior.
    """
    if not torch.cuda.is_available():
        device = "cpu"
    registered = sorted(recon.poses)
    H0, W0 = images[registered[0]].shape[:2]
    s = min(1.0, max_side / max(H0, W0))
    K = recon.K.copy()
    K[:2] *= s
    size = (int(round(W0 * s)), int(round(H0 * s)))

    def gray(i):
        g = cv2.resize(to_gray8(images[i]), size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(g).float().div(255).to(device)[None, None]

    refs = registered[::ref_step]
    depth_maps, ncc_maps, sources = {}, {}, {}
    with torch.inference_mode():
        for n, ref in enumerate(refs):
            src = select_sources(recon, ref, set(registered), n_sources)
            if len(src) < 2:
                continue
            dmin, dmax = depth_range(recon, ref)
            depth, score = plane_sweep(gray(ref), [gray(i) for i in src], K, recon.poses[ref],
                                       [recon.poses[i] for i in src], dmin, dmax, n_depths, device=device)
            ref_gray = gray(ref)
            texture = (box_mean(ref_gray * ref_gray, 7) - box_mean(ref_gray, 7) ** 2).clamp_min(0).sqrt()[0, 0]
            depth[(score < min_ncc) | (texture < min_texture)] = 0
            depth_maps[ref], ncc_maps[ref] = depth.cpu(), score.cpu()
            sources[ref] = src
            if (n + 1) % 20 == 0:
                log(f"  {n + 1}/{len(refs)} mapas de profundidade")

        points, colors = [], []
        ref_set = set(depth_maps)
        for ref, depth in depth_maps.items():
            neighbors = select_sources(recon, ref, ref_set - {ref}, n_sources, min_angle=1.0)
            if not neighbors:
                continue
            d = depth.to(device)
            count = geometric_consistency(d, K, recon.poses[ref],
                                          [(depth_maps[i].to(device), recon.poses[i]) for i in neighbors])
            keep = (d > 0) & (count >= min(min_consistent, len(neighbors)))
            v, u = torch.nonzero(keep, as_tuple=True)
            if not len(v):
                continue
            R, t = recon.poses[ref]
            pix = np.stack([u.cpu().numpy(), v.cpu().numpy(), np.ones(len(u))]).astype(np.float64)
            X_cam = np.linalg.inv(K) @ pix * d[v, u].cpu().numpy()
            points.append(((X_cam - t[:, None]).T @ R))
            color = cv2.resize(images[ref], size, interpolation=cv2.INTER_AREA)
            colors.append(color[v.cpu().numpy(), u.cpu().numpy()][:, ::-1] / 255.0)

    pcd = o3d.geometry.PointCloud()
    if points:
        pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    n_raw = len(pcd.points)
    if voxel_size and n_raw:
        pcd = pcd.voxel_down_sample(voxel_size)
    log(f"  densa: {len(depth_maps)} mapas, {n_raw} pixels aprovados, {len(pcd.points)} pontos apos voxel de {voxel_size} m")
    return pcd, {"n_depth_maps": len(depth_maps), "n_fused_raw": n_raw, "n_points": len(pcd.points)}
