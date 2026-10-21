"""Extra X1 - bundle adjustment: cada foto e uma camera que so gira, com focal,
ponto principal e distorcao radial comuns (o RAW nao vem com a distorcao corrigida)."""
import numpy as np

from .etapa5_homography import _get_H

def make_camera(width, height, focal):
    return {"f": float(focal), "k1": 0.0, "k2": 0.0, "k3": 0.0,
            "width": int(width), "height": int(height),
            "cx": (width - 1) / 2.0, "cy": (height - 1) / 2.0,
            "D": 0.5 * float(np.hypot(width, height))}

def points_to_full_res(pts, scale):
    return (np.asarray(pts, dtype=np.float64) + 0.5) / scale - 0.5

def _homography_to_centered_full_res(H, scale, camera):
    s = scale
    A = np.array([[s, 0, 0.5 * s - 0.5], [0, s, 0.5 * s - 0.5], [0, 0, 1]])
    T = np.array([[1, 0, camera["cx"]], [0, 1, camera["cy"]], [0, 0, 1]])
    M = np.linalg.inv(A @ T) @ H @ (A @ T)
    return M / M[2, 2]

def _radial_factor(r2, k1, k2, k3):
    return 1 + r2 * (k1 + r2 * (k2 + r2 * k3))

def undistort_points(pts, camera):
    u = np.asarray(pts, dtype=np.float64) - (camera["cx"], camera["cy"])
    r2 = (u ** 2).sum(axis=1) / camera["D"] ** 2
    return u * _radial_factor(r2, camera["k1"], camera["k2"], camera["k3"])[:, None]

def distort_points(u, camera):
    D = camera["D"]
    r_d = np.linspace(0, 1.6 * D, 16384)
    r_u = r_d * _radial_factor((r_d / D) ** 2, camera["k1"], camera["k2"], camera["k3"])
    monotonic = np.concatenate([[True], np.cumprod(np.diff(r_u) > 0).astype(bool)])
    r_d, r_u = r_d[monotonic], r_u[monotonic]

    u = np.asarray(u, dtype=np.float64)
    r = np.sqrt((u ** 2).sum(axis=1))
    ratio = np.interp(r, r_u, r_d, right=np.nan) / np.maximum(r, 1e-12)
    ratio[r < 1e-9] = 1.0
    return u * ratio[:, None] + (camera["cx"], camera["cy"])

def _rays(u, focal):
    v = np.concatenate([u / focal, np.ones((len(u), 1))], axis=1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _rodrigues(rvecs):
    rvecs = np.atleast_2d(rvecs)
    theta = np.linalg.norm(rvecs, axis=1)
    k = rvecs / np.maximum(theta, 1e-12)[:, None]
    K = np.zeros((len(rvecs), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)

def _nearest_rotation(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return -R if np.linalg.det(R) < 0 else R

def estimate_focal(homographies, scale, camera):
    candidates = []
    for H in homographies.values():
        h = _homography_to_centered_full_res(H, scale, camera).ravel()
        estimates = []
        for d1, d2, v1n, v2n in (
            (h[6] * h[7], (h[7] - h[6]) * (h[7] + h[6]),
             -(h[0] * h[1] + h[3] * h[4]), h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2),
            (h[0] * h[3] + h[1] * h[4], h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2,
             -h[2] * h[5], h[5] ** 2 - h[2] ** 2),
        ):
            v1 = v1n / d1 if d1 != 0 else -1.0
            v2 = v2n / d2 if d2 != 0 else -1.0
            v1, v2 = max(v1, v2), min(v1, v2)
            if v1 > 0 and v2 > 0:
                estimates.append(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
            elif v1 > 0:
                estimates.append(np.sqrt(v1))
        if len(estimates) == 2:
            candidates.append(np.sqrt(estimates[0] * estimates[1]))
    return float(np.median(candidates)) if candidates else None

def initial_rotations(homographies, connectivity, nodes, ref, scale, camera):
    K = np.diag([camera["f"], camera["f"], 1.0])
    K_inv = np.linalg.inv(K)
    rotations = {ref: np.eye(3)}
    nodes = set(nodes)
    while True:
        best = None
        for i in rotations:
            for j in nodes - rotations.keys():
                if connectivity[i, j] > 0 and (best is None or connectivity[i, j] > best[2]):
                    best = (i, j, connectivity[i, j])
        if best is None:
            return rotations
        i, j, _ = best
        H = _get_H(i, j, homographies)
        R_ij = _nearest_rotation(K_inv @ _homography_to_centered_full_res(H, scale, camera) @ K)
        rotations[j] = R_ij @ rotations[i]

def bundle_adjust(inlier_points, rotations, camera, scale, ref,
                  max_points_per_pair=100, max_pair_error=4.0, loss_scale=1.0):
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix

    rng = np.random.default_rng(0)
    obs = []
    for (i, j), (pts_i, pts_j) in inlier_points.items():
        if i not in rotations or j not in rotations:
            continue
        idx = np.arange(len(pts_i))
        if len(idx) > max_points_per_pair:
            idx = rng.choice(idx, max_points_per_pair, replace=False)
        obs.append((i, j, points_to_full_res(pts_i[idx], scale), points_to_full_res(pts_j[idx], scale)))

    camera = dict(camera)
    nodes = [ref] + sorted(n for n in rotations if n != ref)
    pos = {n: k for k, n in enumerate(nodes)}
    n_rot = 3 * (len(nodes) - 1)
    D = camera["D"]

    def ideal(P, k, c):
        u = P - c
        return u * _radial_factor((u ** 2).sum(axis=1) / D ** 2, *k)[:, None]

    for round_ in range(2):
        I = np.concatenate([np.full(len(p), pos[i]) for i, _, p, _ in obs])
        J = np.concatenate([np.full(len(p), pos[j]) for _, j, p, _ in obs])
        P_i = np.concatenate([p for _, _, p, _ in obs])
        P_j = np.concatenate([q for _, _, _, q in obs])
        R0 = np.stack([rotations[n] for n in nodes])
        f0, c0 = camera["f"], np.array([camera["cx"], camera["cy"]])

        def unpack(x):
            deltas = np.vstack([np.zeros(3), x[:n_rot].reshape(-1, 3)])
            return _rodrigues(deltas) @ R0, f0 * x[n_rot], x[n_rot + 1:n_rot + 4], c0 + 100.0 * x[n_rot + 4:]

        def residuals(x):
            R, f, k, c = unpack(x)
            w_i = np.einsum("nlk,nl->nk", R[I], _rays(ideal(P_i, k, c), f))
            w_j = np.einsum("nlk,nl->nk", R[J], _rays(ideal(P_j, k, c), f))
            return (f * (w_i - w_j)).ravel()

        n_obs = len(I)
        base = 3 * np.arange(n_obs)
        rows, cols = [], []
        for cam in (I, J):
            sel = cam > 0
            for a in range(3):
                for b in range(3):
                    rows.append(base[sel] + a)
                    cols.append(3 * (cam[sel] - 1) + b)
        for a in range(3):
            for b in range(6):
                rows.append(base + a)
                cols.append(np.full(n_obs, n_rot + b))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        sparsity = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3 * n_obs, n_rot + 6))

        x0 = np.concatenate([np.zeros(n_rot), [1.0, camera["k1"], camera["k2"], camera["k3"], 0.0, 0.0]])
        result = least_squares(residuals, x0, jac_sparsity=sparsity, method="trf",
                               loss="soft_l1", f_scale=loss_scale, x_scale="jac", max_nfev=100)
        R, f, k, c = unpack(result.x)
        rotations = {n: R[pos[n]] for n in nodes}
        camera.update(f=float(f), k1=float(k[0]), k2=float(k[1]), k3=float(k[2]),
                      cx=float(c[0]), cy=float(c[1]))

        errors = np.linalg.norm(residuals(result.x).reshape(-1, 3), axis=1)
        stats, per_pair, offset = {}, [], 0
        for i, j, p, q in obs:
            e = errors[offset:offset + len(p)]
            offset += len(p)
            stats[(i, j)] = {"n": len(e), "mediana": float(np.median(e)), "media": float(e.mean())}
            per_pair.append((i, j, p, q, e))
        print(f"  BA passada {round_ + 1}: {len(obs)} pares, {n_obs} correspondencias, "
              f"erro mediano {np.median(errors):.3f} px, RMS {np.sqrt((errors ** 2).mean()):.3f} px | "
              f"f={f:.1f} px, c=({c[0]:.1f}, {c[1]:.1f}), k=({k[0]:.4f}, {k[1]:.4f}, {k[2]:.4f})")
        if round_ == 0:
            keep = {(i, j) for (i, j), st in stats.items() if st["mediana"] <= max_pair_error}
            # nenhuma camera pode ficar sem restricao: mantem os 2 melhores pares de cada uma
            for node in nodes:
                own = sorted((st["mediana"], pair) for pair, st in stats.items() if node in pair)
                if sum(pair in keep for _, pair in own) < 2:
                    keep.update(pair for _, pair in own[:2])
            obs = []
            for i, j, p, q, e in per_pair:
                good = e <= max(3.0 * np.median(e), 2.0)
                if (i, j) in keep and good.sum() >= 8:
                    obs.append((i, j, p[good], q[good]))
            print(f"  BA: {len(per_pair) - len(obs)} pares inconsistentes descartados "
                  f"(erro mediano > {max_pair_error} px)")

    return rotations, camera, stats

def wave_correct(rotations):
    x_axes = np.array([R[0] for R in rotations.values()])
    _, eigvecs = np.linalg.eigh(x_axes.T @ x_axes)
    down = eigvecs[:, 0]
    if np.dot(down, np.array([R[1] for R in rotations.values()]).sum(axis=0)) < 0:
        down = -down
    forward = np.array([R[2] for R in rotations.values()]).sum(axis=0)
    forward -= np.dot(forward, down) * down
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    W = np.stack([right, down, forward])
    return {k: R @ W.T for k, R in rotations.items()}
