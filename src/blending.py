import ctypes
import gc
import time

import numpy as np
import cv2

from .homography import _rays, distort_points, undistort_points

def find_optimal_seam(cost_map, valid_mask):
    h, w = cost_map.shape
    INF = 1e9
    cost = np.where(valid_mask, cost_map, INF).astype(np.float64)

    dp = cost.copy()
    backtrack = np.zeros((h, w), dtype=np.int8)

    for y in range(1, h):
        up = dp[y - 1]
        left = np.roll(up, 1)
        left[0] = INF
        right = np.roll(up, -1)
        right[-1] = INF
        stacked = np.stack([up, left, right], axis=0)
        best_idx = np.argmin(stacked, axis=0)
        dp[y] = cost[y] + stacked[best_idx, np.arange(w)]
        offset_map = np.array([0, -1, 1])
        backtrack[y] = offset_map[best_idx]

    seam = np.zeros(h, dtype=int)
    seam[h - 1] = int(np.argmin(dp[h - 1]))
    for y in range(h - 2, -1, -1):
        seam[y] = seam[y + 1] + backtrack[y + 1, seam[y + 1]]
        seam[y] = int(np.clip(seam[y], 0, w - 1))

    return seam

def free_memory():
    gc.collect()
    # devolve ao sistema a memoria liberada, que fica fragmentada nas arenas das threads
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass

def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)

def linear_to_srgb(x):
    x = np.maximum(x, 0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)

def _to_proj(d, projection):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    rho = np.hypot(x, z)
    return np.arctan2(x, z), (y / rho if projection == "cylindrical" else np.arctan2(y, rho))

def _from_proj(a, b, projection):
    if projection == "cylindrical":
        return np.stack([np.sin(a), b, np.cos(a)], axis=-1)
    cb = np.cos(b)
    return np.stack([np.sin(a) * cb, np.sin(b), np.cos(a) * cb], axis=-1)

def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def make_canvas(rotations, camera, projection="cylindrical"):
    w, h = camera["width"], camera["height"]
    t = np.linspace(0, 1, 129)
    border = np.concatenate([
        np.stack([t * (w - 1), np.zeros_like(t)], axis=1),
        np.stack([t * (w - 1), np.full_like(t, h - 1)], axis=1),
        np.stack([np.zeros_like(t), t * (h - 1)], axis=1),
        np.stack([np.full_like(t, w - 1), t * (h - 1)], axis=1),
    ])
    rays_cam = _rays(undistort_points(border, camera), camera["f"])
    keys = sorted(rotations, key=lambda k: float(_to_proj(rotations[k][2], projection)[0]))
    angles = np.array([float(_to_proj(rotations[k][2], projection)[0]) for k in keys])
    gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
    start = (int(np.argmax(gaps)) + 1) % len(keys)
    keys = keys[start:] + keys[:start]

    footprints, centers, prev = {}, {}, None
    for k in keys:
        a, b = _to_proj(rays_cam @ rotations[k], projection)
        center = float(_to_proj(rotations[k][2], projection)[0])
        if prev is not None:
            center = prev + _wrap(center - prev)
        a = center + _wrap(a - center)
        footprints[k], centers[k], prev = (a.min(), a.max(), b.min(), b.max()), center, center

    hfov = np.median([fp[1] - fp[0] for fp in footprints.values()])
    full_circle = len(keys) > 2 and gaps.max() < 0.5 * hfov
    dup = None
    if full_circle:
        dup = max(keys) + 1
        a0, a1, b0, b1 = footprints[keys[0]]
        footprints[dup] = (a0 + 2 * np.pi, a1 + 2 * np.pi, b0, b1)
        centers[dup] = centers[keys[0]] + 2 * np.pi
        keys.append(dup)
    fp = np.array(list(footprints.values()))
    return {"f": camera["f"], "projection": projection, "order": keys, "dup": dup,
            "a_min": fp[:, 0].min(), "a_max": fp[:, 1].max(),
            "b_min": fp[:, 2].min(), "b_max": fp[:, 3].max(),
            "footprints": footprints, "centers": centers}

def canvas_size(canvas, s):
    fs = canvas["f"] * s
    return (int(np.ceil((canvas["a_max"] - canvas["a_min"]) * fs)),
            int(np.ceil((canvas["b_max"] - canvas["b_min"]) * fs)))

def image_roi(canvas, k, s):
    a0, a1, b0, b1 = canvas["footprints"][k]
    fs = canvas["f"] * s
    W, H = canvas_size(canvas, s)
    x0 = max(int(np.floor((a0 - canvas["a_min"]) * fs)), 0)
    y0 = max(int(np.floor((b0 - canvas["b_min"]) * fs)), 0)
    x1 = min(int(np.ceil((a1 - canvas["a_min"]) * fs)) + 1, W)
    y1 = min(int(np.ceil((b1 - canvas["b_min"]) * fs)) + 1, H)
    return x0, y0, x1, y1

def _sample_offsets(offsets, s, x0, y0, w, rows):
    ox, oy, dx, dy, s_off = offsets
    r = s_off / s
    mx = ((x0 + np.arange(w) + 0.5) * r - 0.5 - ox).astype(np.float32)
    my = ((y0 + rows + 0.5) * r - 0.5 - oy).astype(np.float32)
    MX, MY = np.meshgrid(mx, my)
    sx = cv2.remap(dx, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sy = cv2.remap(dy, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sx / r, sy / r

def warp_maps(R, camera, canvas, s, roi, src_scale, offsets=None, border=8, chunk_rows=256):
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    fs = canvas["f"] * s
    xs = x0 + np.arange(w) + 0.5
    map_x = np.empty((h, w), np.float32)
    map_y = np.empty((h, w), np.float32)
    valid = np.empty((h, w), bool)
    r2 = np.empty((h, w), np.float32)
    c = np.array([camera["cx"], camera["cy"]])
    for r0 in range(0, h, chunk_rows):
        r1 = min(r0 + chunk_rows, h)
        X, Y = np.meshgrid(xs, y0 + np.arange(r0, r1) + 0.5)
        if offsets is not None:
            dx, dy = _sample_offsets(offsets, s, x0, y0, w, np.arange(r0, r1))
            X, Y = X + dx, Y + dy
        A = canvas["a_min"] + X / fs
        B = canvas["b_min"] + Y / fs
        cam = _from_proj(A, B, canvas["projection"]) @ R.T
        z = cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = camera["f"] * cam[..., :2] / z[..., None]
        p = distort_points(u.reshape(-1, 2), camera).reshape(r1 - r0, w, 2)
        # descarta `border` px da borda do sensor, onde a demosaicagem deixa um tom magenta
        ok = ((z > 1e-6) & np.isfinite(p).all(axis=-1)
              & (p[..., 0] >= border) & (p[..., 0] <= camera["width"] - 1 - border)
              & (p[..., 1] >= border) & (p[..., 1] <= camera["height"] - 1 - border))
        p = np.where(ok[..., None], p, -1.0)
        map_x[r0:r1] = (p[..., 0] + 0.5) * src_scale - 0.5
        map_y[r0:r1] = (p[..., 1] + 0.5) * src_scale - 0.5
        valid[r0:r1] = ok
        r2[r0:r1] = ((p - c) ** 2).sum(axis=-1) / camera["D"] ** 2
    return map_x, map_y, valid, r2

def _vignetting(r2, coeffs):
    return np.exp(r2 * (coeffs[0] + r2 * (coeffs[1] + r2 * coeffs[2])))

def warp_corrected(src, R, camera, canvas, s, roi, src_scale, gain=1.0, vignetting=(0, 0, 0),
                   interpolation=cv2.INTER_LINEAR, offsets=None, chunk_rows=512):
    x0, y0, x1, y1 = roi
    img = np.empty((y1 - y0, x1 - x0, 3), np.float32)
    valid = np.empty((y1 - y0, x1 - x0), bool)
    scale = np.float32(np.iinfo(src.dtype).max)
    # em faixas de linhas: as conversoes de cor criam copias float do recorte inteiro
    for r0 in range(0, y1 - y0, chunk_rows):
        r1 = min(r0 + chunk_rows, y1 - y0)
        map_x, map_y, valid[r0:r1], r2 = warp_maps(R, camera, canvas, s, (x0, y0 + r0, x1, y0 + r1),
                                                   src_scale, offsets)
        part = cv2.remap(src, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)
        part = part.astype(np.float32) / scale
        factor = 1.0 / (gain * _vignetting(r2, vignetting))
        img[r0:r1] = linear_to_srgb(srgb_to_linear(part) * factor[..., None])
    return img, valid

def _resize_to_scale(img, full_size, s):
    w, h = full_size
    size = (max(int(round(w * s)), 1), max(int(round(h * s)), 1))
    if (img.shape[1], img.shape[0]) == size:
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def estimate_photometric(small_images, rotations, camera, canvas, s, samples_per_pair=200):
    keys = sorted(small_images)
    data = {}
    for k in keys:
        roi = image_roi(canvas, k, s)
        map_x, map_y, valid, r2 = warp_maps(rotations[k], camera, canvas, s, roi, s)
        img = cv2.remap(small_images[k], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        lin = srgb_to_linear(img.astype(np.float32) / np.iinfo(img.dtype).max)
        lum = lin @ np.float32([0.0722, 0.7152, 0.2126])
        valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        data[k] = (roi, lum, valid, r2)

    rng = np.random.default_rng(0)
    rows_i, rows_j, li, lj, ri, rj = [], [], [], [], [], []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            (ra, La, Va, Ra), (rb, Lb, Vb, Rb) = data[keys[a]], data[keys[b]]
            x0, y0 = max(ra[0], rb[0]), max(ra[1], rb[1])
            x1, y1 = min(ra[2], rb[2]), min(ra[3], rb[3])
            if x1 <= x0 or y1 <= y0:
                continue
            sa = (slice(y0 - ra[1], y1 - ra[1]), slice(x0 - ra[0], x1 - ra[0]))
            sb = (slice(y0 - rb[1], y1 - rb[1]), slice(x0 - rb[0], x1 - rb[0]))
            ok = (Va[sa] & Vb[sb] & (La[sa] > 0.02) & (Lb[sb] > 0.02)
                  & (La[sa] < 0.85) & (Lb[sb] < 0.85))
            idx = np.flatnonzero(ok)
            if len(idx) < 50:
                continue
            idx = rng.choice(idx, min(samples_per_pair, len(idx)), replace=False)
            rows_i.append(np.full(len(idx), a))
            rows_j.append(np.full(len(idx), b))
            li.append(La[sa].ravel()[idx]); lj.append(Lb[sb].ravel()[idx])
            ri.append(Ra[sa].ravel()[idx]); rj.append(Rb[sb].ravel()[idx])

    n = len(keys)
    if not rows_i:
        return {k: 1.0 for k in keys}, (0.0, 0.0, 0.0)
    I, J = np.concatenate(rows_i), np.concatenate(rows_j)
    y = np.log(np.concatenate(li)) - np.log(np.concatenate(lj))
    r_i, r_j = np.concatenate(ri), np.concatenate(rj)
    m = len(y)
    A = np.zeros((m + n, n + 3))
    A[np.arange(m), I] = 1
    A[np.arange(m), J] = -1
    for p in range(3):
        A[:m, n + p] = r_i ** (p + 1) - r_j ** (p + 1)
    A[m + np.arange(n), np.arange(n)] = 0.1 * np.sqrt(m / n)
    rhs = np.concatenate([y, np.zeros(n)])

    weights = np.ones(m + n)
    for _ in range(8):
        sol, *_ = np.linalg.lstsq(A * weights[:, None], rhs * weights, rcond=None)
        res = np.abs(A[:m] @ sol - y)
        weights[:m] = np.sqrt(np.where(res <= 0.05, 1.0, 0.05 / res))
    gains = {k: float(np.exp(sol[a])) for a, k in enumerate(keys)}
    return gains, tuple(float(c) for c in sol[n:])

def find_seams_sequential(seam_images, order, rotations, camera, canvas, s, gains, vignetting,
                          band_frac=0.25, center_weight=0.05, snapshot_counts=()):
    W, H = canvas_size(canvas, s)
    labels = np.full((H, W), -1, np.int16)
    acc = np.zeros((H, W, 3), np.float32)
    naive_sum = np.zeros((H, W, 3), np.float32)
    naive_cnt = np.zeros((H, W), np.float32)
    inconsistency = np.zeros((H, W), np.float32)
    fs = canvas["f"] * s
    kernel = np.ones((5, 5), np.uint8)

    prev, snapshots = None, []
    for n_added, k in enumerate(order, 1):
        x0, y0, x1, y1 = roi = image_roi(canvas, k, s)
        img, valid = warp_corrected(seam_images[k], rotations[k], camera, canvas, s, roi, s,
                                    gains[k], vignetting)
        valid = cv2.erode(valid.astype(np.uint8), kernel) > 0
        naive_sum[y0:y1, x0:x1][valid] += img[valid]
        naive_cnt[y0:y1, x0:x1][valid] += 1

        lab = labels[y0:y1, x0:x1]
        acc_roi = acc[y0:y1, x0:x1]
        overlap = valid & (lab >= 0)
        take = valid & (lab < 0)

        if prev is not None and overlap.any():
            x_mid = 0.5 * (canvas["centers"][prev] + canvas["centers"][k])
            x_mid = (x_mid - canvas["a_min"]) * fs - x0
            half_band = max(band_frac * (x1 - x0), 8.0)
            xx = np.arange(x1 - x0, dtype=np.float32)
            in_band = np.abs(xx - x_mid) <= half_band
            diff = cv2.blur(np.abs(acc_roi - img).mean(axis=2), (3, 3))
            inc = inconsistency[y0:y1, x0:x1]
            np.maximum(inc, np.where(overlap, diff, 0), out=inc)
            cost = diff + center_weight * ((xx - x_mid) / half_band) ** 2
            search = overlap & in_band[None, :]
            rows = np.flatnonzero(search.any(axis=1))
            seam = np.full(y1 - y0, x_mid)
            if len(rows):
                cols = np.flatnonzero(in_band)
                r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
                local = find_optimal_seam(cost[r0:r1, c0:c1], search[r0:r1, c0:c1]) + c0
                seam[r0:r1] = local
                seam[:r0], seam[r1:] = local[0], local[-1]
            take |= overlap & (xx[None, :] > seam[:, None])
        elif prev is None:
            take = valid.copy()

        lab[take] = k
        acc_roi[take] = img[take]
        prev = k
        if n_added in snapshot_counts:
            rows, cols = np.flatnonzero((labels >= 0).any(axis=1)), np.flatnonzero((labels >= 0).any(axis=0))
            snap = acc[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            snapshots.append((n_added, (np.clip(snap, 0, 1) * 255).astype(np.uint8)))

    naive = naive_sum / np.maximum(naive_cnt, 1)[..., None]
    return labels, naive, inconsistency, snapshots

def _flow_gray(img_a, img_b):
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(np.concatenate([ga.ravel(), gb.ravel()]), [1, 99])
    out = []
    for g in (ga, gb):
        g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5
        out.append((g * 255).astype(np.uint8))
    return out

def align_seams(sources, src_scale, labels, label_scale, rotations, camera, canvas, s,
                gains, vignetting, fade=48, max_flow=12.0, smooth=3.0):
    W, H = canvas_size(canvas, s)
    lab = cv2.resize(labels, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    ids, coords = [], []
    for sl_a, sl_b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        la, lb = lab[sl_a], lab[sl_b]
        ys, xs = np.nonzero((la != lb) & (la >= 0) & (lb >= 0))
        a, b = la[ys, xs], lb[ys, xs]
        ids.append(np.minimum(a, b) * 100000 + np.maximum(a, b))
        coords.append(np.stack([xs, ys], axis=1))
    ids, coords = np.concatenate(ids), np.concatenate(coords)
    diff_before, diff_after = [], []

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    acc = {}
    for pid in np.unique(ids):
        a, b = int(pid // 100000), int(pid % 100000)
        pts = coords[ids == pid]
        if len(pts) < 20:
            continue
        m = fade + 8
        x0, y0 = max(pts[:, 0].min() - m, 0), max(pts[:, 1].min() - m, 0)
        x1, y1 = min(pts[:, 0].max() + m + 2, W), min(pts[:, 1].max() + m + 2, H)
        roi = (x0, y0, x1, y1)
        img_a, va = warp_corrected(sources[a], rotations[a], camera, canvas, s, roi,
                                   src_scale, gains[a], vignetting)
        img_b, vb = warp_corrected(sources[b], rotations[b], camera, canvas, s, roi,
                                   src_scale, gains[b], vignetting)
        both = va & vb
        if both.sum() < 100:
            continue
        ga, gb = _flow_gray(img_a, img_b)
        flow_ab = dis.calc(ga, gb, None)
        flow_ba = dis.calc(gb, ga, None)
        hh, ww = ga.shape
        gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
        back = cv2.remap(flow_ba, gx + flow_ab[..., 0], gy + flow_ab[..., 1], cv2.INTER_LINEAR)
        fb_err = np.linalg.norm(flow_ab + back, axis=2)
        conf = (both & (fb_err < 0.5) & (np.linalg.norm(flow_ab, axis=2) < max_flow)).astype(np.float32)

        support = cv2.GaussianBlur(conf, (0, 0), smooth)
        v = cv2.GaussianBlur(flow_ab * conf[..., None], (0, 0), smooth) / np.maximum(support, 1e-6)[..., None]
        boundary = np.zeros((hh, ww), np.uint8)
        boundary[pts[:, 1] - y0, pts[:, 0] - x0] = 1
        dist = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 5)
        weight = np.clip(1 - dist / fade, 0, 1) * np.clip(support / 0.3, 0, 1)
        half = 0.5 * v * weight[..., None]
        for k, sign in ((a, -1.0), (b, 1.0)):
            acc.setdefault(k, []).append((x0, y0, sign * half))

        on_seam = (boundary > 0) & both
        aligned_a = cv2.remap(img_a, gx - half[..., 0], gy - half[..., 1], cv2.INTER_LINEAR)
        aligned_b = cv2.remap(img_b, gx + half[..., 0], gy + half[..., 1], cv2.INTER_LINEAR)
        diff_before.append(np.abs(img_a - img_b).mean(axis=2)[on_seam])
        diff_after.append(np.abs(aligned_a - aligned_b).mean(axis=2)[on_seam])

    offsets = {}
    for k, parts in acc.items():
        ox0 = min(p[0] for p in parts)
        oy0 = min(p[1] for p in parts)
        ox1 = max(p[0] + p[2].shape[1] for p in parts)
        oy1 = max(p[1] + p[2].shape[0] for p in parts)
        field = np.zeros((oy1 - oy0, ox1 - ox0, 2), np.float32)
        for px, py, d in parts:
            field[py - oy0:py - oy0 + d.shape[0], px - ox0:px - ox0 + d.shape[1]] += d
        offsets[k] = (ox0, oy0, np.ascontiguousarray(field[..., 0]), np.ascontiguousarray(field[..., 1]), s)
    mean = lambda parts: 255 * float(np.concatenate(parts).mean()) if parts else float("nan")
    seam_metrics = {"diferenca_costura_antes_alinhamento": mean(diff_before),
                    "diferenca_costura_depois_alinhamento": mean(diff_after)}
    return offsets, seam_metrics

def _gaussian_pyramid(img, levels):
    pyr = [img]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr

def _laplacian_pyramid(img, levels):
    pyr = _gaussian_pyramid(img, levels)
    for l in range(levels):
        pyr[l] -= cv2.pyrUp(pyr[l + 1], dstsize=(pyr[l].shape[1], pyr[l].shape[0]))
    return pyr

def blend_labels(get_source, order, rotations, camera, canvas, s, labels, label_scale,
                 gains, vignetting, offsets, region, num_bands=7):
    X0, Y0, X1, Y1 = region
    W, H = X1 - X0, Y1 - Y0
    unit = 2 ** num_bands
    Wp, Hp = -(-W // unit) * unit, -(-H // unit) * unit
    r = label_scale / s
    iy = np.clip(((Y0 + np.arange(Hp) + 0.5) * r).astype(int), 0, labels.shape[0] - 1)
    ix = np.clip(((X0 + np.arange(Wp) + 0.5) * r).astype(int), 0, labels.shape[1] - 1)
    up_labels = labels[iy[:, None], ix[None, :]]
    up_labels[H:], up_labels[:, W:] = -1, -1
    margin = unit * 4

    bands = [np.zeros((Hp >> l, Wp >> l, 3), np.float32) for l in range(num_bands + 1)]
    weights = [np.zeros((Hp >> l, Wp >> l), np.float32) for l in range(num_bands + 1)]
    interpolation = cv2.INTER_LANCZOS4 if s >= 1.0 else cv2.INTER_LINEAR

    for k in order:
        ys, xs = np.nonzero(labels == k)
        if len(xs) == 0:
            continue
        x0 = max(int(xs.min() / r) - X0 - margin, 0) // unit * unit
        y0 = max(int(ys.min() / r) - Y0 - margin, 0) // unit * unit
        x1 = min(-(-(int((xs.max() + 1) / r) - X0 + 1 + margin) // unit) * unit, Wp)
        y1 = min(-(-(int((ys.max() + 1) / r) - Y0 + 1 + margin) // unit) * unit, Hp)
        if x1 <= x0 or y1 <= y0:
            continue

        src, src_scale = get_source(k)
        img, valid = warp_corrected(src, rotations[k], camera, canvas, s, (x0 + X0, y0 + Y0, x1 + X0, y1 + Y0),
                                    src_scale, gains[k], vignetting, interpolation, offsets.get(k))
        chosen = up_labels[y0:y1, x0:x1] == k
        mask = (chosen & valid).astype(np.float32) + 1e-3 * (~chosen & valid)
        lap = _laplacian_pyramid(img, num_bands)
        gm = _gaussian_pyramid(mask, num_bands)
        for l in range(num_bands + 1):
            sl = (slice(y0 >> l, (y0 >> l) + gm[l].shape[0]), slice(x0 >> l, (x0 >> l) + gm[l].shape[1]))
            bands[l][sl] += lap[l] * gm[l][..., None]
            weights[l][sl] += gm[l]

    # operacoes no lugar: em resolucao total cada copia do panorama ocupa ~2 GB
    result = None
    for l in range(num_bands, -1, -1):
        band = bands[l]
        bands[l] = None
        band /= np.maximum(weights[l], 1e-8)[..., None]
        if result is not None:
            band += cv2.pyrUp(result, dstsize=(band.shape[1], band.shape[0]))
        result = band
    valid = weights[0][:H, :W] > 1e-6
    result = result[:H, :W]
    np.clip(result, 0, 1, out=result)
    return result, valid

def largest_valid_rectangle(mask, max_dim=1500):
    h, w = mask.shape
    f = min(1.0, max_dim / max(h, w))
    small = cv2.resize(mask.astype(np.float32), (max(int(w * f), 1), max(int(h * f), 1)),
                       interpolation=cv2.INTER_AREA) >= 0.999
    sh, sw = small.shape
    heights = np.zeros(sw, int)
    best = (0, 0, 0, 0, 0)
    for y in range(sh):
        heights = np.where(small[y], heights + 1, 0)
        stack = []
        for x in range(sw + 1):
            hx = heights[x] if x < sw else 0
            start = x
            while stack and stack[-1][1] >= hx:
                s0, s_h = stack.pop()
                if s_h * (x - s0) > best[0]:
                    best = (s_h * (x - s0), s0, y - s_h + 1, x, y + 1)
                start = s0
            stack.append((start, hx))
    _, x0, y0, x1, y1 = best
    return (int(np.ceil((x0 + 1) / f)), int(np.ceil((y0 + 1) / f)),
            int(np.floor((x1 - 1) / f)), int(np.floor((y1 - 1) / f)))

def ghost_comparison(inconsistency, panorama, crop, seam_scale, s, order, rotations, camera,
                     canvas, gains, vignetting, get_source, size=(1200, 800)):
    r = seam_scale / s
    w, h = max(int(size[0] * s), 64), max(int(size[1] * s), 64)
    bw, bh = max(int(w * r), 3), max(int(h * r), 3)
    score = cv2.boxFilter(inconsistency, -1, (bw, bh))
    x0, y0, x1, y1 = crop
    inside = np.zeros_like(score, bool)
    inside[int(np.ceil(y0 * r)) + bh // 2:int(y1 * r) - bh // 2, int(np.ceil(x0 * r)) + bw // 2:int(x1 * r) - bw // 2] = True
    cy, cx = np.unravel_index(np.argmax(np.where(inside, score, -1)), score.shape)
    rx0 = int(np.clip(cx / r - w / 2, x0, x1 - w))
    ry0 = int(np.clip(cy / r - h / 2, y0, y1 - h))
    roi = (rx0, ry0, rx0 + w, ry0 + h)

    total, count = np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
    interpolation = cv2.INTER_LANCZOS4 if s >= 1.0 else cv2.INTER_LINEAR
    for k in order:
        kx0, ky0, kx1, ky1 = image_roi(canvas, k, s)
        if kx1 <= roi[0] or kx0 >= roi[2] or ky1 <= roi[1] or ky0 >= roi[3]:
            continue
        src, src_scale = get_source(k)
        img, valid = warp_corrected(src, rotations[k], camera, canvas, s, roi, src_scale,
                                    gains[k], vignetting, interpolation)
        total[valid] += img[valid]
        count[valid] += 1
    naive_crop = total / np.maximum(count, 1)[..., None]
    final_crop = panorama[roi[1] - y0:roi[3] - y0, roi[0] - x0:roi[2] - x0]
    gap = np.ones((h, max(w // 60, 4), 3), np.float32)
    out = (np.clip(np.hstack([naive_crop, gap, final_crop]), 0, 1) * 255).astype(np.uint8)
    font = w / 1000
    for text, x in (("ingenua (media)", 10), ("costura otima + multibanda", w + gap.shape[1] + 10)):
        y = int(40 * font)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), max(int(6 * font), 2))
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), max(int(2 * font), 1))
    return out

def compose_panorama(work_images, work_scale, load_full, rotations, camera,
                     projection="cylindrical", compose_scale=1.0, num_bands=7, align=None):
    t0 = time.time()
    canvas = make_canvas(rotations, camera, projection)
    order, dup = canvas["order"], canvas["dup"]
    full_size = (camera["width"], camera["height"])
    work_images, rotations = dict(work_images), dict(rotations)
    if dup is not None:
        work_images[dup], rotations[dup] = work_images[order[0]], rotations[order[0]]
    W, H = canvas_size(canvas, compose_scale)
    span = np.degrees(canvas["a_max"] - canvas["a_min"])
    print(f"  Canvas {'cilindrico' if projection == 'cylindrical' else 'esferico'}: {W}x{H} px "
          + ("(volta completa de 360 graus)" if dup is not None else f"({span:.1f} graus de largura)"))

    ph_scale = work_scale / 4
    gains, vignetting = estimate_photometric(
        {k: _resize_to_scale(work_images[k], full_size, ph_scale) for k in order},
        rotations, camera, canvas, ph_scale)
    if dup is not None:
        gains[order[0]] = gains[dup] = float(np.sqrt(gains[order[0]] * gains[dup]))
    g = np.array(list(gains.values()))
    print(f"  Fotometria: ganhos em [{g.min():.3f}, {g.max():.3f}]; vinheta nos cantos "
          f"{np.log2(_vignetting(1.0, vignetting)):+.2f} EV")

    seam_scale = work_scale / 2
    print(f"  Costuras... [{time.time() - t0:.0f} s]")
    seam_images = {k: _resize_to_scale(work_images[k], full_size, seam_scale) for k in order}
    snapshot_counts = sorted({min(2 ** i, len(order)) for i in range(1, 16)})
    labels, naive, inconsistency, snapshots = find_seams_sequential(
        seam_images, order, rotations, camera, canvas, seam_scale, gains, vignetting,
        snapshot_counts=snapshot_counts)
    del seam_images
    free_memory()

    print(f"  Alinhamento local nas costuras... [{time.time() - t0:.0f} s]")
    offsets, seam_metrics = align_seams(work_images, work_scale, labels, seam_scale, rotations,
                                        camera, canvas, work_scale, gains, vignetting, **(align or {}))
    free_memory()

    def get_source(k):
        if compose_scale <= work_scale:
            img = work_images[k]
        else:
            img = load_full(order[0] if k == dup else k)
        return _resize_to_scale(img, full_size, compose_scale), compose_scale

    covered = labels >= 0
    rs = compose_scale / seam_scale
    if dup is None:
        cx0, cy0, cx1, cy1 = largest_valid_rectangle(covered)
        x0, x1 = int(np.ceil(cx0 * rs)), int(cx1 * rs)
    else:
        fs_s = canvas["f"] * seam_scale
        turn = int(round(2 * np.pi * fs_s))
        cx0 = int((canvas["centers"][order[0]] - canvas["a_min"]) * fs_s)
        cx1 = cx0 + turn
        cols = covered[:, cx0:cx1]
        cy0 = int(np.argmax(cols, axis=0).max())
        cy1 = int(covered.shape[0] - np.argmax(cols[::-1], axis=0).max())
        turn_c = int(round(2 * np.pi * canvas["f"] * compose_scale))
        margin = turn_c // 32
        x0 = int(round(cx0 * rs)) - margin
        x1 = x0 + turn_c + 2 * margin
    y0, y1 = int(np.ceil(cy0 * rs)), int(cy1 * rs)

    print(f"  Blending multibanda... [{time.time() - t0:.0f} s]")
    bands = max(1, int(round(num_bands + np.log2(compose_scale))))
    panorama, valid = blend_labels(get_source, order, rotations, camera, canvas, compose_scale,
                                   labels, seam_scale, gains, vignetting, offsets, (x0, y0, x1, y1), bands)
    free_memory()
    if 1 - valid.mean() > 1e-4:
        print(f"  Aviso: {100 * (1 - valid.mean()):.3f}% do recorte sem foto valida")
    if dup is not None:
        gray = panorama.mean(axis=2)
        k = min(64, margin)
        shifts = np.arange(-margin + k, margin)
        diffs = [np.abs(gray[:, margin + t] - gray[:, margin + t + turn_c]).mean() for t in shifts]
        first_col = margin + int(shifts[int(np.argmin(diffs))])
        out = panorama[:, first_col:first_col + turn_c]
        alpha = np.linspace(0, 1, k, dtype=np.float32)[None, :, None]
        out[:, -k:] = (1 - alpha) * out[:, -k:] + alpha * panorama[:, first_col - k:first_col]
        panorama = out
        x0, x1 = x0 + first_col, x0 + first_col + turn_c
        cx0 = int(round(x0 / rs))
        cx1 = cx0 + turn
    print(f"  Recorte sem bordas: {x1 - x0}x{y1 - y0} px")

    ghost = ghost_comparison(inconsistency, panorama, (x0, y0, x1, y1), seam_scale, compose_scale,
                             order, rotations, camera, canvas, gains, vignetting, get_source)
    seams = _draw_seams(naive, labels)
    metrics = {"largura_angular_graus": 360.0 if dup is not None else span,
               "volta_completa_360": dup is not None,
               "ganho_min": float(g.min()), "ganho_max": float(g.max()),
               "vinheta_cantos_EV": float(np.log2(_vignetting(1.0, vignetting))),
               **seam_metrics}
    return {"panorama": panorama, "naive": naive[cy0:cy1, cx0:cx1],
            "seams": seams[cy0:cy1, cx0:cx1], "snapshots": snapshots, "ghost": ghost,
            "metrics": metrics}

def _draw_seams(image, labels):
    vis = image.copy()
    edges = np.zeros(labels.shape, bool)
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edges[1:, :] |= labels[1:, :] != labels[:-1, :]
    edges &= labels >= 0
    vis[edges] = (0, 0, 1)
    return vis
