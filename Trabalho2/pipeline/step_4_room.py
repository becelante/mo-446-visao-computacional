"""
Etapa 4c (Variante A): limpeza da nuvem, dimensoes da sala, planta baixa e trajetoria.

Trabalha na reconstrucao ja metrica e alinhada (z para cima, chao em z = 0,
paredes alinhadas com x/y). As paredes sao detectadas com RANSAC (planos
verticais); a largura/comprimento sao as distancias entre as paredes opostas mais
externas, uma de cada lado do centro das cameras, e a altura (pe-direito) e a
distancia chao-teto.
"""
import numpy as np

from .step_4_scale_align import detect_planes, statistical_outlier_mask


def clean_cloud(xyz, rgb, room=None, sor_k=16, sor_std=2.0, crop_margin=0.3, extra_mask=None):
    """
    Limpeza: remocao estatistica de outliers + recorte (crop) na caixa da sala
    (com margem), o que elimina pontos vistos atraves de janelas/portas.
    Retorna (xyz, rgb, mascara em relacao a entrada).
    """
    mask = np.ones(len(xyz), bool) if extra_mask is None else extra_mask.copy()
    idx = np.where(mask)[0]
    m = statistical_outlier_mask(xyz[idx], sor_k, sor_std)
    mask[idx[~m]] = False
    if room is not None and room.get("bounds") is not None:
        (x0, x1), (y0, y1), (z0, z1) = room["bounds"]
        inside = ((xyz[:, 0] > x0 - crop_margin) & (xyz[:, 0] < x1 + crop_margin) &
                  (xyz[:, 1] > y0 - crop_margin) & (xyz[:, 1] < y1 + crop_margin) &
                  (xyz[:, 2] > z0 - crop_margin) & (xyz[:, 2] < z1 + crop_margin))
        mask &= inside
    return xyz[mask], rgb[mask], mask


def voxel_downsample(xyz, rgb, voxel):
    """Um ponto (media) por voxel."""
    if voxel <= 0 or len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(counts), 3))
    col = np.zeros((len(counts), 3))
    np.add.at(out, inv, xyz)
    np.add.at(col, inv, rgb.astype(float))
    return out / counts[:, None], (col / counts[:, None]).astype(np.uint8)


def _refit_offset(points, axis):
    return float(np.median(points[:, axis]))


def _wall_dict(pts, w):
    axis = int(np.argmax(np.abs(w["n"][:2])))
    extent = np.percentile(pts[:, 1 - axis], [2, 98])
    return {"axis": "x" if axis == 0 else "y", "offset": _refit_offset(pts, axis),
            "normal": w["n"].tolist(), "n_inliers": w["n_inliers"],
            "span": [float(extent[0]), float(extent[1])], "z_span": [float(pts[:, 2].min()), float(pts[:, 2].max())]}


def measure_room(xyz, thresh=0.03, min_frac_wall=0.15, min_ceiling_z=1.5, center_xy=None):
    """
    Mede a sala numa nuvem alinhada. Retorna dict com paredes, largura (menor
    dimensao horizontal), comprimento (maior), altura, limites e angulos entre
    paredes vizinhas (Extra: comparar com 90 graus).
    `min_ceiling_z`: o teto so pode ser um plano acima disso (o pipeline usa a altura
    das cameras + 0,7 m): topos de moveis e faixas horizontais das paredes tambem
    formam "planos" horizontais. Sem teto, o topo da sala e o percentil 99,5 de z.
    `center_xy`: ponto dentro da sala (mediana das cameras) que separa as paredes
    opostas; sem ele, usa o meio da extensao 1-99% dos pontos.
    """
    z = xyz[:, 2]
    # teto: plano horizontal mais alto com suporte
    horiz = detect_planes(xyz, thresh, max_planes=6, min_inliers=max(50, len(xyz) // 300),
                          normal_ok=lambda n: np.abs(n[:, 2]) > np.cos(np.deg2rad(10)), seed=2)
    hz = []
    for p in horiz:
        pts = xyz[p["idx"]]
        hz.append((float(np.median(pts[:, 2])), p["n_inliers"], p))
    floor_z = min([h for h in hz if abs(h[0]) < 0.15], key=lambda h: abs(h[0]), default=(0.0, 0, None))[0]
    ceil_cands = [h for h in hz if h[0] > min_ceiling_z]
    ceil_z = max(ceil_cands, key=lambda h: h[0])[0] if ceil_cands else None
    top = ceil_z if ceil_z else float(np.percentile(z, 99.5))

    band = (z > floor_z + 0.15 * (top - floor_z)) & (z < floor_z + 0.85 * (top - floor_z))
    pb = xyz[band]
    walls = detect_planes(pb, thresh, max_planes=14, min_inliers=max(50, len(pb) // 150),
                          normal_ok=lambda n: (np.abs(n[:, 2]) < np.sin(np.deg2rad(10))) &
                                              (np.max(np.abs(n[:, :2]), 1) > np.cos(np.deg2rad(20))),
                          seed=3)
    wall_list = [_wall_dict(pb[w["idx"]], w) for w in walls]

    dims, bounds, extremes = {}, [], {}
    for ai, ax in enumerate("xy"):
        ws = [w for w in wall_list if w["axis"] == ax]
        p2, p98 = np.percentile(pb[:, ai], [1, 99]) if len(pb) else (np.nan, np.nan)
        mid = center_xy[ai] if center_xy is not None else (p2 + p98) / 2
        # paredes opostas: uma de cada lado do centro (as cameras ficam dentro da sala).
        # Paredes muito densas de um lado podem esgotar os planos do RANSAC global e
        # deixar o outro lado so com moveis; por isso o RANSAC roda tambem so com os
        # pontos de cada lado, e o suporte minimo e relativo ao maior plano do lado.
        side = {}
        for s, sel, pick in (("lo", pb[:, ai] < mid, min), ("hi", pb[:, ai] > mid, max)):
            cand = [w for w in ws if (w["offset"] < mid) == (s == "lo")]
            if sel.sum() >= 30:
                ps = pb[sel]
                more = detect_planes(ps, thresh, max_planes=12, min_inliers=max(30, len(ps) // 100),
                                     normal_ok=lambda n: (np.abs(n[:, 2]) < np.sin(np.deg2rad(10))) &
                                                         (np.abs(n[:, ai]) > np.cos(np.deg2rad(20))), seed=4)
                more = [_wall_dict(ps[w["idx"]], w) for w in more]
                wall_list += more
                cand += more
            if cand:
                big_s = max(w["n_inliers"] for w in cand)
                cand = [w for w in cand if w["n_inliers"] >= min_frac_wall * big_s]
            side[s] = pick(cand, key=lambda w: w["offset"]) if cand else None
        lo, hi = side["lo"], side["hi"]
        if lo is not None and hi is not None:
            dims[ax] = hi["offset"] - lo["offset"]
            bounds.append((lo["offset"], hi["offset"]))
            extremes[ax] = (lo, hi)
            src = "paredes (RANSAC)"
        elif lo is not None or hi is not None:  # so um lado: parede de um lado, percentil do outro
            a = lo["offset"] if lo is not None else float(p2)
            b = hi["offset"] if hi is not None else float(p98)
            dims[ax] = b - a
            bounds.append((a, b))
            src = "parede (RANSAC) + percentil 1-99% no lado sem parede"
        else:  # sem paredes: extensao robusta dos pontos
            dims[ax] = float(p98 - p2)
            bounds.append((float(p2), float(p98)))
            src = "percentis 1-99% (paredes nao encontradas)"
        dims[ax + "_source"] = src
    bounds.append((floor_z, top))

    angles = []
    if "x" in extremes and "y" in extremes:
        for wx in extremes["x"]:
            for wy in extremes["y"]:
                nx, ny = np.array(wx["normal"]), np.array(wy["normal"])
                # angulo entre os planos (90 graus para paredes perpendiculares)
                ang = np.degrees(np.arccos(np.clip(abs(nx @ ny), 0, 1)))
                angles.append({"walls": f"x={wx['offset']:.2f} / y={wy['offset']:.2f}",
                               "angle_deg": float(ang), "error_deg": float(abs(90 - ang))})
    w_dim, l_dim = sorted([dims["x"], dims["y"]])
    return {
        "width_m": float(w_dim), "length_m": float(l_dim),
        "x_extent_m": float(dims["x"]), "y_extent_m": float(dims["y"]),
        "x_source": dims["x_source"], "y_source": dims["y_source"],
        "height_m": float(ceil_z - floor_z) if ceil_z else None,
        "floor_z": float(floor_z), "ceiling_z": float(ceil_z) if ceil_z else None,
        "bounds": [list(b) for b in bounds], "walls": wall_list, "wall_angles": angles,
        "floor_area_m2": float(dims["x"] * dims["y"]),
    }


def trajectory_length(centers):
    return float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()) if len(centers) > 1 else 0.0


def valid_cameras(centers, room, unit=1.0, margin_m=0.5, win=3, jump_m=1.0):
    """
    Mascara das cameras com pose coerente. O mapeador global pode deixar algumas
    cameras mal ancoradas longe da sala (dezenas de metros), o que infla o
    comprimento do trajeto e zera a deriva em %. Descarta (1) os centros fora da
    caixa da sala com folga de `margin_m` e (2) saltos isolados: centros a mais de
    max(`jump_m`, 8 passos medianos) da mediana movel dos `win` vizinhos de cada lado.
    """
    centers = np.asarray(centers)
    (x0, x1), (y0, y1) = room["bounds"][0], room["bounds"][1]
    m = margin_m * unit
    floor = room.get("floor_z") or 0.0
    top = max(room.get("ceiling_z") or floor, floor + 2.0 * unit) + 1.0 * unit  # teto detectado pode ser um movel
    ok = ((centers[:, 0] > x0 - m) & (centers[:, 0] < x1 + m) & (centers[:, 1] > y0 - m) & (centers[:, 1] < y1 + m)
          & (centers[:, 2] > floor - m) & (centers[:, 2] < top))
    if ok.mean() < 0.5:  # a caixa que esta errada (medida da sala falhou), nao as cameras: fica so o teste de salto
        ok[:] = True
    idx = np.where(ok)[0]
    if len(idx) > 2 * win + 1:
        C = centers[idx]
        med = np.array([np.median(C[max(0, i - win):i + win + 1], axis=0) for i in range(len(C))])
        step = float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
        ok[idx[np.linalg.norm(C - med, axis=1) > max(jump_m * unit, 8 * step)]] = False
    return ok


def plot_floorplan(xyz, rgb, centers, Rs, room, out_path, title="Planta baixa", drift=None):
    """Planta baixa: nuvem vista de cima (sem teto), paredes, cotas e trajetoria da camera."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z0 = room["floor_z"]
    top = room["ceiling_z"] or np.percentile(xyz[:, 2], 98)
    sel = (xyz[:, 2] > z0 + 0.05) & (xyz[:, 2] < top - 0.1)
    fig, ax = plt.subplots(figsize=(9, 9 * max(0.5, room["y_extent_m"] / max(room["x_extent_m"], 1e-6))))
    sc = ax.scatter(xyz[sel, 0], xyz[sel, 1], c=xyz[sel, 2], s=0.4, cmap="viridis", alpha=0.6, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("altura z (m)")

    (x0, x1), (y0, y1) = room["bounds"][0], room["bounds"][1]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "-", color="#c0392b", lw=1.6, label="paredes (modelo)")

    # cotas
    off = 0.12 * max(x1 - x0, y1 - y0)
    ax.annotate("", (x0, y0 - off / 2), (x1, y0 - off / 2), arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text((x0 + x1) / 2, y0 - off / 2 - off * 0.25, f"{room['x_extent_m'] * 100:.1f} cm",
            ha="center", va="top", fontsize=11, weight="bold")
    ax.annotate("", (x1 + off / 2, y0), (x1 + off / 2, y1), arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(x1 + off / 2 + off * 0.15, (y0 + y1) / 2, f"{room['y_extent_m'] * 100:.1f} cm",
            ha="left", va="center", rotation=90, fontsize=11, weight="bold")

    ax.plot(centers[:, 0], centers[:, 1], "-", color="#e67e22", lw=1.4, label="trajetoria da camera")
    step = max(1, len(centers) // 40)
    for C, R in zip(centers[::step], Rs[::step]):
        d = R[2, :2]  # eixo optico projetado no plano
        if np.linalg.norm(d) > 1e-6:
            d = d / np.linalg.norm(d) * 0.25
            ax.arrow(C[0], C[1], d[0], d[1], head_width=0.06, color="#d35400", lw=0.8, length_includes_head=True)
    if len(centers):
        ax.plot(*centers[0, :2], "o", color="#27ae60", ms=9, label="inicio (C1)")
        ax.plot(*centers[-1, :2], "s", color="#8e44ad", ms=8, label="fim (CN)")
    if drift is not None:
        ax.set_title(f"{title}  |  deriva do laco = {drift['drift_cm']:.1f} cm ({drift['drift_pct']:.2f}% do trajeto)")
    else:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(x0 - 1.6 * off, x1 + 2.0 * off)
    ax.set_ylim(y0 - 1.6 * off, y1 + 1.0 * off)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_occupancy(xyz, room, out_path, cell=0.02):
    """Planta como imagem de densidade (histograma 2D dos pontos das paredes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z0 = room["floor_z"]
    top = room["ceiling_z"] or np.percentile(xyz[:, 2], 98)
    sel = (xyz[:, 2] > z0 + 0.1) & (xyz[:, 2] < top - 0.1)
    (x0, x1), (y0, y1) = room["bounds"][0], room["bounds"][1]
    m = 0.4
    H, xe, ye = np.histogram2d(xyz[sel, 0], xyz[sel, 1],
                               bins=[np.arange(x0 - m, x1 + m, cell), np.arange(y0 - m, y1 + m, cell)])
    fig, ax = plt.subplots(figsize=(8, 8 * (y1 - y0 + 2 * m) / (x1 - x0 + 2 * m)))
    ax.imshow(np.log1p(H.T), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="gray_r")
    ax.set_title(f"Ocupacao vista de cima (celula de {cell * 100:.0f} cm)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
