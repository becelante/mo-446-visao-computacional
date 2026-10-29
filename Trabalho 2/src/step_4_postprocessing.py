"""
Pos-processamento da nuvem esparsa (Variante A: ambiente interno), com Open3D.

  - limpeza: remocao estatistica de outliers, downsample em voxels e crop;
  - poses: frustums das cameras e a trajetoria percorrida;
  - alinhamento: chao no plano z = 0 com z para cima e paredes paralelas aos eixos;
  - planta baixa: nuvem vista de cima, com largura e comprimento medidos no modelo;
  - planos: alturas das superficies horizontais (mesas, teto) e angulo entre paredes vizinhas;
  - deriva do laco e densidade da nuvem (Etapa 5).

Tudo assume a reconstrucao ja em metros (`checkerboard_scale` + `scale_reconstruction`).
Depois de `align_to_floor`, aplique T tambem a reconstrucao (`transform_reconstruction`)
para que cameras, frustums e trajetoria fiquem no mesmo referencial da nuvem.
"""
import numpy as np
import open3d as o3d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step_4_sfm import camera_center, point_colors


def to_open3d(recon, all_kps, images):
    """Nuvem do Open3D com os pontos da reconstrucao e as cores das imagens."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.float64(list(recon.points.values())).reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(point_colors(recon, all_kps, images)[:, ::-1] / 255.0)
    return pcd


def clean_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0, voxel_size=None, crop_percentile=None, bbox=None):
    """
    Limpa a nuvem em ate tres passos, nesta ordem:
      - remocao estatistica: para cada ponto, a distancia media aos `nb_neighbors` vizinhos;
        sai quem fica acima de media + `std_ratio` desvios (pontos isolados, tipicamente
        matches errados que sobreviveram ao SfM);
      - crop: mantem a caixa entre os percentis `crop_percentile` e 100 - `crop_percentile`
        de cada eixo, ou a caixa `bbox` = (min_xyz, max_xyz) dada a mao (p.ex. para cortar
        o que aparece pela janela, fora da sala);
      - downsample: um ponto por voxel de lado `voxel_size` (deixa a densidade uniforme).
    Retorna (nuvem limpa, dict com o numero de pontos apos cada passo).
    """
    counts = {"inicial": len(pcd.points)}
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    counts["outliers"] = len(pcd.points)
    if crop_percentile is not None or bbox is not None:
        if bbox is None:
            pts = np.asarray(pcd.points)
            bbox = (np.percentile(pts, crop_percentile, axis=0), np.percentile(pts, 100 - crop_percentile, axis=0))
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(*bbox))
        counts["crop"] = len(pcd.points)
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
        counts["downsample"] = len(pcd.points)
    return pcd, counts


def camera_centers(recon):
    """Centros das cameras registradas, em ordem de imagem (a ordem da captura)."""
    return np.float64([camera_center(*recon.poses[img]) for img in sorted(recon.poses)])


def camera_frustums(recon, width, height, size=0.15, color=(0.1, 0.4, 1.0), step=1):
    """
    Um frustum (piramide) por camera: o apice e o centro optico e a base, a `size` metros
    a frente, tem a proporcao da imagem. Mostra posicao e orientacao de cada foto. `step` > 1
    desenha uma a cada `step` cameras (com milhares, as piramides cobrem a cena).
    """
    frustums = o3d.geometry.LineSet()
    for img in sorted(recon.poses)[::step]:
        R, t = recon.poses[img]
        extrinsic = np.eye(4)
        extrinsic[:3, :3], extrinsic[:3, 3] = R, t
        frustums += o3d.geometry.LineSet.create_camera_visualization(width, height, recon.K, extrinsic, size)
    frustums.paint_uniform_color(color)
    return frustums


def trajectory(recon, color=(1.0, 0.1, 0.1)):
    """Linha ligando os centros das cameras na ordem da captura."""
    centers = camera_centers(recon)
    lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(centers),
                                 o3d.utility.Vector2iVector([(k, k + 1) for k in range(len(centers) - 1)]))
    lines.paint_uniform_color(color)
    return lines


def rotation_between(a, b):
    """Rotacao que leva o vetor unitario a em b (formula de Rodrigues)."""
    v, c = np.cross(a, b), np.dot(a, b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def align_to_floor(pcd, recon, distance_threshold=0.02, max_planes=8, max_tilt=15.0, min_fraction=0.2):
    """
    Transformacao 4 x 4 que leva a nuvem para um referencial "de planta": z para cima,
    chao em z = 0 e paredes paralelas a x e y.
      1. "Para cima" inicial: o oposto da media do eixo y das cameras (no OpenCV o y da
         imagem aponta para baixo e, andando pela sala, a camera fica quase de pe).
      2. Procura o chao com RANSAC: extrai ate `max_planes` planos, um de cada vez, e
         guarda os horizontais (normal a ate `max_tilt` graus da vertical) que ficam abaixo
         das cameras. O chao e o mais baixo deles com pelo menos `min_fraction` dos pontos
         do maior; pegar o maior seria arriscado, porque um piso liso tem poucos pontos e
         uma mesa grande pode ter mais. Sem chao, fica a vertical das cameras e a altura do
         chao vira o percentil 1 dos pontos.
      3. Gira em torno de z para alinhar as paredes aos eixos (`wall_yaw`).
    Retorna (T, info), com info["floor_found"] dizendo se o plano do chao foi achado.
    """
    up = -np.mean([R[1] for R, _ in recon.poses.values()], axis=0)
    up /= np.linalg.norm(up)
    centers = camera_centers(recon)
    candidates = []  # (altura das cameras sobre o plano, n de pontos, n, d)
    rest = pcd
    for _ in range(max_planes):
        if len(rest.points) < 100:
            break
        plane, inliers = rest.segment_plane(distance_threshold, ransac_n=3, num_iterations=2000)
        n, d = np.array(plane[:3]), plane[3]
        if np.dot(n, up) < 0:
            n, d = -n, -d
        horizontal = np.degrees(np.arccos(np.clip(np.dot(n, up), -1, 1))) <= max_tilt
        heights = centers @ n + d
        if horizontal and np.all(heights > 0):  # cameras acima do plano
            candidates.append((heights.mean(), len(inliers), n, d))
        rest = rest.select_by_index(inliers, invert=True)
    floor = None
    if candidates:
        largest = max(c[1] for c in candidates)
        _, _, n, d = max(c for c in candidates if c[1] >= min_fraction * largest)
        floor = (n, d)

    if floor is not None:
        up = floor[0]
    R_level = rotation_between(up, np.array([0.0, 0.0, 1.0]))
    pts = np.asarray(pcd.points) @ R_level.T
    floor_z = -floor[1] if floor is not None else np.percentile(pts[:, 2], 1)

    yaw = wall_yaw(pts[:, :2])
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    T = np.eye(4)
    T[:3, :3] = R_yaw @ R_level
    T[2, 3] = -floor_z
    return T, {"floor_found": floor is not None, "up": up}


def wall_yaw(xy, cell=0.05, max_points=300_000, seed=0):
    """
    Giro (rad) em torno de z que deixa as paredes paralelas aos eixos. Com a parede alinhada a um
    eixo, os pontos dela caem todos na mesma faixa de x (ou de y) e o histograma fica com picos
    estreitos; a nitidez e medida pela soma dos quadrados das contagens dos histogramas de x e y.
    Busca de 0 a 90 graus em passos de 0,5 e refino de 0,05. O retangulo de area minima
    (cv2.minAreaRect), usado antes, e puxado pelos pontos soltos fora da sala e errou por ~40 graus.
    """
    rng = np.random.default_rng(seed)
    if len(xy) > max_points:
        xy = xy[rng.choice(len(xy), max_points, replace=False)]
    xy = xy - xy.mean(axis=0)

    def sharpness(theta):
        c, s = np.cos(theta), np.sin(theta)
        rotated = xy @ np.array([[c, s], [-s, c]])
        return sum(np.sum(np.histogram(v, bins=np.arange(v.min(), v.max() + cell, cell))[0].astype(float) ** 2)
                   for v in rotated.T)

    coarse = np.radians(np.arange(0, 90, 0.5))
    best = coarse[np.argmax([sharpness(t) for t in coarse])]
    fine = best + np.radians(np.arange(-0.5, 0.5, 0.05))
    return float(fine[np.argmax([sharpness(t) for t in fine])])


def transform_points(points, T):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def wall_positions(coords, cell=0.02, peak_fraction=0.25):
    """
    Posicao das duas paredes perpendiculares a um eixo: no histograma da coordenada, uma
    parede vira um pico (concentra pontos de toda a sua area numa faixa fina). Cada lado do
    centro (mediana) e tratado separadamente: pega o pico mais externo com pelo menos
    `peak_fraction` do maior pico daquele lado, e refina pela media ponderada do pico e dos
    vizinhos. Com um limiar global, uma parede muito texturizada (no video C, a de baixo tem
    varias vezes mais pontos) apagava a oposta. Pontos soltos alem das paredes nao chegam a
    formar pico, entao nao inflam a medida (o que acontece com min/max ou percentis).
    """
    edges = np.arange(coords.min(), coords.max() + 2 * cell, cell)
    hist, edges = np.histogram(coords, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")  # um bin isolado nao e parede
    middle = np.searchsorted(centers, np.median(coords))

    def refine(k):
        sl = slice(max(k - 1, 0), k + 2)
        return float(np.average(centers[sl], weights=hist[sl]))

    low, high = smooth[:middle], smooth[middle:]
    first = np.flatnonzero(low >= peak_fraction * low.max())[0]
    last = middle + np.flatnonzero(high >= peak_fraction * high.max())[-1]
    return refine(first), refine(last)


def room_dimensions(points, cell=0.02, peak_fraction=0.25):
    """
    Largura e comprimento da sala com a nuvem ja alinhada (`align_to_floor`): distancia
    entre as paredes opostas em x e em y (`wall_positions`). Supoe que cada parede tem
    textura suficiente para formar um pico; se uma parede lisa ficar sem pontos, a medida
    cai no movel mais externo, entao confira na planta baixa. Retorna dict com as medidas
    (m) e a caixa das paredes (min, max) em xy.
    """
    (x0, x1), (y0, y1) = (wall_positions(points[:, k], cell, peak_fraction) for k in (0, 1))
    return {"width": min(x1 - x0, y1 - y0), "length": max(x1 - x0, y1 - y0),
            "x_extent": x1 - x0, "y_extent": y1 - y0,
            "bounds": (np.array([x0, y0]), np.array([x1, y1]))}


def draw_floor_plan(points, centers, dims, out_path, cell=0.02, max_height=None):
    """
    Planta baixa: histograma 2D (escala log) dos pontos vistos de cima, em celulas de
    `cell` m. As paredes aparecem como linhas escuras, porque concentram pontos em toda a
    altura. `max_height` corta pontos acima (p.ex. o teto, que cobriria a planta).
    Sobrepoe a trajetoria das cameras e a caixa com largura e comprimento medidos.
    """
    if max_height is not None:
        points = points[points[:, 2] <= max_height]
    lo, hi = dims["bounds"]
    margin = 0.3
    # a figura cobre a nuvem (percentis 1 a 99), a trajetoria e a caixa das paredes: se a medida
    # automatica errar a parede, isso fica visivel em vez de os pontos sumirem do recorte
    view_lo = np.minimum.reduce([np.percentile(points[:, :2], 1, axis=0), centers[:, :2].min(0), lo]) - margin
    view_hi = np.maximum.reduce([np.percentile(points[:, :2], 99, axis=0), centers[:, :2].max(0), hi]) + margin
    bins = [np.arange(view_lo[0], view_hi[0] + cell, cell), np.arange(view_lo[1], view_hi[1] + cell, cell)]
    hist, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=bins)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.log1p(hist.T), origin="lower", cmap="Greys",
              extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]))
    ax.plot(centers[:, 0], centers[:, 1], "-", color="tab:red", lw=1.5, label="trajetoria")
    ax.plot(*centers[0, :2], "o", color="tab:green", label="inicio")
    ax.plot(*centers[-1, :2], "s", color="tab:orange", label="fim")
    ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec="tab:blue", ls="--"))
    ax.annotate(f"{dims['x_extent']:.2f} m", ((lo[0] + hi[0]) / 2, lo[1]), xytext=(0, -14),
                textcoords="offset points", ha="center", color="tab:blue")
    ax.annotate(f"{dims['y_extent']:.2f} m", (lo[0], (lo[1] + hi[1]) / 2), xytext=(-14, 0),
                textcoords="offset points", va="center", rotation=90, color="tab:blue")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def measure_interactive(pcd):
    """
    Abre a nuvem para medir uma dimensao (p.ex. a altura da porta): marque 2 pontos com
    shift + clique e feche a janela. Retorna a distancia entre eles (m), ou None.
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("shift + clique em 2 pontos, depois feche")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked = vis.get_picked_points()
    if len(picked) < 2:
        return None
    pts = np.asarray(pcd.points)[picked[-2:]]
    return float(np.linalg.norm(pts[0] - pts[1]))


def detect_planes(points, distance_threshold=0.02, max_planes=12, min_points=500, seed=0):
    """
    Planos dominantes por RANSAC sequencial (extrai o maior, tira os inliers, repete). Retorna
    lista de (normal unitaria, d, numero de inliers) com n . X + d = 0, do maior para o menor.
    """
    o3d.utility.random.seed(seed)
    rest = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points)))
    planes = []
    for _ in range(max_planes):
        if len(rest.points) < min_points:
            break
        plane, inliers = rest.segment_plane(distance_threshold, ransac_n=3, num_iterations=2000)
        if len(inliers) < min_points:
            break
        n = np.array(plane[:3])
        planes.append((n / np.linalg.norm(n), plane[3] / np.linalg.norm(n), len(inliers)))
        rest = rest.select_by_index(inliers, invert=True)
    return planes


def plane_measurements(points, max_tilt=10.0, **plane_kw):
    """
    Com a nuvem alinhada (z para cima, chao em z = 0):
      - horizontais: altura (m) de cada plano quase horizontal, p.ex. tampos de mesa e teto;
      - paredes: planos quase verticais; para cada par nao paralelo, o angulo entre eles (o
        extra da Variante A compara com 90 graus), e para cada par paralelo a distancia entre
        eles (confere largura/comprimento por outro caminho).
    """
    planes = detect_planes(points, **plane_kw)
    tilt = [np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1))) for n, _, _ in planes]
    horizontal = [{"height": float(-d / n[2]), "n_points": k} for (n, d, k), a in zip(planes, tilt) if a <= max_tilt]
    walls = [(n, d, k) for (n, d, k), a in zip(planes, tilt) if a >= 90 - max_tilt]
    angles, distances = [], []
    for a in range(len(walls)):
        for b in range(a + 1, len(walls)):
            (n1, d1, k1), (n2, d2, k2) = walls[a], walls[b]
            h1, h2 = n1[:2] / np.linalg.norm(n1[:2]), n2[:2] / np.linalg.norm(n2[:2])
            angle = float(np.degrees(np.arccos(np.clip(abs(h1 @ h2), 0, 1))))
            pair = {"walls": (a, b), "n_points": (k1, k2)}
            if angle > 45:
                angles.append({**pair, "angle": angle, "deviation_from_90": abs(90 - angle)})
            else:
                sign = np.sign(h1 @ h2)
                distances.append({**pair, "angle": angle, "distance": float(abs(d1 - sign * d2))})
    return {"horizontal": sorted(horizontal, key=lambda h: h["height"]), "n_walls": len(walls),
            "wall_angles": angles, "parallel_walls": distances}


def loop_drift(recon):
    """
    Deriva do laco: o video termina onde comecou, entao o centro da primeira e o da ultima
    camera (C_1 e C_N, em ordem de captura) deveriam coincidir. Retorna delta = |C_N - C_1| (m),
    o comprimento do trajeto (soma dos passos entre cameras consecutivas) e delta em % dele.
    """
    centers = camera_centers(recon)
    delta = float(np.linalg.norm(centers[-1] - centers[0]))
    length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    imgs = sorted(recon.poses)
    return {"delta_m": delta, "path_length_m": length, "delta_percent": 100 * delta / length if length else float("nan"),
            "first_camera": imgs[0], "last_camera": imgs[-1]}


def point_density(n_points, dims, ceiling_height=None):
    """
    Densidade aproximada (pontos/m^2): pela area da planta (largura x comprimento) e, com a
    altura do teto, pela area das superficies da sala (chao, teto e quatro paredes), onde os
    pontos de fato estao.
    """
    floor = dims["x_extent"] * dims["y_extent"]
    out = {"floor_area_m2": floor, "points_per_m2_floor": n_points / floor if floor > 0 else float("nan")}
    if ceiling_height:
        surface = 2 * floor + 2 * ceiling_height * (dims["x_extent"] + dims["y_extent"])
        out.update(surface_area_m2=surface, points_per_m2_surface=n_points / surface)
    return out


def save_cameras(recon, width, height, out_dir, size=0.15):
    """Frustums e trajetoria como LineSet em PLY (abrem no Open3D e no MeshLab)."""
    o3d.io.write_line_set(str(out_dir / "cameras.ply"), camera_frustums(recon, width, height, size))
    o3d.io.write_line_set(str(out_dir / "trajetoria.ply"), trajectory(recon))


def show(*geometries):
    """Abre o visualizador do Open3D com as geometrias e os eixos (x vermelho, y verde, z azul)."""
    o3d.visualization.draw_geometries([*geometries, o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)])
