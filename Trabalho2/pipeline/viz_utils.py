"""
Utilitarios de visualizacao: PLY (nuvem e linhas), piramides das cameras e um
renderizador de nuvem de pontos em numpy (z-buffer por "pintor"), que funciona
sem placa de video/janela e e usado nas figuras e no video de navegacao.
"""
import numpy as np
import cv2


# ----------------------------------------------------------------------------------------
# PLY
# ----------------------------------------------------------------------------------------

def write_ply(path, xyz, rgb=None):
    """PLY binario (rapido para milhoes de pontos). rgb em RGB 0-255."""
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    n = len(xyz)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
        dtype += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    header.append("end_header")
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz.T
    if rgb is not None:
        rgb = np.asarray(rgb, np.uint8).reshape(-1, 3)
        arr["r"], arr["g"], arr["b"] = rgb.T
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(arr.tobytes())


def read_ply(path):
    """Le PLY (binario ou ASCII) de vertices com cor opcional. Retorna (xyz, rgb|None)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = [h for h in header if h.startswith("format")][0].split()[1]
        n = 0
        props = []
        in_vertex = False
        for h in header:
            if h.startswith("element"):
                in_vertex = h.split()[1] == "vertex"
                if in_vertex:
                    n = int(h.split()[2])
            elif h.startswith("property") and in_vertex:
                _, typ, name = h.split()[:3]
                props.append((name, typ))
        tmap = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1",
                "uint8": "u1", "int": "i4", "int32": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n)
            cols = {name: data[:, i] for i, (name, _) in enumerate(props)}
        else:
            end = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(name, end + tmap[t]) for name, t in props])
            arr = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            cols = {name: arr[name] for name, _ in props}
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    rgb = None
    if "red" in cols:
        rgb = np.stack([cols["red"], cols["green"], cols["blue"]], 1).astype(np.uint8)
    return xyz, rgb


def write_lines_ply(path, vertices, edges, colors=None):
    """PLY com elementos 'edge' (abre no MeshLab e no Open3D como LineSet)."""
    vertices = np.asarray(vertices, float)
    edges = np.asarray(edges, int)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\nproperty int vertex1\nproperty int vertex2\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for k, (a, b) in enumerate(edges):
            line = f"{a} {b}"
            if colors is not None:
                line += " " + " ".join(str(int(c)) for c in colors[k])
            f.write(line + "\n")


def camera_frustums(centers, Rs, K, size, depth=0.25, every=1):
    """Vertices/arestas das piramides das cameras (apice no centro, base a `depth` m)."""
    w, h = size
    Kinv = np.linalg.inv(K)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ Kinv.T
    corners = corners / corners[:, 2:3] * depth
    V, E = [], []
    for C, R in zip(centers[::every], Rs[::every]):
        base = len(V)
        V.append(C)
        for c in corners:
            V.append(R.T @ c + C)
        E += [(base, base + k) for k in range(1, 5)]
        E += [(base + k, base + 1 + k % 4) for k in range(1, 5)]
        # triangulo marcando o "para cima" da imagem
        up = R.T @ (np.array([w / 2, -0.15 * h, 1]) @ Kinv.T / 1.0 * depth) + C
        V.append(up)
        E += [(base + 1, base + 5), (base + 2, base + 5)]
    return np.array(V), np.array(E)


def trajectory_lines(centers):
    return np.asarray(centers), np.array([(i, i + 1) for i in range(len(centers) - 1)])


# ----------------------------------------------------------------------------------------
# Renderizacao
# ----------------------------------------------------------------------------------------

def look_at(eye, target, up=(0, 0, 1)):
    """cam_from_world (R, t) para uma camera em `eye` olhando para `target` (y da imagem para baixo)."""
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(fwd, [0, 1.0, 0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return R, -R @ eye


def render_points(xyz, rgb, K, R, t, size, bg=(28, 28, 32), splat_m=0.012, max_px=4, near=0.05, img=None):
    """
    Renderiza a nuvem (rgb em RGB) com algoritmo do pintor: ordena do mais longe para o
    mais perto e desenha quadrados cujo tamanho cai com a profundidade. Retorna BGR.
    """
    w, h = size
    if img is None:
        img = np.empty((h, w, 3), np.uint8)
        img[:] = bg
    Xc = xyz @ R.T + t
    z = Xc[:, 2]
    ok = z > near
    Xc, col = Xc[ok], rgb[ok]
    z = Xc[:, 2]
    u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    r = np.clip(np.round(K[0, 0] * splat_m / z), 1, max_px).astype(np.int64)
    ok = (u > -max_px) & (u < w + max_px) & (v > -max_px) & (v < h + max_px)
    u, v, z, r, col = u[ok], v[ok], z[ok], r[ok], col[ok]
    order = np.argsort(-z, kind="stable")
    u, v, r, col = u[order].astype(np.int64), v[order].astype(np.int64), r[order], col[order]
    counts = r * r
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    local = np.arange(counts.sum()) - starts
    rr = np.repeat(r, counts)
    uu = np.repeat(u - r // 2, counts) + local % rr
    vv = np.repeat(v - r // 2, counts) + local // rr
    cc = np.repeat(col, counts, axis=0)
    inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    img[vv[inside], uu[inside]] = cc[inside][:, ::-1]  # RGB -> BGR
    return img


def draw_lines(img, vertices, edges, K, R, t, color=(0, 140, 255), thickness=1, near=0.05):
    """Desenha segmentos 3D (sem teste de profundidade, ficam por cima da nuvem)."""
    Xc = np.asarray(vertices) @ R.T + t
    for a, b in edges:
        pa, pb = Xc[a].copy(), Xc[b].copy()
        if pa[2] < near and pb[2] < near:
            continue
        if pa[2] < near or pb[2] < near:  # recorta no plano proximo
            s = (near - pa[2]) / (pb[2] - pa[2])
            p = pa + s * (pb - pa)
            if pa[2] < near:
                pa = p
            else:
                pb = p
        ua = (K[0, 0] * pa[0] / pa[2] + K[0, 2], K[1, 1] * pa[1] / pa[2] + K[1, 2])
        ub = (K[0, 0] * pb[0] / pb[2] + K[0, 2], K[1, 1] * pb[1] / pb[2] + K[1, 2])
        if max(abs(x) for x in ua + ub) > 1e5:
            continue
        cv2.line(img, (int(ua[0]), int(ua[1])), (int(ub[0]), int(ub[1])), color, thickness, cv2.LINE_AA)
    return img


def intrinsics_for(size, fov_deg=60):
    w, h = size
    f = 0.5 * w / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def cutaway_mask(xyz, eye, room_center, half_extent, ceiling_z, keep_below=0.25):
    """
    Corte "casa de bonecas": esconde o teto e a parede mais proxima do observador,
    para enxergar o interior da sala de fora.
    """
    m = xyz[:, 2] < ceiling_z - 0.12
    v = np.asarray(eye[:2]) - room_center[:2]
    if np.linalg.norm(v) > 1e-6:
        v = v / np.linalg.norm(v)
        s = (xyz[:, :2] - room_center[:2]) @ v
        ext = np.abs(v[0]) * half_extent[0] + np.abs(v[1]) * half_extent[1]
        m &= ~((s > ext - 0.4) & (xyz[:, 2] > keep_below))
    return m
