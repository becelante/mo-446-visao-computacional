"""
Gera um video sintetico de uma sala com dimensoes conhecidas para testar o
pipeline de ponta a ponta (sem precisar de captura real).

Sala: caixa W x L x H (padrao 5,00 x 4,00 x 2,70 m) com paredes, chao e teto
texturizados e dois marcadores ArUco de 20 cm nas paredes. A camera percorre
uma elipse olhando para as paredes e termina onde comecou (laco fechado).

Uso:
    python tests/synthetic_room.py --out data/sala_sintetica.mp4
"""
import argparse
import json
import os

import cv2
import numpy as np

PX_PER_M = 400  # resolucao das texturas (2,5 mm por pixel)


def make_texture(width_m, height_m, rng, base_color):
    """Textura procedural: ruido em varias escalas + formas ("posteres", "fitas")."""
    w, h = int(width_m * PX_PER_M), int(height_m * PX_PER_M)
    tex = np.zeros((h, w, 3), np.float32)
    for cell, amp in [(400, 50), (100, 40), (25, 35), (6, 30), (2, 20)]:
        small = rng.random((max(2, h // cell), max(2, w // cell), 3)).astype(np.float32)
        tex += amp * (cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) - 0.5)
    tex += np.array(base_color, np.float32)
    for _ in range(int(width_m * height_m * 30)):
        color = rng.integers(20, 235, 3).tolist()
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(PX_PER_M // 30, PX_PER_M // 6))
        if rng.random() < 0.5:
            cv2.rectangle(tex, (x, y), (x + s, y + int(s * rng.uniform(0.3, 1.5))), color, -1)
        else:
            cv2.circle(tex, (x, y), s // 2, color, -1)
    return np.clip(tex, 0, 255).astype(np.uint8)


def paste_marker(tex, marker_id, u_m, v_m, size_m, dictionary):
    """Cola um marcador ArUco (com borda branca) com o centro em (u_m, v_m) na textura."""
    s = int(size_m * PX_PER_M)
    border = s // 4
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, s)
    marker = cv2.copyMakeBorder(marker, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    cx, cy = int(u_m * PX_PER_M), int(v_m * PX_PER_M)
    half = marker.shape[0] // 2
    tex[cy - half:cy - half + marker.shape[0], cx - half:cx - half + marker.shape[1]] = marker[..., None]


class Room:
    """Superficies planas da sala: (origem, eixo_u, eixo_v, largura_u, altura_v, textura)."""

    def __init__(self, W, L, H, seed=0, marker_size=0.20):
        rng = np.random.default_rng(seed)
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        e = np.eye(3)
        self.surfaces = []

        def add(origin, eu, ev, wu, hv, base):
            tex = make_texture(wu, hv, rng, base)
            self.surfaces.append([np.array(origin, float), eu, ev, wu, hv, tex])
            return tex

        # paredes (v cresce para baixo na textura -> eixo v = -z, origem no topo)
        # eixo u escolhido para a textura ser lida sem espelhar, vista de dentro da sala
        t0 = add([W, 0, H], -e[0], -e[2], W, H, (170, 160, 150))     # y = 0
        add([0, L, H], e[0], -e[2], W, H, (150, 170, 160))          # y = L
        t2 = add([0, 0, H], e[1], -e[2], L, H, (160, 150, 175))     # x = 0
        add([W, L, H], -e[1], -e[2], L, H, (140, 150, 160))         # x = W
        add([0, 0, 0], e[0], e[1], W, L, (110, 100, 90))            # chao
        add([0, L, H], e[0], -e[1], W, L, (215, 215, 210))          # teto
        paste_marker(t0, 0, 1.2, H - 1.40, marker_size, d)
        paste_marker(t0, 1, 3.6, H - 1.10, marker_size, d)
        paste_marker(t2, 2, 2.0, H - 1.30, marker_size, d)

    def render(self, K, R, t, size):
        """Ray casting: para cada pixel, a superficie mais proxima e a cor da textura."""
        w, h = size
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        rays = rays_cam @ R  # direcoes no mundo (R^T aplicado a cada raio)
        C = -R.T @ t
        best_t = np.full((h, w), np.inf)
        img = np.zeros((h, w, 3), np.uint8)
        for origin, eu, ev, wu, hv, tex in self.surfaces:
            n = np.cross(eu, ev)
            denom = rays @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = ((origin - C) @ n) / denom
            P = C + rays * tt[..., None]
            su, sv = (P - origin) @ eu, (P - origin) @ ev
            ok = (tt > 1e-6) & (tt < best_t) & (su >= 0) & (su <= wu) & (sv >= 0) & (sv <= hv)
            if not ok.any():
                continue
            mx = (su * PX_PER_M).astype(np.float32)
            my = (sv * PX_PER_M).astype(np.float32)
            sampled = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            img[ok] = sampled[ok]
            best_t[ok] = tt[ok]
        return img


def look_rotation(yaw, pitch):
    """cam_from_world para camera olhando na direcao (yaw, pitch), com z do mundo para cima."""
    fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sala_sintetica.mp4")
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--frames", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--room", default="5.0x4.0x2.7")
    args = ap.parse_args()

    W, L, H = map(float, args.room.split("x"))
    w, h = map(int, args.size.split("x"))
    f = 0.72 * w  # ~69 graus de FOV horizontal, tipico de celular
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    room = Room(W, L, H)
    rng = np.random.default_rng(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    centers = []
    for k in range(args.frames):
        s = 2 * np.pi * k / (args.frames - 1)  # ultimo quadro volta ao inicio
        # caminha rente as paredes (superelipse a ~1 m delas), olhando para a parede em diagonal
        cs, sn = np.cos(s), np.sin(s)
        C = np.array([W / 2 + (W / 2 - 1.0) * np.sign(cs) * abs(cs) ** 0.5,
                      L / 2 + (L / 2 - 1.0) * np.sign(sn) * abs(sn) ** 0.5,
                      1.50 + 0.03 * np.sin(5 * s)])
        yaw = s + 0.55 + 0.25 * np.sin(2 * s)
        pitch = np.deg2rad(-8 + 24 * np.sin(3 * s))
        R = look_rotation(yaw, pitch)
        t = -R @ C
        img = room.render(K, R, t, (w, h))
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        writer.write(img)
        centers.append(C.tolist())
        if k % 100 == 0:
            print(f"quadro {k}/{args.frames}")
    writer.release()
    gt = {"room_m": [W, L, H], "K": K.tolist(), "centers": centers, "marker_size_m": 0.20}
    with open(os.path.splitext(args.out)[0] + "_gt.json", "w") as fp:
        json.dump(gt, fp)
    print("video salvo em", args.out)


if __name__ == "__main__":
    main()
