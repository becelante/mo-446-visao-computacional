"""
Renderizacoes da cena: vista geral (nuvem + poses + trajetoria) e o video de
navegacao de 30-60 s pedido nos entregaveis (sobrevoo em volta da sala com
corte do teto/parede da frente + caminhada pela trajetoria real da camera).
"""
import cv2
import numpy as np

from .viz_utils import (camera_frustums, cutaway_mask, draw_lines, intrinsics_for, look_at,
                        render_points, trajectory_lines)


def _room_geom(room, xyz):
    (x0, x1), (y0, y1) = room["bounds"][0], room["bounds"][1]
    top = room["ceiling_z"] or float(np.percentile(xyz[:, 2], 98))
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2, top * 0.35])
    half = np.array([(x1 - x0) / 2, (y1 - y0) / 2])
    return center, half, top


def _overlays(img, sc, K, R, t, frustum_depth, every):
    V, E = camera_frustums(sc["centers"], sc["R"], sc["K"], (sc["camera"].width, sc["camera"].height),
                           depth=frustum_depth, every=every)
    draw_lines(img, V, E, K, R, t, color=(0, 150, 255), thickness=1)
    V, E = trajectory_lines(sc["centers"])
    draw_lines(img, V, E, K, R, t, color=(60, 60, 255), thickness=2)
    return img


def _label(img, text, y=30):
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def render_overview(xyz, rgb, sc, room, out_path, size=(1600, 1000)):
    """Vista obliqua de cima (sem teto e sem a parede da frente) com cameras e trajetoria."""
    center, half, top = _room_geom(room, xyz)
    r = 1.25 * np.linalg.norm(half) + 1.0
    eye = center + np.array([r * np.cos(-2.2), r * np.sin(-2.2), top * 1.9 + 1.5])
    m = cutaway_mask(xyz, eye, center, half, top)
    K = intrinsics_for(size, 55)
    R, t = look_at(eye, center)
    img = render_points(xyz[m], rgb[m], K, R, t, size, splat_m=0.015, bg=(245, 245, 245))
    every = max(1, len(sc["centers"]) // 60)
    _overlays(img, sc, K, R, t, 0.18, every)
    _label(img, "Nuvem + poses das cameras (laranja) + trajetoria (vermelho)")
    cv2.imwrite(out_path, img)


def flythrough_video(xyz, rgb, sc, room, out_path, seconds=45, fps=30, size=(1280, 720),
                     title="Reconstrucao 3D", orbit_frac=0.6, log=print):
    """
    Video de navegacao: (1) orbita completa em volta da sala, vista de cima e com
    corte; (2) percurso pela trajetoria real (suavizada), em primeira pessoa,
    mostrando as cameras. Renderizado em numpy (nao precisa de janela/GPU).
    """
    center, half, top = _room_geom(room, xyz)
    K = intrinsics_for(size, 62)
    n_total = int(seconds * fps)
    n_orbit = int(n_total * orbit_frac)
    n_walk = n_total - n_orbit
    every = max(1, len(sc["centers"]) // 60)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    r = 1.2 * np.linalg.norm(half) + 1.0

    for k in range(n_orbit):
        a = -np.pi / 2 + 2 * np.pi * k / n_orbit
        elev = top * 1.6 + 1.0 + 0.6 * np.sin(2 * np.pi * k / n_orbit)
        eye = center + np.array([r * np.cos(a), r * np.sin(a), elev])
        m = cutaway_mask(xyz, eye, center, half, top)
        R, t = look_at(eye, center)
        img = render_points(xyz[m], rgb[m], K, R, t, size)
        _overlays(img, sc, K, R, t, 0.15, every)
        _label(img, title)
        _label(img, f"{len(xyz)} pontos | {len(sc['centers'])} cameras", 58)
        writer.write(img)
        if k % 200 == 0:
            log(f"  video: quadro {k}/{n_total}")

    # caminhada: suaviza a trajetoria e olha na direcao media das cameras proximas
    C = sc["centers"]
    fw = sc["R"][:, 2, :]
    wlen = max(1, len(C) // 25)
    ker = np.ones(2 * wlen + 1) / (2 * wlen + 1)
    pad = lambda a: np.concatenate([a[-wlen:], a, a[:wlen]])  # laco fechado: borda circular
    Cs = np.stack([np.convolve(pad(C[:, i]), ker, "valid") for i in range(3)], 1)
    Fs = np.stack([np.convolve(pad(fw[:, i]), ker, "valid") for i in range(3)], 1)
    no_ceiling = xyz[:, 2] < top - 0.12
    for k in range(n_walk):
        s = (len(Cs) - 1) * k / max(1, n_walk - 1)
        i0 = int(np.floor(s))
        i1 = min(i0 + 1, len(Cs) - 1)
        a = s - i0
        pos = (1 - a) * Cs[i0] + a * Cs[i1]
        d = (1 - a) * Fs[i0] + a * Fs[i1]
        d[2] *= 0.5
        d /= np.linalg.norm(d)
        eye = pos - 0.6 * d + np.array([0, 0, 0.25])  # um pouco atras e acima da camera
        R, t = look_at(eye, eye + d)
        img = render_points(xyz[no_ceiling], rgb[no_ceiling], K, R, t, size)
        _overlays(img, sc, K, R, t, 0.15, every)
        _label(img, title + " - percurso da camera")
        writer.write(img)
        if (n_orbit + k) % 200 == 0:
            log(f"  video: quadro {n_orbit + k}/{n_total}")
    writer.release()
