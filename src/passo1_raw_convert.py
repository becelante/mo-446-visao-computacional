"""
Conversao de imagens RAW (.ARW, formato Sony) para PNG, para uso no pipeline
de panoramas (as demais etapas trabalham com PNG/JPG via OpenCV).

Todas as fotos de um conjunto sao reveladas com o mesmo balanco de branco
(mediana do "as shot"; com AWB cada foto vem com um diferente) e o mesmo
brilho, para nao criar degraus de cor e exposicao entre vizinhas.
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rawpy
import cv2

SRGB_GAMMA = (2.4, 12.92)


def common_white_balance(paths):
    """Mediana, por canal, do balanco de branco "as shot" de todas as fotos."""
    wbs = []
    for path in paths:
        with rawpy.imread(path) as raw:
            wbs.append(raw.camera_whitebalance)
    wb = np.median(np.array(wbs, dtype=np.float64), axis=0)
    if wb[3] == 0:
        wb[3] = wb[1]
    return [float(v) for v in wb]


def _bright_percentile(path, user_wb):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(user_wb=user_wb, no_auto_bright=True, gamma=(1, 1),
                              output_bps=16, half_size=True)
    lum = rgb.astype(np.float32) @ np.float32([0.2126, 0.7152, 0.0722]) / 65535.0
    return float(np.percentile(lum, 99))


def estimate_common_brightness(paths, user_wb, executor):
    """Brilho unico: leva a mediana (entre as fotos) do percentil 99 da luminancia
    a 1.0, o criterio do auto-brilho do LibRaw aplicado ao conjunto inteiro."""
    p99 = list(executor.map(_bright_percentile, paths, [user_wb] * len(paths)))
    return float(np.clip(1.0 / max(np.median(p99), 1e-6), 1.0, 64.0))


def _develop_one(path, out_path, user_wb, bright, max_dim, bits):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT,
                              user_wb=user_wb, no_auto_bright=True, bright=bright,
                              gamma=SRGB_GAMMA, output_bps=16)
    h, w = rgb.shape[:2]
    if max_dim and max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    if bits == 8:
        rgb = np.round(rgb / 257.0).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return rgb.shape[1], rgb.shape[0]


def convert_arw_to_png(input_dir, output_dir, max_dim=1600, bits=8, workers=None):
    arw_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".arw"))
    if not arw_files:
        print(f"Nenhum arquivo .ARW encontrado em {input_dir}")
        return []
    os.makedirs(output_dir, exist_ok=True)
    for old in os.listdir(output_dir):
        if old.lower().endswith(".png"):
            os.remove(os.path.join(output_dir, old))

    paths = [os.path.join(input_dir, f) for f in arw_files]
    out_paths = [os.path.join(output_dir, os.path.splitext(f)[0] + ".png") for f in arw_files]

    user_wb = common_white_balance(paths)
    print(f"Balanco de branco comum (R, G, B, G2): {[round(v, 1) for v in user_wb]}")
    # spawn: o LibRaw usa OpenMP, que pode travar em processos criados com fork
    with ProcessPoolExecutor(workers or min(8, os.cpu_count() or 1), mp_context=mp.get_context("spawn")) as ex:
        bright = estimate_common_brightness(paths, user_wb, ex)
        print(f"Fator de brilho comum: {bright:.3f}")
        n = len(paths)
        sizes = ex.map(_develop_one, paths, out_paths, [user_wb] * n, [bright] * n, [max_dim] * n, [bits] * n)
        for filename, out_path, (w, h) in zip(arw_files, out_paths, sizes):
            print(f"Convertido: {filename} -> {os.path.basename(out_path)}  ({w}x{h}, {bits} bits)")
    return out_paths
