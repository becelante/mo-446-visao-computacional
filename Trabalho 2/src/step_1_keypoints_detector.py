"""
Detecta keypoints e extrai descritores de imagens.

  - SIFT e ORB (OpenCV, CPU): cada imagem numa thread, porque o OpenCV solta o GIL. O SIFT na
    GPU (kornia) foi ~10x mais lento que o do OpenCV nos quadros 1080p e devolve sempre o
    numero maximo de pontos, mesmo em parede lisa.
  - SUPERPOINT, DISK e ALIKED (redes do pacote LightGlue, GPU): detectam mais pontos repetiveis
    nas regioes de pouca textura do video comprimido; sao casados com o LightGlue (Etapa 3).

Visualizacao com escala/orientacao (SIFT/ORB) ou so a posicao (redes).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

LEARNED = ("SUPERPOINT", "DISK", "ALIKED")


def create_detector(method="SIFT", max_keypoints=4096, sift_contrast=0.04):
    method = method.upper()
    if method == "SIFT":
        return cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=sift_contrast)
    elif method == "ORB":
        return cv2.ORB_create(nfeatures=max_keypoints)
    else:
        raise ValueError(f"Detector desconhecido: {method}")


def to_uint8(image):
    """Converte imagens de 16 bits (ex.: PNGs vindos do RAW) para 8 bits."""
    if image.dtype == np.uint8:
        return image
    return cv2.convertScaleAbs(image, alpha=255.0 / np.iinfo(image.dtype).max)


def to_gray8(image, equalize=False):
    """Cinza de 8 bits; `equalize` aplica CLAHE, que multiplica os keypoints em cenas escuras."""
    gray = to_uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image)
    if equalize:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def detect_features(image, method="SIFT", equalize=False, max_keypoints=4096, sift_contrast=0.04):
    """Retorna (keypoints, descriptors) para uma imagem BGR ou em tons de cinza (8 ou 16 bits)."""
    gray = to_gray8(image, equalize)
    detector = create_detector(method, max_keypoints, sift_contrast)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def load_extractor(method, max_keypoints, device):
    """Rede de deteccao do LightGlue na GPU."""
    from lightglue import ALIKED, DISK, SuperPoint
    cls = {"SUPERPOINT": SuperPoint, "DISK": DISK, "ALIKED": ALIKED}[method.upper()]
    return cls(max_num_keypoints=max_keypoints).eval().to(device)


def image_tensor(image, equalize, device):
    """Cinza (com CLAHE) em [0, 1], repetido em 3 canais: o DISK e o ALIKED esperam RGB."""
    import torch
    gray = torch.from_numpy(to_gray8(image, equalize)).to(device).float() / 255
    return gray[None].expand(3, -1, -1)


def detect_learned_all(images, method, equalize, max_keypoints, device):
    """Keypoints (cv2.KeyPoint, para o resto do pipeline) e descritores float32 de cada imagem."""
    import torch
    extractor = load_extractor(method, max_keypoints, device)
    all_kps, all_descs = [], []
    with torch.inference_mode():
        for img in images:
            feats = extractor.extract(image_tensor(img, equalize, device), resize=None)
            pts = feats["keypoints"][0].cpu().numpy()
            scores = feats["keypoint_scores"][0].cpu().numpy()
            all_kps.append([cv2.KeyPoint(float(x), float(y), 8.0, -1, float(s)) for (x, y), s in zip(pts, scores)])
            all_descs.append(feats["descriptors"][0].cpu().numpy().astype(np.float32))
    del extractor
    torch.cuda.empty_cache()
    return all_kps, all_descs


def detect_features_all(images, method="SIFT", equalize=False, max_keypoints=4096, sift_contrast=0.04,
                        device=None):
    """Aplica deteccao em uma lista de imagens. Retorna listas paralelas de kps e descritores."""
    if method.upper() in LEARNED:
        return detect_learned_all(images, method, equalize, max_keypoints, device)
    def detect(i):  # le o quadro dentro da thread: so os que estao em processamento ficam na memoria
        kps, descs = detect_features(images[i], method, equalize, max_keypoints, sift_contrast)
        # os descritores do SIFT do OpenCV sao inteiros de 0 a 255: uint8 e exato e ocupa 1/4
        if descs is not None and method.upper() == "SIFT":
            descs = descs.astype(np.uint8)
        return list(kps), descs

    # 4 threads: cada uma gasta ~0,5 GB na piramide do SIFT em 1080p, e mais que isso nao acelera
    # (medido: 48 quadros em 2,9 s com 4 threads e 3,1 s com 12, com 2,4 GB x 6,0 GB de pico)
    with ThreadPoolExecutor(min(4, os.cpu_count())) as executor:
        results = list(executor.map(detect, range(len(images))))
    return [k for k, _ in results], [d for _, d in results]


class Keypoints:
    """
    Keypoints de uma imagem guardados num array N x 6 (x, y, tamanho, angulo, resposta, oitava).
    Funciona como a lista de cv2.KeyPoint (`kps[k].pt`, `len`, iteracao), mas cria cada objeto so
    quando pedido: com ~2300 quadros x 4096 pontos, a lista de objetos ocuparia mais de 1 GB de RAM.
    """

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32).reshape(-1, 6)
        self.xy = self.arr[:, :2].astype(np.float64)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, k):
        x, y, s, a, r, o = self.arr[k]
        return cv2.KeyPoint(float(x), float(y), float(s), float(a), float(r), int(o))

    def __iter__(self):
        return (self[k] for k in range(len(self)))


def keypoints_to_array(kps):
    """KeyPoints -> array N x 6 (x, y, tamanho, angulo, resposta, oitava), para salvar em disco."""
    if isinstance(kps, Keypoints):
        return kps.arr
    return np.float64([(*k.pt, k.size, k.angle, k.response, k.octave) for k in kps]).reshape(-1, 6)


def array_to_keypoints(arr):
    return Keypoints(arr)


def draw_keypoints(image, keypoints, out_path=None):
    """Desenha keypoints com circulo (escala) e raio (orientacao)."""
    vis = cv2.drawKeypoints(
        to_uint8(image), list(keypoints), None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    if out_path:
        cv2.imwrite(out_path, vis)
    return vis
