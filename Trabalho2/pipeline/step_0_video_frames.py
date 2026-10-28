"""
Etapa 1 (a partir de video): extracao de quadros-chave.

Em vez de pegar um quadro a cada N, o quadro-chave seguinte e escolhido pela
sobreposicao com o anterior: o fluxo optico (KLT) entre os dois mede quanto a
imagem andou; quando o deslocamento mediano passa de (1 - sobreposicao) x
largura, escolhe-se o quadro mais nitido (variancia do Laplaciano) entre os
candidatos daquele trecho. Quadros borrados (movimento rapido) sao evitados.

Duas passagens no video: a primeira guarda miniaturas em cinza e a nitidez de
todos os quadros (barato); a selecao roda nas miniaturas e pode ser refeita com
outro limiar sem reler o video; a segunda passagem salva so os quadros escolhidos.
"""
import json
import os

import cv2
import numpy as np

THUMB_W = 240  # largura das miniaturas usadas no fluxo optico


def _resize_max(img, max_dim):
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    if max_dim and s < 1:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def scan_video(video_path):
    """1a passagem: miniaturas em cinza, nitidez e metadados de todos os quadros."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Nao foi possivel abrir o video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    thumbs, sharpness = [], []
    size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = frame.shape[1], frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mid = _resize_max(gray, 640)
        sharpness.append(cv2.Laplacian(mid, cv2.CV_64F).var())
        th = int(round(THUMB_W * gray.shape[0] / gray.shape[1]))
        thumbs.append(cv2.resize(gray, (THUMB_W, th), interpolation=cv2.INTER_AREA))
    cap.release()
    if not thumbs:
        raise IOError(f"Video sem quadros: {video_path}")
    return {"fps": fps, "size": size, "thumbs": thumbs, "sharpness": np.array(sharpness)}


def _median_shift(prev, cur, prev_pts):
    """Deslocamento mediano (fracao da largura) e fracao de pontos rastreados de prev para cur."""
    if prev_pts is None or len(prev_pts) < 10:
        return 1.0, 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, prev_pts, None, winSize=(21, 21), maxLevel=3)
    st = st.ravel() == 1
    if st.sum() < 10:
        return 1.0, st.mean()
    d = np.linalg.norm((nxt - prev_pts).reshape(-1, 2)[st], axis=1)
    return float(np.median(d)) / prev.shape[1], float(st.mean())


def select_keyframes(scan, target_overlap=0.8, blur_rel_thresh=0.5, max_gap_s=2.0, untracked_step_s=0.4):
    """
    Escolhe os indices dos quadros-chave. Um novo quadro-chave e criado quando o
    deslocamento desde o ultimo passa de (1 - target_overlap) da largura, quando
    menos da metade dos pontos continua rastreada, ou apos `max_gap_s` segundos
    (`untracked_step_s` se o quadro-chave nao tem textura para rastrear).
    """
    thumbs, sharp, fps = scan["thumbs"], scan["sharpness"], scan["fps"]
    n = len(thumbs)
    # nitidez relativa a vizinhanca de ~1 s (a nitidez absoluta muda com a cena)
    k = max(3, int(fps) | 1)
    local = np.array([np.median(sharp[max(0, i - k):i + k + 1]) for i in range(n)])
    rel_sharp = sharp / np.maximum(local, 1e-9)
    shift_thr = 1.0 - target_overlap
    max_gap = int(max_gap_s * fps)

    def sharpest(lo, hi):
        idx = np.arange(lo, hi + 1)
        good = idx[rel_sharp[idx] >= blur_rel_thresh]
        idx = good if len(good) else idx
        return int(idx[np.argmax(sharp[idx])])

    # quadro-chave sem textura para rastrear (parede lisa): sem medida de deslocamento,
    # usa um passo fixo no tempo em vez de criar um quadro-chave a cada quadro do video
    untracked_gap = max(1, int(untracked_step_s * fps))

    first = sharpest(0, min(n - 1, int(0.3 * fps)))
    keys, overlaps = [first], []
    key = first
    key_pts = cv2.goodFeaturesToTrack(thumbs[key], 300, 0.01, 5)
    j = key + 1
    while j < n:
        if key_pts is None or len(key_pts) < 10:
            shift, tracked = (1.0, 0.0) if j - key >= untracked_gap else (0.0, 1.0)
        else:
            shift, tracked = _median_shift(thumbs[key], thumbs[j], key_pts)
        if shift >= shift_thr or tracked < 0.5 or j - key >= max_gap:
            # candidatos: ultimo terco do trecho (ainda com sobreposicao suficiente)
            lo = max(key + 1, j - max(1, (j - key) // 3))
            new = sharpest(lo, j)
            if key_pts is not None and len(key_pts) >= 10:
                overlaps.append(1.0 - _median_shift(thumbs[key], thumbs[new], key_pts)[0])
            keys.append(new)
            key = new
            key_pts = cv2.goodFeaturesToTrack(thumbs[key], 300, 0.01, 5)
            j = key + 1
            continue
        j += 1
    # garante um quadro no final do video (fechamento do laco)
    last = sharpest(max(keys[-1] + 1, n - 1 - int(0.3 * fps)), n - 1) if keys[-1] < n - 1 else None
    if last is not None and last - keys[-1] >= max(1, int(0.1 * fps)):
        keys.append(last)
    return keys, overlaps


def extract_keyframes(video_path, out_dir, target_overlap=0.8, min_frames=100, max_frames=300,
                      max_dim=1920, blur_rel_thresh=0.5, max_gap_s=2.0):
    """
    Extrai os quadros-chave de `video_path` para `out_dir` (frame_00000.jpg, ...),
    ajustando a sobreposicao para o numero de quadros ficar em [min_frames, max_frames].
    Retorna o dict salvo em frames.json (indices, tempos, sobreposicao estimada).
    """
    os.makedirs(out_dir, exist_ok=True)
    scan = scan_video(video_path)
    overlap = target_overlap
    for _ in range(12):
        keys, overlaps = select_keyframes(scan, overlap, blur_rel_thresh, max_gap_s)
        if len(keys) < min_frames and overlap < 0.97:
            overlap = min(0.97, overlap + (1 - overlap) * 0.3)   # menos deslocamento -> mais quadros
        elif len(keys) > max_frames and overlap > 0.4:
            overlap = max(0.4, overlap - (1 - overlap) * 0.3)
        else:
            break

    keyset = set(keys)
    cap = cv2.VideoCapture(video_path)
    i, names = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in keyset:
            name = f"frame_{len(names):05d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), _resize_max(frame, max_dim),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append(name)
        i += 1
    cap.release()

    info = {
        "video": os.path.abspath(video_path),
        "fps": scan["fps"],
        "video_size": scan["size"],
        "n_video_frames": len(scan["thumbs"]),
        "duration_s": len(scan["thumbs"]) / scan["fps"],
        "target_overlap_used": overlap,
        "n_keyframes": len(names),
        "keyframe_indices": [int(k) for k in keys],
        "timestamps_s": [k / scan["fps"] for k in keys],
        "mean_overlap_est": float(np.mean(overlaps)) if overlaps else None,
        "min_overlap_est": float(np.min(overlaps)) if overlaps else None,
        "names": names,
    }
    with open(os.path.join(out_dir, "frames.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def list_images(image_dir):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    return sorted(f for f in os.listdir(image_dir) if f.lower().endswith(exts))
