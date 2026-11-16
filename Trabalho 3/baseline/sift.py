"""Baseline classico: autocorrespondencia de pontos-chave SIFT dentro da propria imagem.

Etapas (uma funcao para cada):
  1. self_matches   pontos SIFT da imagem que casam com outro ponto da mesma imagem, inclusive
                    com a imagem espelhada (o SIFT nao e invariante a espelho). Comparacao na GPU.
  2. find_groups    agrupa os pares que seguem a mesma transformacao (origem -> destino) com RANSAC.
                    Cada grupo e uma copia; a transformacao inclui rotacao, escala e espelhamento.
  3. group_mask     mascara densa: aplica a transformacao a imagem e marca os pixels em que a imagem
                    e sua versao transformada coincidem (correlacao local alta), perto dos pontos casados.
  4. merge          junta mascaras que se sobrepoem: uma origem colada em varios destinos vira uma
                    instancia so, como nas mascaras verdadeiras (analise exploratoria, secao 3.1).
  5. predict        aplica os limiares e devolve as mascaras na resolucao original
                    (lista vazia = imagem prevista como autentica).
"""
import cv2
import numpy as np
import torch


def prepare(img, max_side):
    """Tons de cinza, reduzida para no maximo max_side px no lado maior. Devolve (gray, escala)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, max_side / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def sift_features(gray, n_features):
    kps, des = cv2.SIFT_create(nfeatures=n_features).detectAndCompute(gray, None)
    if des is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.array([k.pt for k in kps], np.float32), des


# ---------------------------------------------------------------- 1. pares de pontos casados

def self_matches(gray, n_features, min_dist, device):
    """Pares (P[i], Q[i]) de pontos com descritores parecidos e distantes pelo menos min_dist px.

    Candidatos: os pontos da imagem e os da imagem espelhada (com as coordenadas trazidas de volta).
    Para cada ponto, d1 e d2 sao as distancias de descritor ao 1o e ao 2o candidato mais proximos,
    ignorando os que estao a menos de min_dist px (o proprio ponto e a vizinhanca dele).
    razao = d1 / d2 (teste de Lowe): perto de 0 = casamento unico e confiavel; perto de 1 = ambiguo.
    espelhado[i] = True se Q[i] veio da imagem espelhada (copia colada espelhada).
    """
    pts, des = sift_features(gray, n_features)
    pts_m, des_m = sift_features(np.ascontiguousarray(gray[:, ::-1]), n_features)
    pts_m[:, 0] = gray.shape[1] - 1 - pts_m[:, 0]
    if len(pts) < 3:
        empty = np.zeros((0, 2), np.float32)
        return empty, empty, np.zeros(0, np.float32), np.zeros(0, bool)

    cand_pts, cand_des = np.concatenate([pts, pts_m]), np.concatenate([des, des_m])
    t = lambda x: torch.from_numpy(x).float().to(device)
    dist = torch.cdist(t(des), t(cand_des))                         # descritores: (n, 2n)
    dist[torch.cdist(t(pts), t(cand_pts)) < min_dist] = float("inf")
    d, idx = dist.topk(2, largest=False)
    ok = torch.isfinite(d[:, 1]).cpu().numpy()
    ratio = (d[:, 0] / d[:, 1].clamp_min(1e-6)).cpu().numpy()
    best = idx[:, 0].cpu().numpy()[ok]
    P, Q, ratio, mirrored = pts[ok], cand_pts[best], ratio[ok], best >= len(pts)

    # p->q e q->p sao o mesmo par: fica so uma vez
    key = np.round(np.concatenate([np.minimum(P, Q), np.maximum(P, Q)], 1)).astype(int)
    _, first = np.unique(key, axis=0, return_index=True)
    return P[first], Q[first], ratio[first], mirrored[first]


# ---------------------------------------------------------------- 2. grupos com a mesma transformacao

def plausible(T, max_scale=3.0):
    """Descarta transformacoes degeneradas: escala absurda ou cisalhamento forte."""
    s = np.linalg.svd(T[:, :2], compute_uv=False)
    return 1 / max_scale < s[1] and s[0] < max_scale and s[0] / s[1] < 2


def find_groups(P, Q, min_inliers, ransac_px, max_groups=8):
    """RANSAC sequencial: acha a transformacao afim que explica mais pares, separa esses pares e repete.

    Cada par entra nas duas direcoes (P->Q e Q->P), porque nao sabemos qual ponta e a origem.
    Devolve lista de (T 2x3, pontos de origem, pontos de destino) dos pares que seguem T.
    """
    n = len(P)
    src, dst, pair = np.concatenate([P, Q]), np.concatenate([Q, P]), np.concatenate([np.arange(n)] * 2)
    alive = np.ones(n, bool)
    groups = []
    while alive.sum() >= min_inliers and len(groups) < max_groups:
        sel = alive[pair]
        T, inl = cv2.estimateAffine2D(src[sel], dst[sel], method=cv2.RANSAC, ransacReprojThreshold=ransac_px,
                                      maxIters=2000, confidence=0.995)
        if T is None:
            break
        inl = inl.ravel().astype(bool)
        members = np.unique(pair[sel][inl])
        if len(members) < min_inliers:
            break
        alive[members] = False
        if plausible(T):
            groups.append((T, src[sel][inl], dst[sel][inl]))
    return groups


# ---------------------------------------------------------------- 3. mascara densa de um grupo

def local_zncc(a, b, win):
    """Correlacao normalizada entre a e b numa janela win x win em volta de cada pixel."""
    blur = lambda x: cv2.blur(x, (win, win))
    ma, mb = blur(a), blur(b)
    va, vb = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2
    zncc = (blur(a * b) - ma * mb) / np.sqrt(np.maximum(va * vb, 1e-6))
    return zncc, np.minimum(va, vb)


def similarity(gray, T, win=9, min_var=4.0):
    """Compara a imagem com ela mesma levada por T: moved(x) = gray(T x).
    Devolve (moved, zncc, valido): zncc so vale onde ha textura e T x cai dentro da imagem."""
    h, w = gray.shape
    g = gray.astype(np.float32)
    moved = cv2.warpAffine(g, T, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    inside = cv2.warpAffine(np.ones_like(g), T, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    zncc, var = local_zncc(g, moved, win)
    return moved, zncc, (var > min_var) & inside


def group_mask(gray, T, src_pts, corr_thr):
    """Origem + destino de um grupo como uma mascara so.

    origem = pixels x em que gray(x) se parece com gray(T x), com textura suficiente para a
             comparacao valer, e na mesma mancha de algum ponto casado
    destino = a origem levada por T
    """
    h, w = gray.shape
    _, zncc, valid = similarity(gray, T)
    similar = ((zncc > corr_thr) & valid).astype(np.uint8)
    similar = cv2.morphologyEx(similar, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, labels = cv2.connectedComponents(similar)
    xy = np.clip(np.round(src_pts).astype(int), 0, [w - 1, h - 1])
    hit = np.unique(labels[xy[:, 1], xy[:, 0]])
    source = np.isin(labels, hit[hit > 0])
    if not source.any():                                   # sem textura: usa o casco dos pontos casados
        source = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(source, cv2.convexHull(np.round(src_pts).astype(np.int32)), 1)
        source = source > 0
    target = cv2.warpAffine(source.astype(np.uint8), T, (w, h), flags=cv2.INTER_NEAREST) > 0
    return fill_holes(source | target)


def fill_holes(mask):
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(m, contours, -1, 1, cv2.FILLED) > 0


# ---------------------------------------------------------------- 4. juntar instancias sobrepostas

def merge(masks, min_overlap=0.2):
    """Junta mascaras cuja intersecao cobre mais de min_overlap da menor delas."""
    masks = list(masks)
    merged = True
    while merged:
        merged = False
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                inter = (masks[i] & masks[j]).sum()
                if inter > min_overlap * min(masks[i].sum(), masks[j].sum()):
                    masks[i] = masks[i] | masks.pop(j)
                    merged = True
                    break
            if merged:
                break
    return masks


# ---------------------------------------------------------------- 5. predicao de uma imagem

def predict(gray, P, Q, ratio, orig_shape, params):
    """Mascaras previstas (resolucao original) a partir dos pares ja extraidos. [] = autentica."""
    keep = ratio < params["ratio"]
    groups = find_groups(P[keep], Q[keep], params["min_inliers"], params["ransac_px"])
    masks = [group_mask(gray, T, src, params["corr_thr"]) for T, src, _ in groups]
    masks = [m for m in merge(masks) if m.mean() >= params["min_area"]]
    h, w = orig_shape
    return [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]
