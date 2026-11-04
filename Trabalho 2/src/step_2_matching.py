"""
Emparelhamento de caracteristicas na GPU e escolha dos pares de imagens.

  - SIFT/ORB: os 2 vizinhos mais proximos exatos de cada descritor (distancias por produto de
    matrizes na GPU: L2 para SIFT, Hamming pelos bits para ORB) e ratio test de Lowe, com
    verificacao cruzada opcional. Substitui o FLANN, que e aproximado e roda na CPU.
  - SUPERPOINT/DISK/ALIKED: LightGlue (atencao entre os pontos das duas imagens), na GPU.
  - pares: exaustivo, sequencial (janela de quadros vizinhos) ou sequencial + deteccao de laco
    por bag of visual words (k-means dos descritores na GPU, histogramas tf-idf).
"""
import numpy as np
import cv2
import torch

from .step_1_keypoints_detector import LEARNED


def descriptor_tensor(descs, method, device):
    """Descritores na GPU; os binarios do ORB viram 256 bits em float (para Hamming por produto)."""
    if method.upper() == "ORB":
        return torch.from_numpy(np.unpackbits(descs, axis=1)).to(device).float()
    return torch.from_numpy(np.ascontiguousarray(descs, np.float32)).to(device)


def nearest_two(d1, d2, method, chunk=8192):
    """
    Para cada linha de d1: indice do vizinho mais proximo em d2 e as distancias ao 1o e ao 2o.
    L2: |a|^2 + |b|^2 - 2 a.b; Hamming entre bits 0/1: |a| + |b| - 2 a.b.
    """
    binary = method.upper() == "ORB"
    n2 = d2.sum(1) if binary else (d2 * d2).sum(1)
    idx, best, second = [], [], []
    for s in range(0, len(d1), chunk):
        a = d1[s:s + chunk]
        n1 = a.sum(1) if binary else (a * a).sum(1)
        dist = (n1[:, None] + n2[None] - 2 * a @ d2.T).clamp_min(0)
        if not binary:
            dist = dist.sqrt()
        vals, ids = dist.topk(2, dim=1, largest=False)
        idx.append(ids[:, 0])
        best.append(vals[:, 0])
        second.append(vals[:, 1])
    return torch.cat(idx), torch.cat(best), torch.cat(second)


def ratio_matches(d1, d2, method, ratio_thresh=0.75, cross_check=False):
    """Pares (i1, i2, distancia) aprovados no ratio test, como arrays numpy."""
    if len(d1) < 2 or len(d2) < 2:
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    idx, best, second = nearest_two(d1, d2, method)
    keep = best < ratio_thresh * second
    if cross_check:
        back = nearest_two(d2, d1, method)[0]
        keep &= back[idx] == torch.arange(len(d1), device=d1.device)
    q = torch.nonzero(keep).ravel()
    return q.cpu().numpy(), idx[q].cpu().numpy(), best[q].cpu().numpy()


def to_dmatches(q, t, dist):
    """Lista de cv2.DMatch ordenada por distancia (melhores primeiro)."""
    order = np.argsort(dist, kind="stable")
    return [cv2.DMatch(int(q[k]), int(t[k]), float(dist[k])) for k in order]


def knn_match(descs1, descs2, method="SIFT", device="cuda"):
    """Os 2 vizinhos mais proximos (matches brutos) de cada descritor de descs1 em descs2."""
    if descs1 is None or descs2 is None or len(descs1) < 2 or len(descs2) < 2:
        return []
    d1, d2 = descriptor_tensor(descs1, method, device), descriptor_tensor(descs2, method, device)
    n2 = d2.sum(1) if method.upper() == "ORB" else (d2 * d2).sum(1)
    n1 = d1.sum(1) if method.upper() == "ORB" else (d1 * d1).sum(1)
    dist = (n1[:, None] + n2[None] - 2 * d1 @ d2.T).clamp_min(0)
    if method.upper() != "ORB":
        dist = dist.sqrt()
    vals, ids = (v.cpu().numpy() for v in dist.topk(2, dim=1, largest=False))
    return [[cv2.DMatch(q, int(ids[q, 0]), float(vals[q, 0])), cv2.DMatch(q, int(ids[q, 1]), float(vals[q, 1]))]
            for q in range(len(ids))]


def match_features(descs1, descs2, method="SIFT", ratio_thresh=0.75, cross_check=False, device="cuda"):
    """
    Emparelha descritores entre duas imagens usando k-NN (k=2) na GPU e aplica o
    ratio test de Lowe. Retorna a lista de cv2.DMatch aprovados,
    ordenada por distância (melhores primeiro).
    """
    if descs1 is None or descs2 is None:
        return []
    d1, d2 = descriptor_tensor(descs1, method, device), descriptor_tensor(descs2, method, device)
    return to_dmatches(*ratio_matches(d1, d2, method, ratio_thresh, cross_check))


class Matcher:
    """
    Casa as imagens `i` e `j` por indice: `matcher(i, j)` -> lista de cv2.DMatch. Os
    descritores ficam na GPU (todos de uma vez, se couberem em `gpu_budget` bytes).
    """

    def __init__(self, all_kps, all_descs, method, image_size, ratio_thresh=0.75, cross_check=False,
                 device="cuda", gpu_budget=1.5e9):
        self.method, self.ratio, self.cross, self.device = method.upper(), ratio_thresh, cross_check, device
        self.all_kps, self.all_descs = all_kps, all_descs
        self.image_size = image_size  # (largura, altura)
        nbytes = sum(0 if d is None else d.shape[0] * (256 if self.method == "ORB" else d.shape[1]) * 4
                     for d in all_descs)
        self.cache = {} if nbytes <= gpu_budget else None
        if self.method in LEARNED:
            from lightglue import LightGlue
            self.lightglue = LightGlue(features=self.method.lower()).eval().to(device)

    def _features(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        if self.method in LEARNED:
            pts = torch.tensor([k.pt for k in self.all_kps[i]], dtype=torch.float32, device=self.device)
            feats = {"keypoints": pts[None],
                     "descriptors": torch.from_numpy(self.all_descs[i]).to(self.device)[None],
                     "image_size": torch.tensor([self.image_size], dtype=torch.float32, device=self.device)}
        else:
            feats = None if self.all_descs[i] is None else descriptor_tensor(self.all_descs[i], self.method, self.device)
        if self.cache is not None:
            self.cache[i] = feats
        return feats

    @torch.inference_mode()
    def __call__(self, i, j):
        f1, f2 = self._features(i), self._features(j)
        if self.method in LEARNED:
            if f1["keypoints"].shape[1] < 2 or f2["keypoints"].shape[1] < 2:
                return []
            out = self.lightglue({"image0": f1, "image1": f2})
            m = out["matches"][0].cpu().numpy()
            scores = out["scores"][0].cpu().numpy()
            return to_dmatches(m[:, 0], m[:, 1], 1 - scores)
        if f1 is None or f2 is None:
            return []
        return to_dmatches(*ratio_matches(f1, f2, self.method, self.ratio, self.cross))


def kmeans(x, k, iters=15, seed=0):
    """K-means na GPU (atribuicao pelo produto de matrizes). Retorna os centroides k x D."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        labels = assign_words(x, centers)
        sums = torch.zeros_like(centers).index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).clamp_min(1)[:, None]
        centers = torch.where(counts > 1, sums / counts, centers)
    return centers


def assign_words(x, centers, chunk=65536):
    cn = (centers * centers).sum(1)
    return torch.cat([((cn[None] - 2 * x[s:s + chunk] @ centers.T).argmin(1)) for s in range(0, len(x), chunk)])


def retrieval_similarity(all_descs, method, device, n_words=1024, sample=200_000, seed=0):
    """
    Similaridade N x N entre imagens por bag of visual words: vocabulario por k-means de uma
    amostra dos descritores, histograma de palavras por imagem com peso tf-idf (palavras que
    aparecem em quase todas as imagens, como as de parede lisa, pesam pouco) e cosseno.
    """
    # amostra de mesmo tamanho por imagem para o k-means; os histogramas sao feitos imagem a imagem,
    # para nao levar todos os descritores para a GPU de uma vez
    rng = np.random.default_rng(seed)
    per_image = max(1, sample // max(1, len(all_descs)))
    pool = np.concatenate([d[rng.choice(len(d), min(per_image, len(d)), replace=False)]
                           for d in all_descs if d is not None and len(d)])
    centers = kmeans(descriptor_tensor(pool, method, device), n_words, seed=seed)
    hist = torch.zeros(len(all_descs), len(centers), device=device)
    for n, d in enumerate(all_descs):
        if d is not None and len(d):
            hist[n] = torch.bincount(assign_words(descriptor_tensor(d, method, device), centers),
                                     minlength=len(centers)).float()
    idf = torch.log(len(all_descs) / (hist > 0).sum(0).clamp_min(1).float())
    tfidf = hist / hist.sum(1, keepdim=True).clamp_min(1) * idf
    tfidf = tfidf / tfidf.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (tfidf @ tfidf.T).cpu().numpy()


def select_pairs(n, strategy="exhaustive", all_descs=None, method="SIFT", window=10, top_k=5, device="cuda"):
    """
    Pares (i, j), i < j, a casar:
      - "exhaustive": todos, n (n - 1) / 2;
      - "sequential": cada quadro com os `window` seguintes (o video garante a sobreposicao);
      - "sequential_loop": sequencial + os `top_k` quadros mais parecidos (bag of words) fora da
        janela, que achem o reencontro com lugares ja vistos, como o fim do laco com o inicio.
    """
    if strategy == "exhaustive":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs = {(i, j) for i in range(n) for j in range(i + 1, min(n, i + window + 1))}
    if strategy == "sequential_loop":
        sim = retrieval_similarity(all_descs, method, device)
        for i in range(n):
            far = np.abs(np.arange(n) - i) > window
            for j in np.flatnonzero(far)[np.argsort(-sim[i, far])[:top_k]]:
                pairs.add((min(i, j), max(i, j)))
    elif strategy != "sequential":
        raise ValueError(f"Estrategia de pares desconhecida: {strategy}")
    return sorted(pairs)


def draw_matches_before_after(img1, kps1, img2, kps2, all_matches, good, inlier_mask,
                              out_prefix, max_draw=150, seed=0):
    """"Antes": amostra de todos os matches brutos; "Depois": amostra
    dos aprovados no ratio test, em azul os inliers do RANSAC e em vermelho os outliers."""
    rng = np.random.default_rng(seed)
    kps1, kps2 = list(kps1), list(kps2)  # o OpenCV quer listas de cv2.KeyPoint
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    idx = rng.choice(len(all_matches), min(max_draw, len(all_matches)), replace=False) if all_matches else []
    before = cv2.drawMatches(img1, kps1, img2, kps2, [all_matches[i] for i in idx], None,
                             matchColor=(0, 100, 200), flags=flags)
    cv2.imwrite(f"{out_prefix}_antes.jpg", before)

    mask = inlier_mask if inlier_mask is not None else np.ones(len(good), bool)
    idx = rng.choice(len(good), min(max_draw, len(good)), replace=False) if good else []
    after = cv2.drawMatches(img1, kps1, img2, kps2, [good[i] for i in idx if mask[i]], None,
                            matchColor=(255, 100, 0), flags=flags)
    after = cv2.drawMatches(img1, kps1, img2, kps2, [good[i] for i in idx if not mask[i]], after,
                            matchColor=(0, 0, 255), flags=flags | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    cv2.imwrite(f"{out_prefix}_depois.jpg", after)
