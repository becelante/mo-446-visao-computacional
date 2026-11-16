"""Funcoes da analise exploratoria (usadas pelo notebook analise_exploratoria.ipynb).

A leitura dos dados (list_images, load_image, load_masks) fica em pipeline/data.py.

Termos: cada mascara e uma INSTANCIA (uma origem e todas as copias dela); cada mancha separada
dentro dela e um PEDACO (componente conexo). Em geral uma instancia tem dois pedacos.
"""
import cv2
import numpy as np

from pipeline.data import list_images, load_image, load_masks  # noqa: F401 (usadas pelo notebook)


# ---------------------------------------------------------------- estatisticas por imagem

def pieces(mask):
    """Pedacos (componentes conexos) de uma mascara: lista de (mascara do pedaco, area, lado menor da caixa)."""
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [(labels == k, st[k, cv2.CC_STAT_AREA], min(st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]))
            for k in range(1, n)]


def image_stats(row):
    """Tamanho, cor, instancias, pedacos e area relativa de uma imagem."""
    img = load_image(row["path"])
    h, w = img.shape[:2]
    gray_img = bool(np.all(img[..., 0] == img[..., 1]) and np.all(img[..., 1] == img[..., 2]))
    stats = {"altura": h, "largura": w, "megapixels": h * w / 1e6, "proporcao": w / h,
             "cinza": gray_img, "n_paineis": count_panels(img),
             "n_instancias": 0, "area_uniao": 0.0, "areas_instancias": [], "pedacos_por_instancia": [],
             "areas_pedacos": [], "lado_min_pedaco_px": np.nan}
    if row["classe"] == "forged":
        masks = load_masks(row["mask_path"], (h, w))
        per_mask = [pieces(m) for m in masks]
        all_pieces = [p for ps in per_mask for p in ps]
        stats.update(n_instancias=len(masks), area_uniao=masks.any(0).mean(),
                     areas_instancias=list(masks.reshape(len(masks), -1).mean(1)),
                     pedacos_por_instancia=[len(ps) for ps in per_mask],
                     areas_pedacos=[area / (h * w) for _, area, _ in all_pieces],
                     lado_min_pedaco_px=min(side for _, _, side in all_pieces) if all_pieces else np.nan)
    return stats


def count_panels(img, bg_tol=12, min_frac=0.05):
    """Estimativa grosseira do numero de paineis de uma figura composta.

    Linhas (e colunas) quase inteiras da cor do fundo sao separadores. Cada trecho continuo de
    nao-separadores com pelo menos min_frac da imagem conta como uma faixa de paineis.
    Paineis ~ faixas horizontais x faixas verticais.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(int)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = np.median(border)                                  # cor de fundo = mediana da borda
    is_bg = np.abs(gray - bg) <= bg_tol

    def bands(sep):                                         # sep[i] = True se a linha i e separador
        runs, n = [], 0
        for s in sep:
            if s:
                runs.append(n); n = 0
            else:
                n += 1
        runs.append(n)
        return max(1, sum(r >= min_frac * len(sep) for r in runs))

    return bands(is_bg.mean(1) > 0.98) * bands(is_bg.mean(0) > 0.98)


# ---------------------------------------------------------------- transformacao entre as copias

DIHEDRAL = {                                                # as 8 orientacoes possiveis de uma copia
    "identidade": lambda x: x,
    "espelho horizontal": lambda x: x[:, ::-1],
    "espelho vertical": lambda x: x[::-1],
    "rotacao 90": lambda x: np.rot90(x, 1),
    "rotacao 180": lambda x: np.rot90(x, 2),
    "rotacao 270": lambda x: np.rot90(x, 3),
    "transposta": lambda x: x.T,
    "anti-transposta": lambda x: np.rot90(x, 2).T,
}


def copy_transform(img, mask):
    """Para uma instancia: qual das 8 orientacoes melhor leva um pedaco no outro (correlacao entre
    os recortes dos dois maiores pedacos) e a razao de escala entre eles. None se ha menos de 2 pedacos."""
    ps = sorted(pieces(mask), key=lambda p: -p[1])[:2]
    if len(ps) < 2:
        return None
    pair = ps[0][0], ps[1][0]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    def crop(m):
        ys, xs = np.nonzero(m)
        return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    a, b = crop(pair[0]), crop(pair[1])
    if min(a.shape + b.shape) < 4:
        return None
    corr = {}
    for name, t in DIHEDRAL.items():
        tb = cv2.resize(np.ascontiguousarray(t(b)), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        corr[name] = float(np.corrcoef(a.ravel(), tb.ravel())[0, 1]) if a.std() > 0 and tb.std() > 0 else np.nan
    best = max(corr, key=lambda k: -np.inf if np.isnan(corr[k]) else corr[k])
    return {"melhor_orientacao": best, "corr_melhor": corr[best], "corr_identidade": corr["identidade"],
            "escala": float(np.sqrt(pair[0].sum() / pair[1].sum()))}


# ---------------------------------------------------------------- forjada x autentica do mesmo case_id

def diff_vs_authentic(forged_path, authentic_path, masks, thr=10):
    """Onde a forjada difere da autentica de mesmo case_id, comparado com a mascara.
    por_pedaco_alterado: fracao alterada de cada pedaco (~0 = origem intacta, ~1 = destino colado)."""
    f, a = load_image(forged_path), load_image(authentic_path)
    if f.shape != a.shape:
        return None
    changed = np.abs(f.astype(int) - a.astype(int)).max(2) > thr
    union = masks.any(0)
    return {"frac_mascara_alterada": changed[union].mean() if union.any() else np.nan,
            "frac_alterado_fora_mascara": changed[~union].sum() / max(changed.sum(), 1),
            "por_pedaco_alterado": [changed[p].mean() for m in masks for p, _, _ in pieces(m)]}


def instance_roles(forged_path, authentic_path, masks, thr=10):
    """Papel dos pedacos de cada instancia, comparando a forjada com a autentica do mesmo case_id.

    Pedaco quase igual a autentica (<10% dos pixels alterados) = origem; quase todo alterado (>90%) = destino.
    Devolve uma linha por instancia, ou None se as duas imagens tem tamanhos diferentes.
    """
    f, a = load_image(forged_path), load_image(authentic_path)
    if f.shape != a.shape:
        return None
    changed = np.abs(f.astype(int) - a.astype(int)).max(2) > thr
    rows = []
    for k, m in enumerate(masks):
        ps = pieces(m)
        altered = [changed[p].mean() for p, _, _ in ps]
        areas = [area for _, area, _ in ps]
        rows.append({"instancia": k, "n_pedacos": len(ps),
                     "origens": sum(x < 0.1 for x in altered), "destinos": sum(x > 0.9 for x in altered),
                     "frac_alterada": changed[m].mean(),
                     "menor_sobre_maior": min(areas) / max(areas) if areas else np.nan})
    return rows


def roles_overlay(forged, authentic, mask, thr=10, alpha=0.55):
    """Pinta a instancia: azul onde a forjada e igual a autentica (origem), vermelho onde mudou (destino)."""
    changed = np.abs(forged.astype(int) - authentic.astype(int)).max(2) > thr
    out = forged.astype(float).copy()
    out[mask & ~changed] = (1 - alpha) * out[mask & ~changed] + alpha * np.array([0, 120, 255])
    out[mask & changed] = (1 - alpha) * out[mask & changed] + alpha * np.array([230, 30, 30])
    return out.astype(np.uint8)


# ---------------------------------------------------------------- autocorrespondencia SIFT

def sift_self_matches(img, masks=None, max_side=1024, ratio=0.6, min_dist=20):
    """Pares de pontos-chave SIFT da imagem que casam com outro ponto da propria imagem.

    Para cada ponto, o 1o vizinho e ele mesmo; usa o 2o e o 3o no teste da razao de Lowe e
    descarta pares muito proximos (vizinhanca do mesmo objeto).
    Devolve (n_pontos, n_pares, fracao dos pares com as duas pontas dentro da mascara).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s = min(1.0, max_side / max(gray.shape))
    gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    kps, des = cv2.SIFT_create().detectAndCompute(gray, None)
    if des is None or len(kps) < 3:
        return len(kps), 0, np.nan
    pairs = []
    for m in cv2.BFMatcher().knnMatch(des, des, k=3):
        if len(m) == 3 and m[1].distance < ratio * m[2].distance:
            p, q = np.array(kps[m[1].queryIdx].pt), np.array(kps[m[1].trainIdx].pt)
            if np.linalg.norm(p - q) > min_dist:
                pairs.append((p / s, q / s))
    inside = np.nan
    if masks is not None and pairs:
        union = masks.any(0)
        h, w = union.shape
        hit = lambda pt: union[min(int(pt[1]), h - 1), min(int(pt[0]), w - 1)]
        inside = float(np.mean([hit(p) and hit(q) for p, q in pairs]))
    return len(kps), len(pairs), inside


# ---------------------------------------------------------------- visualizacao

COLORS = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240]])


def overlay(img, masks, alpha=0.45):
    """Imagem com cada instancia (mascara) pintada de uma cor e o contorno destacado."""
    out = img.astype(float).copy()
    for i, m in enumerate(masks):
        c = COLORS[i % len(COLORS)]
        out[m] = (1 - alpha) * out[m] + alpha * c
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = cv2.drawContours(out, contours, -1, c.tolist(), max(1, max(img.shape[:2]) // 300))
    return out.astype(np.uint8)
