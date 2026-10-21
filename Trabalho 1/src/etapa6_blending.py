"""
Etapa 6 - Composicao, Blending e Remocao de Fantasmas (obrigatoria).

Implementa:
  - composicao ingenua (media simples), para evidenciar os fantasmas
    (Figura 6, "composicao ingenua")
  - busca de costura otima via programacao dinamica na regiao de
    sobreposicao (equivalente simplificado da busca de costura citada na
    Etapa 6.3, que tende a desviar de objetos moveis inconsistentes entre
    as vistas)
  - feathering (blending linear) numa faixa estreita ao redor da costura,
    para suavizar a transicao (Etapa 6.2)
"""
import numpy as np
import cv2


def get_canvas_bounds(images, H_list):
    """Calcula o tamanho do canvas e a translacao necessaria para acomodar
    todas as imagens warpeadas no referencial comum, sem cortar nada."""
    all_corners = []
    for img, H in zip(images, H_list):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)
    all_corners = np.concatenate(all_corners, axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_size = (int(x_max - x_min), int(y_max - y_min))  # (largura, altura)
    return canvas_size, translation


def warp_to_canvas(img, H, translation, canvas_size):
    """Aplica H (para o referencial de composicao) + translacao, gerando a
    imagem e sua mascara de pixels validos no tamanho do canvas (Etapa 5.2)."""
    H_total = translation @ H
    warped = cv2.warpPerspective(img, H_total, canvas_size)
    mask = cv2.warpPerspective(
        np.full(img.shape[:2], 255, dtype=np.uint8), H_total, canvas_size
    )
    # a borda da imagem warpada e interpolada com o preto de fora; sem a erosao vira uma linha escura
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    return warped, mask


def naive_composite(warped_images, masks):
    """Etapa 6.1 (versao ingenua) - media simples nas regioes de sobreposicao.
    Serve de baseline para evidenciar os fantasmas."""
    h, w = masks[0].shape
    acc = np.zeros((h, w, 3), dtype=np.float64)
    count = np.zeros((h, w, 1), dtype=np.float64)
    for warped, mask in zip(warped_images, masks):
        m = (mask > 0).astype(np.float64)[..., None]
        acc += warped.astype(np.float64) * m
        count += m
    count[count == 0] = 1
    return (acc / count).astype(np.uint8)


def find_optimal_seam(cost_map, valid_mask):
    """
    Encontra, via programacao dinamica (estilo seam carving), o caminho de
    topo a base de custo minimo dentro da regiao de sobreposicao
    (`valid_mask`). Pixels fora da mascara recebem custo infinito. Retorna
    um array 1D com a posicao x (local, dentro do recorte) da costura para
    cada linha y.
    """
    h, w = cost_map.shape
    INF = 1e9
    cost = np.where(valid_mask, cost_map, INF).astype(np.float64)

    dp = cost.copy()
    backtrack = np.zeros((h, w), dtype=np.int8)

    for y in range(1, h):
        up = dp[y - 1]
        left = np.roll(up, 1)
        left[0] = INF
        right = np.roll(up, -1)
        right[-1] = INF
        # ordem [up, left, right]: em caso de empate, argmin prefere manter a
        # coluna (offset 0), evitando zigue-zague desnecessario em regioes de
        # custo uniforme e produzindo costuras mais suaves
        stacked = np.stack([up, left, right], axis=0)
        best_idx = np.argmin(stacked, axis=0)
        dp[y] = cost[y] + stacked[best_idx, np.arange(w)]
        offset_map = np.array([0, -1, 1])
        backtrack[y] = offset_map[best_idx]

    seam = np.zeros(h, dtype=int)
    seam[h - 1] = int(np.argmin(dp[h - 1]))
    for y in range(h - 2, -1, -1):
        seam[y] = seam[y + 1] + backtrack[y + 1, seam[y + 1]]
        seam[y] = int(np.clip(seam[y], 0, w - 1))

    return seam


def blend_pair_with_seam(base, base_mask, new_img, new_mask, feather_width=15):
    """
    Combina `base` (panorama acumulado ate agora) com `new_img` (proxima
    imagem warpeada), removendo fantasmas com costura otima na regiao de
    sobreposicao e suavizando a transicao com feathering linear ao redor
    da costura.
    """
    overlap = (base_mask > 0) & (new_mask > 0)
    only_new = (new_mask > 0) & (~overlap)

    result = base.copy()
    result[only_new] = new_img[only_new]

    if not overlap.any():
        return result, (base_mask | new_mask)

    ys, xs = np.where(overlap)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    base_roi = base[y0:y1, x0:x1].astype(np.float64)
    new_roi = new_img[y0:y1, x0:x1].astype(np.float64)
    overlap_roi = overlap[y0:y1, x0:x1]

    cost_map = np.sum((base_roi - new_roi) ** 2, axis=2)
    seam_x = find_optimal_seam(cost_map, overlap_roi)  # posicao x local da costura, por linha

    hloc, wloc = overlap_roi.shape
    xx = np.tile(np.arange(wloc), (hloc, 1))
    seam_col = seam_x.reshape(-1, 1)
    dist = xx - seam_col  # > 0 => lado da nova imagem; < 0 => lado da base
    weight = np.clip((dist + feather_width / 2) / feather_width, 0, 1)

    blended_roi = base_roi * (1 - weight[..., None]) + new_roi * weight[..., None]
    blended_roi = np.where(overlap_roi[..., None], blended_roi, base_roi)

    result[y0:y1, x0:x1][overlap_roi] = blended_roi[overlap_roi].astype(np.uint8)

    combined_mask = base_mask | new_mask
    return result, combined_mask


def compose_panorama(images, H_list, feather_width=15, remove_ghosts=True, snapshots=None,
                     snapshot_width=2400):
    """
    Pipeline completo da Etapa 6: calcula o canvas comum, warpa todas as
    imagens e as compoe sequencialmente. Retorna (panorama_final,
    panorama_ingenuo) para a comparacao lado a lado pedida na Etapa 6.4.
    Se `snapshots` e uma lista, recebe o mosaico (reduzido) apos cada imagem
    adicionada (alinhamento progressivo, Etapa 5.4).
    """
    canvas_size, translation = get_canvas_bounds(images, H_list)

    warped_list, mask_list = [], []
    for img, H in zip(images, H_list):
        warped, mask = warp_to_canvas(img, H, translation, canvas_size)
        warped_list.append(warped)
        mask_list.append(mask)

    naive = naive_composite(warped_list, mask_list)

    if not remove_ghosts:
        return naive.copy(), naive

    def snapshot(img):
        if snapshots is not None:
            f = snapshot_width / img.shape[1]
            snapshots.append(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))

    panorama = warped_list[0].copy()
    panorama_mask = mask_list[0].copy()
    snapshot(panorama)
    for warped, mask in zip(warped_list[1:], mask_list[1:]):
        panorama, panorama_mask = blend_pair_with_seam(
            panorama, panorama_mask, warped, mask, feather_width
        )
        snapshot(panorama)

    return panorama, naive


def ghost_comparison(naive, panorama, size=(700, 450)):
    """Etapa 6.4 - recorte da regiao onde a composicao ingenua mais difere da final
    (fantasmas), lado a lado: media simples (esquerda) e costura otima (direita)."""
    w, h = size
    valid = (naive.max(axis=2) > 0) & (panorama.max(axis=2) > 0)
    diff = np.abs(naive.astype(np.float32) - panorama.astype(np.float32)).mean(axis=2) * valid
    score = cv2.boxFilter(diff, -1, (w, h))
    score[cv2.erode(valid.astype(np.uint8), np.ones((h, w), np.uint8)) == 0] = -1
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    x0, y0 = cx - w // 2, cy - h // 2
    gap = np.full((h, 12, 3), 255, np.uint8)
    return np.hstack([naive[y0:y0 + h, x0:x0 + w], gap, panorama[y0:y0 + h, x0:x0 + w]])


def overlap_metrics(images, H_list, names, thresh=30):
    """Etapa 6.5 - para cada par consecutivo ja alinhado: diferenca absoluta media de
    cor na sobreposicao (0-255), fracao de pixels com diferenca acima de `thresh` e
    razao entre os brilhos medios das duas imagens."""
    canvas_size, translation = get_canvas_bounds(images, H_list)
    warped = [warp_to_canvas(img, H, translation, canvas_size) for img, H in zip(images, H_list)]
    rows = []
    for k in range(len(images) - 1):
        (a, ma), (b, mb) = warped[k], warped[k + 1]
        overlap = cv2.erode(((ma > 0) & (mb > 0)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)[overlap]
        la = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)[overlap].astype(np.float32).mean()
        lb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)[overlap].astype(np.float32).mean()
        rows.append({"imagem_1": names[k], "imagem_2": names[k + 1],
                     "diferenca_media": round(float(d.mean()), 2),
                     "fracao_inconsistente": round(float((d > thresh).mean()), 4),
                     "razao_brilho": round(float(max(la, lb) / min(la, lb)), 4)})
    return rows
