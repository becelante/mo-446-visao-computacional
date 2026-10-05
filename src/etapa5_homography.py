"""
Etapa 5 - Estimacao de Homografia e Alinhamento.
"""
import numpy as np
import cv2


def estimate_homography(kps1, kps2, matches, ransac_thresh=4.0):
    """
    Estima H (que leva pontos de img1 para o referencial de img2) via RANSAC
    (Etapa 5.1). Retorna H, mascara de inliers e metricas (taxa de inliers,
    erro de reprojecao medio) pedidas na Etapa 5.3.
    """
    if len(matches) < 4:
        return None, None, {"inlier_ratio": 0.0, "mean_reproj_error": None,
                             "n_inliers": 0, "n_matches": len(matches)}

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None, None, {"inlier_ratio": 0.0, "mean_reproj_error": None,
                             "n_inliers": 0, "n_matches": len(matches)}

    mask = mask.ravel().astype(bool)
    inlier_ratio = mask.sum() / len(matches)

    pts1_in = pts1[mask]
    pts2_in = pts2[mask]
    if len(pts1_in) > 0:
        pts1_proj = cv2.perspectiveTransform(pts1_in, H)
        errors = np.linalg.norm(pts1_proj - pts2_in, axis=2).ravel()
        mean_error = float(errors.mean())
    else:
        mean_error = None

    metrics = {
        "inlier_ratio": float(inlier_ratio),
        "mean_reproj_error": mean_error,
        "n_inliers": int(mask.sum()),
        "n_matches": len(matches),
    }
    return H, mask, metrics


def _get_H(a, b, homographies):
    """Retorna a homografia que mapeia a imagem `a` para o referencial de `b`,
    reaproveitando homographies[(a,b)] (a->b) ou invertendo homographies[(b,a)] (b->a)."""
    if (a, b) in homographies:
        return homographies[(a, b)]
    if (b, a) in homographies:
        return np.linalg.inv(homographies[(b, a)])
    return None


def compose_pairwise_homographies(order, homographies, ref_index=None):
    """
    Compoe homografias par-a-par para levar todas as imagens ao referencial
    de uma imagem de referencia (por padrao, a do meio da sequencia
    ordenada). Encadear transformacoes par-a-par e uma simplificacao do
    bundle adjustment global (ver Extra X1 do enunciado): erros pequenos em
    cada par se acumulam ao longo da cadeia.

    `homographies` e um dict {(i, j): H_i_para_j} vindo de
    ordering.build_connectivity_matrix. Retorna dict {indice_original: H_para_referencia}.
    """
    if not order:
        return {}
    if ref_index is None:
        ref_index = order[len(order) // 2]

    ref_pos = order.index(ref_index)
    H_to_ref = {ref_index: np.eye(3)}

    # propaga para a direita a partir da referencia
    for k in range(ref_pos, len(order) - 1):
        i, j = order[k], order[k + 1]
        if i not in H_to_ref:
            break  # cadeia quebrada (par sem homografia valida)
        H_i_to_j = _get_H(i, j, homographies)
        if H_i_to_j is None:
            continue
        H_to_ref[j] = H_to_ref[i] @ np.linalg.inv(H_i_to_j)

    # propaga para a esquerda a partir da referencia
    for k in range(ref_pos, 0, -1):
        i, j = order[k], order[k - 1]
        if i not in H_to_ref:
            break
        H_i_to_j = _get_H(i, j, homographies)
        if H_i_to_j is None:
            continue
        H_to_ref[j] = H_to_ref[i] @ np.linalg.inv(H_i_to_j)

    return H_to_ref
