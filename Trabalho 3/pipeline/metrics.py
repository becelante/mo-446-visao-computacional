"""Metricas do relatorio, calculadas com a metrica oficial (recodai_f1.py, sem modificacoes).

solution:   DataFrame case_id, annotation, shape   (verdade; shape = "[H, W]")
submission: DataFrame case_id, annotation          (predicao)
annotation = "authentic" ou as mascaras em RLE separadas por ';'.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from pipeline import recodai_f1


def masks_to_annotation(masks):
    """Lista de mascaras (H, W) com 0/1 -> anotacao oficial."""
    masks = [m.astype(np.uint8) for m in masks if m.any()]
    return recodai_f1.rle_encode(masks) if masks else "authentic"


def annotation_to_masks(annotation, shape):
    """Anotacao oficial -> lista de mascaras (H, W)."""
    if annotation == "authentic":
        return []
    return [recodai_f1.rle_decode(rle, shape) for rle in annotation.split(";")]


def image_score(label, pred, shape):
    """Escore de uma imagem: 1 se ambas authentic, 0 se so uma, senao oF1 oficial."""
    if label == "authentic" or pred == "authentic":
        return float(label == pred)
    return recodai_f1.evaluate_single_image(label, pred, shape)


def evaluate(solution, submission):
    """Devolve (dicionario de metricas, escore por imagem). Imagem sem predicao conta como authentic."""
    solution = solution.reset_index(drop=True)
    pred = solution[["case_id"]].merge(submission, on="case_id", how="left").fillna("authentic")

    scores = np.array([image_score(l, p, s) for l, p, s in
                       zip(solution["annotation"], pred["annotation"], solution["shape"])])
    is_forged = (solution["annotation"] != "authentic").to_numpy()     # verdade
    said_forged = (pred["annotation"] != "authentic").to_numpy()       # predicao
    detected = is_forged & said_forged
    all_authentic = solution[["case_id"]].assign(annotation="authentic")
    tn, fp, fn, tp = confusion_matrix(is_forged, said_forged, labels=[False, True]).ravel()

    metrics = {
        "S": recodai_f1.score(solution.copy(), pred, "case_id"),
        "S_baseline_authentic": recodai_f1.score(solution.copy(), all_authentic, "case_id"),
        "acuracia": accuracy_score(is_forged, said_forged),
        "precisao": precision_score(is_forged, said_forged, zero_division=0),
        "revocacao": recall_score(is_forged, said_forged, zero_division=0),
        "f1_deteccao": f1_score(is_forged, said_forged, zero_division=0),
        "matriz_confusao": {"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)},
        "oF1_forjadas_detectadas": scores[detected].mean() if detected.any() else float("nan"),
        "n_autenticas": int((~is_forged).sum()),
        "n_forjadas": int(is_forged.sum()),
        "n_forjadas_detectadas": int(detected.sum()),
    }
    metrics = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}
    return metrics, solution[["case_id"]].assign(forjada=is_forged, prevista_forjada=said_forged, escore=scores)


def report(m):
    cm = m["matriz_confusao"]
    return f"""\
S (metrica oficial):        {m['S']:.4f}
S baseline 'authentic':     {m['S_baseline_authentic']:.4f}   (ganho {m['S'] - m['S_baseline_authentic']:+.4f})
Deteccao ({m['n_autenticas']} autenticas, {m['n_forjadas']} forjadas):
  acuracia {m['acuracia']:.4f} | precisao {m['precisao']:.4f} | revocacao {m['revocacao']:.4f} | F1 {m['f1_deteccao']:.4f}
                   prevista autentica   prevista forjada
  autentica        {cm['VN']:>18}   {cm['FP']:>16}
  forjada          {cm['FN']:>18}   {cm['VP']:>16}
Localizacao: oF1 medio nas {m['n_forjadas_detectadas']} forjadas detectadas = {m['oF1_forjadas_detectadas']:.4f}"""
