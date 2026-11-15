"""Testes: doctests da metrica oficial, ida e volta do RLE e o exemplo numerico do enunciado."""
import doctest

import numpy as np
import pandas as pd

from pipeline import metrics, recodai_f1


def rect(r0, r1, c0, c1, h=100, w=100):
    m = np.zeros((h, w), np.uint8)
    m[r0:r1, c0:c1] = 1
    return m


# 1. Metrica oficial: exemplos do proprio notebook do Kaggle
assert doctest.testmod(recodai_f1).failed == 0

# 2. RLE: mascara -> texto -> mascara devolve o mesmo
masks = [rect(5, 20, 10, 30), (np.random.default_rng(0).random((100, 100)) > 0.7).astype(np.uint8)]
back = metrics.annotation_to_masks(metrics.masks_to_annotation(masks), (100, 100))
assert all(np.array_equal(a, b) for a, b in zip(masks, back))

# 3. Quatro imagens como no enunciado: forjada com mascara extra, autentica certa,
#    autentica com mascara prevista e forjada prevista authentic
g1, g2 = rect(0, 50, 0, 50), rect(50, 100, 50, 100)
p1, p2, extra = rect(0, 50, 0, 40), rect(50, 100, 50, 90), rect(0, 5, 60, 70)
A = metrics.masks_to_annotation
solution = pd.DataFrame({"case_id": list("abcd"),
                         "annotation": [A([g1, g2]), "authentic", "authentic", A([g1])],
                         "shape": ["[100, 100]"] * 4})
submission = pd.DataFrame({"case_id": list("abcd"),
                           "annotation": [A([p1, p2, extra]), "authentic", A([extra]), "authentic"]})
m, per_image = metrics.evaluate(solution, submission)

f1 = 2 * 2000 / (2000 + 2500)            # p1 cobre 2000 dos 2500 pixels de g1, sem falso positivo
of1 = 2 * f1 / 3                         # soma dos F1 casados / max(3 previstas, 2 verdadeiras)
assert np.allclose(per_image["escore"], [of1, 1, 0, 0])
assert np.isclose(m["S"], (of1 + 1) / 4) and m["S_baseline_authentic"] == 0.5
assert m["matriz_confusao"] == {"VN": 1, "FP": 1, "FN": 1, "VP": 1}
assert np.isclose(m["oF1_forjadas_detectadas"], of1)
print(metrics.report(m))
print("\ntodos os testes passaram")
