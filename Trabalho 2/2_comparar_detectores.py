"""
Etapa 2: compara os detectores/descritores de config.DETECTORS_COMPARED em results/deteccao/:
  - keypoints desenhados em dois quadros por detector;
  - comparacao.csv / comparacao.md: keypoints por quadro, tempo de deteccao, matches e inliers
    de E (RANSAC) em pares a 1 e a 4 quadros de distancia, espalhados pelo video.
"""
import csv
import time

import numpy as np
import torch

from src import config
from src.step_0_frames import load_frames
from src.step_1_keypoints_detector import detect_features_all, draw_keypoints
from src.step_2_matching import Matcher
from src.step_3_epipolar import build_intrinsics, two_view_geometry

N_SAMPLES = 30   # pares de partida espalhados pelo video
GAPS = (1, 4)    # distancia (em quadros) entre as duas imagens do par


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    paths, images = load_frames(config.frames_dir())
    out = config.video_results() / "deteccao"
    out.mkdir(parents=True, exist_ok=True)
    starts = np.linspace(0, len(paths) - 1 - max(GAPS), N_SAMPLES).astype(int)
    used = sorted({s + g for s in starts for g in (0, *GAPS)})
    local = {k: n for n, k in enumerate(used)}
    sample = [images[k] for k in used]
    height, width = images[0].shape[:2]
    K = build_intrinsics(width, height)

    rows = []
    for method in config.DETECTORS_COMPARED:
        detect_features_all(sample[:1], method, config.EQUALIZE, config.MAX_KEYPOINTS, config.SIFT_CONTRAST,
                            config.device())  # aquece (carrega a rede / inicializa CUDA)
        sync()
        t0 = time.perf_counter()
        kps, descs = detect_features_all(sample, method, config.EQUALIZE, config.MAX_KEYPOINTS,
                                         config.SIFT_CONTRAST, config.device())
        sync()
        t_detect = (time.perf_counter() - t0) / len(sample)
        for k in (starts[len(starts) // 3], starts[2 * len(starts) // 3]):
            draw_keypoints(images[k], kps[local[k]], str(out / f"{method}_{paths[k].stem}.jpg"))

        matcher = Matcher(kps, descs, method, (width, height), config.RATIO_THRESH, config.CROSS_CHECK,
                          device=config.device())
        row = {"detector": method, "keypoints_por_quadro": float(np.mean([len(k) for k in kps])),
               "deteccao_ms": 1000 * t_detect}
        for gap in GAPS:
            matches, inliers, ratios, t_match = [], [], [], 0.0
            for s in starts:
                i, j = local[s], local[s + gap]
                sync()
                t0 = time.perf_counter()
                good = matcher(i, j)
                sync()
                t_match += time.perf_counter() - t0
                g = two_view_geometry(kps[i], kps[j], good, K, config.RANSAC_THRESH, config.MIN_MATCHES)
                matches.append(len(good))
                inliers.append(g["metrics"]["inliers_E"])
                ratios.append(g["metrics"]["inliers_E"] / len(good) if good else 0.0)
            row.update({f"matches_d{gap}": float(np.median(matches)), f"inliers_d{gap}": float(np.median(inliers)),
                        f"taxa_inliers_d{gap}": float(np.median(ratios)),
                        f"pares_ok_d{gap}": int(np.sum(np.array(inliers) >= 50)),
                        f"matching_ms_d{gap}": 1000 * t_match / len(starts)})
        rows.append(row)
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()})
        del matcher
        torch.cuda.empty_cache()

    with open(out / "comparacao.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(f"{r[h]:.2f}" if isinstance(r[h], float) else str(r[h]) for h in header) + " |"
              for r in rows]
    (out / "comparacao.md").write_text(
        f"Medianas sobre {N_SAMPLES} pares por distancia; pares_ok = pares com >= 50 inliers de E.\n\n"
        + "\n".join(lines) + "\n")
    print((out / "comparacao.md").read_text())
