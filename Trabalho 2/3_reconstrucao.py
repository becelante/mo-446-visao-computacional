"""
Etapas 2 a 5 no video da Variante A, em results/<detector>/ (--detector SIFT ou ORB):
  - geometria epipolar: matches antes/depois do RANSAC e linhas epipolares de dois pares
    (um sequencial e o fechamento de laco com mais inliers), em results/epipolar/;
  - SfM, escala, limpeza, alinhamento ao chao e planta baixa (esparsa*.ply, cameras.ply,
    trajetoria.ply, planta_baixa.png) e as metricas (metricas.json, metricas.md);
  - com --densa, a nuvem densa por plane-sweep (densa.ply) e a densidade dela.
"""
import argparse

import open3d as o3d

from src import config
from src.pipeline import load_features, reconstruct, save_outputs, unpack_matches, write_metrics
from src.step_0_frames import load_frames
from src.step_2_matching import draw_matches_before_after, knn_match
from src.step_3_epipolar import draw_epipolar_lines, matched_points
from src.step_4_dense import dense_reconstruction
from src.step_4_postprocessing import point_density


def epipolar_figures(paths, images, result, out_dir, method):
    """Um par sequencial tipico (inliers medianos) e o par de laco (mais distante no video com >= 100 inliers)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    geoms = {p: g for p, g in result["geometries"].items() if g["F"] is not None and g["mask_F"].sum() >= 30}
    seq = sorted((g["metrics"]["inliers_E"], p) for p, g in geoms.items() if p[1] - p[0] <= 3)
    chosen = {"sequencial": seq[len(seq) // 2][1]} if seq else {}
    loops = [(p[1] - p[0], p) for p, g in geoms.items() if g["metrics"]["inliers_E"] >= 100]
    if loops:
        chosen["laco"] = max(loops)[1]
    all_kps = result["all_kps"]
    _, all_descs = load_features(paths, images, method)
    for name, (i, j) in chosen.items():
        g = geoms[(i, j)]
        raw = [m for m, _ in knn_match(all_descs[i], all_descs[j], method, config.device())]
        prefix = str(out_dir / f"{name}_{paths[i].stem}_{paths[j].stem}")
        good = unpack_matches(g["good"])
        draw_matches_before_after(images[i], all_kps[i], images[j], all_kps[j], raw, good, g["mask_E"], prefix)
        pts1, pts2 = matched_points(all_kps[i], all_kps[j], good)
        draw_epipolar_lines(images[i], images[j], pts1[g["mask_F"]], pts2[g["mask_F"]], g["F"],
                            f"{prefix}_epipolares.jpg")
        print(f"Figuras do par {name} {paths[i].stem}-{paths[j].stem}: {g['metrics']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--densa", action="store_true", help="tambem gera a nuvem densa (MVS)")
    parser.add_argument("--detector", default=config.DETECTOR, help="SIFT ou ORB (saidas em results/<detector>/)")
    args = parser.parse_args()

    paths, images = load_frames(config.frames_dir())
    out = config.results_dir(args.detector)
    result = reconstruct(paths, images, method=args.detector)
    paths, images = result["paths"], result["images"]  # os quadros-chave, na indexacao da reconstrucao
    height, width = images[0].shape[:2]
    save_outputs(result, images, out, width, height)
    epipolar_figures(paths, images, result, out / "epipolar", args.detector)

    if args.densa:
        dense, info = dense_reconstruction(result["recon"], images, **config.DENSE, device=config.device())
        dense, _ = dense.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        o3d.io.write_point_cloud(str(out / "densa.ply"), dense)
        metrics = result["metrics"]
        ceilings = [h["height"] for h in metrics["planes"]["horizontal"] if h["height"] > 2.0]
        ceiling = max(ceilings) if ceilings else None
        metrics["dense"] = {**info, "n_points_filtered": len(dense.points),
                            **point_density(len(dense.points), result["dims"], ceiling)}
        write_metrics(metrics, out)
    print((out / "metricas.md").read_text())
